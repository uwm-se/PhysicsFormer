"""Smoke test for Phase 5 free-form QA mode in PhysicsReasoningDataset.

Verifies that:
  * the new ``freeform_qa_data_path`` constructor short-circuit bypasses
    the synthetic generator + CLEVRER MCQ mix entirely
  * physics_dataset=None is accepted in this mode
  * the per-scene state cache builds and persists correctly
  * each emitted record has ``choices=None`` (key contract for V3 Format A)
  * the existing ``collate_fn`` consumes the records cleanly
  * records pass the same shape/key contract as the legacy CLEVRER MCQ records
  * QA records with missing scenes / blank fields are dropped, not crashed on

This does NOT load the real physics_former encoder or run training; it
exercises only the data-pipeline glue. A passing run gives confidence
to commit a Colab training run.

Usage:
    python physics_llm_adapter/tests/test_freeform_qa_dataset.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Make the package importable when run from any CWD inside the repo.
_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
sys.path.insert(0, str(_PKG))
# physics_former layout is referenced by train_adapter_v2.py; mirror that here.
sys.path.insert(0, str(_PKG.parent / 'physics_former'))


# ─── Fake scene_converter so we can run without $CLEVRER_DIR mounted ────────
# train_adapter_v2.py does `from scene_converter import clevrer_scene_to_state_tensor`.
# We pre-install a stub module under that import name BEFORE importing the
# training module, so the smoke test stays self-contained.
import types as _types

_fake_scene_converter = _types.ModuleType('scene_converter')

def _fake_clevrer_scene_to_state_tensor(scene):
    """Return synthetic [T, N, D] states from a small JSON-like dict."""
    T = scene.get('seq_len', 32)   # tiny by default for fast tests
    N = scene.get('num_objects', 3)
    D = 35
    states = np.random.RandomState(scene.get('seed', 0)).randn(T, N, D).astype(np.float32)
    masks = np.ones((T, N), dtype=np.float32)
    return states, masks, {}

_fake_scene_converter.clevrer_scene_to_state_tensor = _fake_clevrer_scene_to_state_tensor
sys.modules['scene_converter'] = _fake_scene_converter


# Now we can import the patched training module.
from train_adapter_v2 import PhysicsReasoningDataset, collate_fn  # noqa: E402


# ─── Fixtures ─────────────────────────────────────────────────────────────

def _write_records(tmp: Path, n_scenes: int, n_qa_per_scene: int = 3):
    """Create a synthetic QA JSON + scene JSON files matching the schema."""
    scenes_dir = tmp / 'scenes'
    scenes_dir.mkdir()
    qa_path = tmp / 'qa.json'

    # Write small scene JSONs (real loader would use clevrer_scene_to_state_tensor;
    # our stub above ignores the contents, but we still want files on disk so
    # the cache builder's existence checks pass).
    scene_indices = list(range(10000, 10000 + n_scenes))
    for sid in scene_indices:
        scene_path = scenes_dir / f'annotation_{sid:05d}.json'
        scene_path.write_text(json.dumps({
            'scene_index': sid, 'seq_len': 32, 'num_objects': 3, 'seed': sid,
        }))

    qa_records = []
    qa_types_cycle = ['predictive', 'counterfactual', 'explanatory', 'interventional']
    for sid in scene_indices:
        for j in range(n_qa_per_scene):
            qa_records.append({
                'qa_type': qa_types_cycle[(sid + j) % len(qa_types_cycle)],
                'scene_index': sid,
                'scene_path': str(scenes_dir / f'annotation_{sid:05d}.json'),
                'scene_desc': f'A scene with {j + 1} active objects.',
                'question': f'What will happen next in scene {sid}?',
                'target': f'The objects collide (synthetic answer #{j}).',
                'prompt': '',
            })
    # Add one malformed record (blank question) -- should be dropped, not crash.
    qa_records.append({
        'qa_type': 'predictive', 'scene_index': scene_indices[0],
        'scene_path': str(scenes_dir / f'annotation_{scene_indices[0]:05d}.json'),
        'question': '', 'target': 'should be dropped',
    })
    # And one referencing a non-existent scene_index -- dropped by state lookup.
    qa_records.append({
        'qa_type': 'predictive', 'scene_index': 999999,
        'scene_path': str(scenes_dir / 'annotation_99999.json'),
        'question': 'invalid', 'target': 'should be dropped',
    })

    with open(qa_path, 'w', encoding='utf-8') as f:
        json.dump(qa_records, f)
    return qa_path, scenes_dir, scene_indices


# ─── Tests ────────────────────────────────────────────────────────────────

def test_freeform_mode_short_circuits():
    """physics_dataset=None is OK in freeform mode; synthetic generator is skipped."""
    print('[1] freeform-mode short-circuit (physics_dataset=None) ...')
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        qa_path, scenes_dir, sids = _write_records(tmp, n_scenes=4)

        ds = PhysicsReasoningDataset(
            physics_dataset=None,                       # explicitly None
            freeform_qa_data_path=str(qa_path),
            freeform_qa_scenes_dir=str(scenes_dir),
            freeform_qa_cache_path=str(tmp / 'state_cache.pt'),
            freeform_max_objects=4,
            freeform_seq_len=32,
        )
        # 4 scenes * 3 records = 12 valid; plus 1 blank-Q + 1 missing-scene -> dropped
        assert len(ds) == 12, f'expected 12 valid records, got {len(ds)}'
        assert (tmp / 'state_cache.pt').exists(), 'state cache should be persisted'

        # No CLEVRER MCQ should have been loaded.
        assert ds.clevrer_samples == [], \
            'CLEVRER MCQ samples should NOT load in freeform mode'

        # All records use Format-A flavor (choices=None).
        for r in ds.qa_pairs:
            assert r['choices'] is None, \
                f'every freeform record must have choices=None, got {r["choices"]!r}'
            assert r['correct_choice_idx'] is None
    print('    PASS')


def test_record_shape_contract():
    """Every record matches the keys/shapes that collate_fn expects."""
    print('[2] record shape contract (collate_fn compatibility) ...')
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        qa_path, scenes_dir, sids = _write_records(tmp, n_scenes=3)

        ds = PhysicsReasoningDataset(
            freeform_qa_data_path=str(qa_path),
            freeform_qa_scenes_dir=str(scenes_dir),
            freeform_qa_cache_path=str(tmp / 'cache.pt'),
            freeform_max_objects=4,
            freeform_seq_len=32,
        )
        r = ds[0]
        # Required keys
        for k in ['states', 'mask', 'question', 'answer', 'choices',
                  'correct_choice_idx', 'question_type', 'metadata',
                  'numerical_targets']:
            assert k in r, f'missing key {k!r} in record'

        # Tensor shapes after padding to (seq_len=32, max_objects=4, state_dim=35)
        assert r['states'].shape == (32, 4, 35), \
            f'states shape {tuple(r["states"].shape)} != (32, 4, 35)'
        assert r['mask'].shape == (32, 4), \
            f'mask shape {tuple(r["mask"].shape)} != (32, 4)'
        assert r['states'].dtype == torch.float32
        assert r['mask'].dtype == torch.float32

        # Stringy / dict fields
        assert isinstance(r['question'], str) and r['question']
        assert isinstance(r['answer'], str) and r['answer']
        assert isinstance(r['question_type'], str)
        assert isinstance(r['metadata'], dict)
        assert r['metadata']['qa_source'] == 'free_form_causal_qa'

        # Numerical targets dict has the same 6 keys as legacy CLEVRER records.
        nt = r['numerical_targets']
        for k in ['distance', 'speed', 'time_to_collision', 'kinetic_energy',
                  'momentum', 'object_count']:
            assert k in nt, f'missing numerical target key {k!r}'

        # collate_fn must consume a batch without crashing.
        batch = collate_fn([ds[0], ds[1], ds[2]])
        B = 3
        assert batch['states'].shape == (B, 32, 4, 35)
        assert batch['masks'].shape == (B, 4)
        assert len(batch['questions']) == B
        assert len(batch['answers']) == B
        # In freeform mode, every sample has choices=None ->
        # any_valid=False -> batch-level choices=None.
        assert batch['choices'] is None, \
            'freeform-only batch must collate to choices=None'
    print('    PASS')


def test_state_cache_persistence_and_reuse():
    """The state cache is written, then re-read on a second instantiation."""
    print('[3] state cache: persist on first build, reload on second ...')
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        qa_path, scenes_dir, sids = _write_records(tmp, n_scenes=3)
        cache_path = tmp / 'cache.pt'

        # First build
        ds_a = PhysicsReasoningDataset(
            freeform_qa_data_path=str(qa_path),
            freeform_qa_scenes_dir=str(scenes_dir),
            freeform_qa_cache_path=str(cache_path),
            freeform_max_objects=4, freeform_seq_len=32,
        )
        n_a = len(ds_a)
        assert cache_path.exists()
        first_mtime = cache_path.stat().st_mtime

        # Second build -- should reload, NOT rewrite
        ds_b = PhysicsReasoningDataset(
            freeform_qa_data_path=str(qa_path),
            freeform_qa_scenes_dir=str(scenes_dir),
            freeform_qa_cache_path=str(cache_path),
            freeform_max_objects=4, freeform_seq_len=32,
        )
        n_b = len(ds_b)
        second_mtime = cache_path.stat().st_mtime
        assert n_a == n_b, f'two builds yielded different sizes ({n_a} vs {n_b})'
        assert first_mtime == second_mtime, \
            'state cache should NOT be rewritten on the second instantiation'
    print('    PASS')


def test_legacy_mcq_path_still_requires_physics_dataset():
    """Ensure the non-freeform branch still validates physics_dataset."""
    print('[4] legacy MCQ path raises when physics_dataset is omitted ...')
    try:
        PhysicsReasoningDataset()  # no freeform path, no physics_dataset
    except ValueError as e:
        assert 'physics_dataset' in str(e)
        print('    PASS  (raised as expected: ' + str(e)[:60] + '...)')
        return
    raise AssertionError('expected ValueError when both physics_dataset and freeform_qa_data_path are missing')


if __name__ == '__main__':
    print('Phase 5 free-form QA dataset smoke tests:\n')
    test_freeform_mode_short_circuits()
    test_record_shape_contract()
    test_state_cache_persistence_and_reuse()
    test_legacy_mcq_path_still_requires_physics_dataset()
    print('\nAll smoke tests passed.')
