"""
Phase 9 diverse-QA dataset generator (Lever 1).

Walks the CLEVRER training-scene partition (indices 10000-14498) and
emits a JSON file of free-form QA records using
``PhysicsQAGenerator(state_schema='clevrer')``. The record schema
matches ``causal_qa_dataset.json`` byte-for-byte so the existing
``_load_freeform_qa_data`` ingester in
``physics_llm_adapter/train_adapter_v2.py`` can consume the output
without modification.

Why a sibling script and not an extension of ``generate_qa_dataset.py``:
the legacy generator reads physics-former HDF5s in 28-D legacy schema
and writes HDF5 output. Phase 9 reads CLEVRER scene JSONs in 35-D
schema and writes JSON output; the I/O paths share nothing. Forcing
both flows into one CLI would obscure both.

Usage
-----
::

    python compsac_2026_code/data_generation/qa_generation/generate_phase9_diverse_qa.py \
        --clevrer_scenes_dir $CLEVRER_DIR/scenes/clevrer_scenes \
        --output compsac_2026_code/data/phase9/physics_diverse_qa_dataset.json \
        --records_per_scene 10 \
        --seed 42

Outputs a JSON list of records identical in shape to
``causal_qa_dataset.json``::

    [{"qa_type": "object_velocity",
      "scene_desc": "annotation_10000",
      "question": "How many objects are currently moving?",
      "target": "3",
      "prompt": "How many objects are currently moving?",
      "scene_index": 10000,
      "scene_path": "$CLEVRER_DIR/scenes/clevrer_scenes/annotation_10000.json"},
     ...]

The held-out question types (``KINETIC_ENERGY``, ``COLLISION_PREDICTION``,
``SPEED_COMPARISON``, ``MASS_COMPARISON``, ``TIME_TO_EVENT``) are
EXCLUDED -- they are reserved for the held-out-type generalization
eval. See ``physics_llm_adapter/phase9_splits.py``.

Phase 10 (Lever 4 Fix 4) adds ``--include_answer_cue {none|all|mix}``::

  none : Phase 9 default. Question text is the bare generator output;
         no cue. Reproduces the existing dataset byte-for-byte.
  all  : Append the answer-space cue from
         ``scoring.answer_space.TRAINED_TYPE_CUES`` to every record
         whose qa_type has a registered cue. Free-form types
         (trajectory_extrapolation, relative_position) get no cue
         and pass through unchanged.
  mix  : Per-record Bernoulli(p=cue_prob, default 0.5) coin flip
         between bare and cued. The training set ends up with both
         forms in distribution so the LoRA learns to be robust to
         (and benefit from) cue presence at eval time.

The cue, when present, is appended in parentheses BEFORE the answer
marker so the eval-time prompt distribution at
``eval_phase9_heldout_type.py`` (with --answer_cue) lands in-distribution.
The ``target`` field is unchanged -- only the ``question`` and
``prompt`` text are augmented.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------
# Path setup. We import three modules from the snapshot:
#   - phase9_splits.py        -- canonical train/eval splits
#   - qa_generator.py         -- the patched generator with state_schema
#   - clevrer_benchmark.scene_converter -- CLEVRER -> 35-D tensor
# ---------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_GEN_DIR = _SCRIPT_DIR.parent              # data_generation/
_SNAPSHOT_ROOT = _DATA_GEN_DIR.parent           # compsac_2026_code/

# Make ``qa_generator``, ``physics_llm_adapter.phase9_splits``, and
# ``clevrer_benchmark.scene_converter`` importable without polluting
# the package layout.
sys.path.insert(0, str(_SNAPSHOT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from qa_generator import PhysicsQAGenerator, QuestionType  # type: ignore  # noqa: E402
from physics_llm_adapter.phase9_splits import (  # type: ignore  # noqa: E402
    PHASE9_DIVERSE_QA_FILE,
    TRAIN_QUESTION_TYPES,
    TRAIN_SCENE_RANGE,
    describe as describe_splits,
    is_train_scene,
)
from clevrer_benchmark.scene_converter import (  # type: ignore  # noqa: E402
    clevrer_scene_to_state_tensor, load_clevrer_scene,
)
# Phase 10 cue dictionary. Imported lazily inside ``_maybe_inject_cue``
# only when --include_answer_cue is not 'none' so callers running the
# legacy Phase 9 path don't pull in the scoring package.
_TRAINED_TYPE_CUES_CACHE = None


def _trained_type_cues() -> Dict[str, str]:
    """Lazy accessor for the TRAINED_TYPE_CUES dict.

    Imports from ``clevrer_benchmark.scoring.answer_space`` on first
    use to keep the Phase 9 path import-clean. The result is cached
    so repeated lookups are free.
    """
    global _TRAINED_TYPE_CUES_CACHE
    if _TRAINED_TYPE_CUES_CACHE is None:
        from clevrer_benchmark.scoring.answer_space import (  # noqa: E402
            TRAINED_TYPE_CUES,
        )
        _TRAINED_TYPE_CUES_CACHE = dict(TRAINED_TYPE_CUES)
    return _TRAINED_TYPE_CUES_CACHE


def _maybe_inject_cue(
    question: str,
    qa_type: str,
    mode: str,
    cue_prob: float,
    rng: random.Random,
) -> str:
    """Conditionally append an answer-space cue to ``question``.

    mode='none' -> return question verbatim (Phase 9 reproduction path).
    mode='all'  -> always append the cue if registered for this qa_type.
    mode='mix'  -> Bernoulli(cue_prob) coin flip per record.

    Output format is ``"{question} ({cue})"`` (preserving the same
    parenthetical placement that ``scoring.answer_space.build_prompt``
    uses at eval time, so the prompt distributions agree).
    """
    if mode == 'none':
        return question
    cue = _trained_type_cues().get(qa_type)
    if not cue:
        # Free-form qa_type or one without a registered cue: pass through.
        return question
    if mode == 'mix' and rng.random() >= cue_prob:
        return question
    return f'{question} ({cue})'


# ---------------------------------------------------------------------
# Record-emission helpers
# ---------------------------------------------------------------------
def _qa_to_record(
    qa,
    scene_index: int,
    scene_path: Path,
    *,
    cue_mode: str = 'none',
    cue_prob: float = 0.5,
    rng: Optional[random.Random] = None,
) -> Dict:
    """Coerce a ``PhysicsQAGenerator.QAPair`` to the causal_qa_dataset JSON schema.

    ``causal_qa_dataset.json`` records have these fields (verified by
    inspecting the existing 27,413-record dataset shipped with the
    repo): ``qa_type, scene_desc, question, target, prompt,
    scene_index, scene_path``. We mirror them so the
    ``_load_freeform_qa_data`` ingester needs no changes.

    Phase 10 cue injection (Lever 4 Fix 4): when ``cue_mode != 'none'``
    the question (and prompt) are augmented with a parenthetical
    answer-space cue from ``scoring.answer_space.TRAINED_TYPE_CUES``.
    See module docstring for the supported modes.
    """
    q_text = str(qa.question if hasattr(qa, 'question') else qa['question'])
    a_text = str(qa.answer   if hasattr(qa, 'answer')   else qa['answer'])
    qt_obj = qa.question_type if hasattr(qa, 'question_type') else qa['question_type']
    qt_str = qt_obj.value if hasattr(qt_obj, 'value') else str(qt_obj)
    if cue_mode != 'none':
        if rng is None:
            rng = random.Random()
        q_text = _maybe_inject_cue(q_text, qt_str, cue_mode, cue_prob, rng)
    return {
        'qa_type': qt_str,
        'scene_desc': scene_path.stem,            # e.g. 'annotation_10000'
        'question': q_text,
        'target': a_text,
        'prompt': q_text,                         # no Options; free-form
        'scene_index': int(scene_index),
        'scene_path': str(scene_path).replace('\\', '/'),
    }


def _scenes_in_train_range(
    scenes_dir: Path,
    limit: Optional[int] = None,
) -> List[Path]:
    """List CLEVRER scene JSONs whose index lies in the training range.

    The CLEVRER files are named ``annotation_{N}.json`` where ``N`` is
    a 5-digit integer; we filter with ``is_train_scene`` from
    ``phase9_splits`` so this stays in lockstep with the splits config.
    """
    paths: List[Path] = []
    for p in sorted(scenes_dir.glob('annotation_*.json')):
        try:
            scene_num = int(p.stem.split('_')[-1])
        except ValueError:
            continue
        if not is_train_scene(scene_num):
            continue
        paths.append(p)
        if limit is not None and len(paths) >= limit:
            break
    return paths


# ---------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------
def generate(
    scenes_dir: Path,
    output_path: Path,
    records_per_scene: int,
    seed: int,
    scene_limit: Optional[int],
    flush_every: int,
    cue_mode: str = 'none',
    cue_prob: float = 0.5,
) -> None:
    """Walk training scenes, generate diverse QA, write JSON output.

    See module docstring for the ``cue_mode`` / ``cue_prob`` semantics.
    With cue_mode='none' (default) the output is byte-for-byte the
    Phase 9 dataset; cue_mode in {'all','mix'} produces the Phase 10
    augmented variant.
    """
    print(describe_splits())
    if cue_mode != 'none':
        cued_types = sorted(_trained_type_cues().keys())
        print(
            f'[cue] mode={cue_mode}  prob={cue_prob}  '
            f'registered_cues={len(cued_types)}'
        )
        for qt in cued_types:
            print(f'  - {qt}: {_trained_type_cues()[qt]}')
    print()

    # Resolve question types from the strings in phase9_splits to
    # the actual QuestionType enum members.
    train_qts = []
    for s in TRAIN_QUESTION_TYPES:
        try:
            train_qts.append(QuestionType(s))
        except ValueError:
            print(f'[WARN] phase9_splits TRAIN_QUESTION_TYPES contains unknown '
                  f'qa_type {s!r}; skipping')
    print(f'[gen] {len(train_qts)} active training question types')
    for qt in train_qts:
        print(f'  - {qt.value}')
    print()

    # Locate scenes.
    scene_paths = _scenes_in_train_range(scenes_dir, limit=scene_limit)
    print(f'[gen] {len(scene_paths):,} training scenes resolved under '
          f'{scenes_dir}')
    if not scene_paths:
        raise SystemExit(
            f'No training-range CLEVRER scenes found under {scenes_dir}. '
            f'Verify --clevrer_scenes_dir.'
        )
    print()

    # Build the generator -- IMPORTANT: state_schema='clevrer' so mass
    # is read from index 13 and radius from index 14. Without this the
    # mass / radius / energy answers are silently wrong. See the audit
    # script that motivated the schema fix.
    gen = PhysicsQAGenerator(
        question_types=train_qts,
        seed=seed,
        state_schema='clevrer',
    )

    # Reproducible per-scene template choice. We sample
    # ``records_per_scene`` (question_type, template) pairs out of the
    # full templates dict at each scene, with a per-scene-seeded RNG so
    # re-running the script produces an identical dataset.
    rng = random.Random(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    by_qt: Dict[str, int] = {qt.value: 0 for qt in train_qts}
    n_skipped_load = 0
    n_skipped_gen = 0
    t0 = time.time()
    last_flush_n = 0

    for scene_path in tqdm(scene_paths, desc='Phase 9 diverse QA'):
        scene_num = int(scene_path.stem.split('_')[-1])
        try:
            scene = load_clevrer_scene(str(scene_path))
            states_np, masks_np, _ = clevrer_scene_to_state_tensor(scene)
        except Exception as e:
            n_skipped_load += 1
            continue

        # Use the trajectory-mean state for QA generation. Frame 0
        # alone misses objects that enter mid-trajectory; the mask
        # union picks up all objects ever visible. For state values we
        # use frame 0 because that's what most ``_answer_*`` methods
        # reason about (positions and velocities at scene start).
        states_t = torch.from_numpy(states_np[0]).float()  # [N, 35]
        # Union of object visibility across the whole trajectory --
        # avoids dropping objects that exit camera view late.
        mask_union = (masks_np.max(axis=0) > 0.5).astype('float32')
        mask_t = torch.from_numpy(mask_union).float()      # [N]

        # Pick records_per_scene question types with replacement so a
        # 4-object scene can still produce e.g. 10 records spanning
        # multiple types. Mix across types is uniform.
        for _ in range(records_per_scene):
            qt = rng.choice(train_qts)
            try:
                qa = gen.generate_qa_pair(
                    states=states_t,
                    mask=mask_t,
                    question_type=qt,
                )
            except Exception:
                n_skipped_gen += 1
                continue

            rec = _qa_to_record(
                qa, scene_num, scene_path,
                cue_mode=cue_mode, cue_prob=cue_prob, rng=rng,
            )
            records.append(rec)
            by_qt[rec['qa_type']] += 1

        # Periodic atomic save so a Ctrl-C mid-run leaves a usable
        # partial dataset. Cheap (one re-write of the whole file every
        # ``flush_every`` records) but robust.
        if flush_every > 0 and len(records) - last_flush_n >= flush_every:
            _atomic_write(output_path, records)
            last_flush_n = len(records)

    # Final flush.
    _atomic_write(output_path, records)
    dt = time.time() - t0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print(f'Phase 9 diverse QA generation complete in {dt/60:.1f} min')
    print('=' * 70)
    print(f'  scenes processed:        {len(scene_paths):,}')
    print(f'  records emitted:         {len(records):,}')
    print(f'  scenes skipped (load):   {n_skipped_load}')
    print(f'  qa skipped (generate):   {n_skipped_gen}')
    print(f'  output path:             {output_path}')
    print(f'  output size:             {output_path.stat().st_size/1e6:.1f} MB')
    print()
    print('  qa_type distribution:')
    for qt_str, n in sorted(by_qt.items(), key=lambda x: -x[1]):
        pct = 100.0 * n / max(len(records), 1)
        print(f'    {qt_str:<28s} {n:>7,}  ({pct:>5.1f}%)')
    print()


def _atomic_write(path: Path, records: List[Dict]) -> None:
    """Write records to ``path`` atomically (tmp -> rename)."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'Generate the Phase 9 diverse synthetic QA dataset over '
            'CLEVRER training scenes. Output JSON is consumable by '
            'PhysicsReasoningDataset._load_freeform_qa_data without '
            'modification.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--clevrer_scenes_dir', type=Path,
                   default=Path(r'$CLEVRER_DIR/scenes/clevrer_scenes'),
                   help='Directory containing annotation_{NNNNN}.json scene files.')
    p.add_argument('--output', type=Path, default=PHASE9_DIVERSE_QA_FILE,
                   help='Output JSON path.')
    p.add_argument('--records_per_scene', type=int, default=10,
                   help='QA records to emit per scene (uniform random over '
                        'train question types).')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for reproducibility.')
    p.add_argument('--scene_limit', type=int, default=None,
                   help='Cap number of scenes processed (smoke testing).')
    p.add_argument('--flush_every', type=int, default=5000,
                   help='Flush partial output every N records (0 = only at end).')
    p.add_argument(
        '--include_answer_cue',
        choices=['none', 'all', 'mix'],
        default='none',
        help=(
            'Phase 10 (Lever 4 Fix 4): inject answer-space cues into '
            'the question text. none = Phase 9 default (no cues, '
            'output is byte-for-byte the Phase 9 dataset); all = always '
            'cue when a cue is registered for the qa_type; mix = '
            'Bernoulli(--cue_prob) per record. Free-form types without '
            'a registered cue (trajectory_extrapolation, '
            'relative_position) are never cued.'
        ),
    )
    p.add_argument(
        '--cue_prob',
        type=float,
        default=0.5,
        help='Per-record cue-inclusion probability when '
             '--include_answer_cue=mix. Ignored otherwise.',
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    if not args.clevrer_scenes_dir.exists():
        raise SystemExit(
            f'CLEVRER scenes directory not found: {args.clevrer_scenes_dir}'
        )
    generate(
        scenes_dir=args.clevrer_scenes_dir,
        output_path=args.output,
        records_per_scene=args.records_per_scene,
        seed=args.seed,
        scene_limit=args.scene_limit,
        flush_every=args.flush_every,
        cue_mode=args.include_answer_cue,
        cue_prob=args.cue_prob,
    )


if __name__ == '__main__':
    main()
