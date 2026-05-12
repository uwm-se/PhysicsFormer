"""Estimate Phase 3 generalization accuracy on the held-out 10% of CLEVRER val.

The training notebook's 90/10 split is by index order after the H5 is filtered
to causal types (21,378 samples). Since the H5 preserves scene_index order,
the last 10% of the filtered list corresponds to the ~500 scenes with the
HIGHEST scene_indices in the CLEVRER validation set.

This script identifies those scenes by walking the H5 in the SAME order the
dataset used, then filters our Phase 3 details.jsonl to only the held-out
scenes and recomputes accuracy. The resulting numbers are our best estimate of
what the model achieves on scenes it never saw during training.

Caveats: still within-distribution (same simulator, same object types, same
question templates). Not a substitute for a truly out-of-distribution test on
IntPhys/CoPhy/CRAFT/PhysBench.

Usage:
    python compsac_2026_code/clevrer_benchmark/scripts/holdout_generalization_check.py \
        [--h5 PATH] [--details PATH]

The ``--details`` default resolves to ``../results/phase3_GENERATE_singleframe_FULL5000.details.jsonl``
relative to this script, so it works from any CWD inside the snapshot. Pass
``--h5`` (or set the ``CLEVRER_H5`` env var) to point at the CLEVRER training
H5 -- it is too large to ship in the snapshot.
"""

import argparse
import os
import h5py
import json
from collections import defaultdict
from pathlib import Path

# Anchor defaults on this script's location so they work from any CWD.
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DETAILS = _SCRIPT_DIR.parent / 'results' / 'phase3_GENERATE_singleframe_FULL5000.details.jsonl'
DEFAULT_H5 = os.environ.get(
    'CLEVRER_H5',
    os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'),
)

# Match the colab notebook's filter logic
FOCUS_QUESTION_TYPES = {'counterfactual', 'explanatory', 'predictive'}


def identify_heldout_scenes(h5_path):
    """Walk the H5 in dataset order, apply the same filter the notebook used,
    and return the set of scene_indices whose questions sit in the top 10%.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(
            f'CLEVRER training H5 not found at {h5_path}. '
            f'Pass --h5 PATH or set the CLEVRER_H5 env var.'
        )
    with h5py.File(str(h5_path), 'r') as hf:
        n = hf['questions'].shape[0]
        print(f'H5 total samples: {n:,}')
        qtypes = hf['question_types'][:]
        metadata = hf['metadata'][:]

    # Apply same filter as the notebook (FOCUS_QUESTION_TYPES)
    filtered_idx = []
    for i, qt in enumerate(qtypes):
        s = qt.decode('utf-8') if isinstance(qt, bytes) else str(qt)
        if s.lower() in FOCUS_QUESTION_TYPES:
            filtered_idx.append(i)
    print(f'After filter to {FOCUS_QUESTION_TYPES}: {len(filtered_idx):,} samples')

    train_size = int(0.9 * len(filtered_idx))
    test_idx = filtered_idx[train_size:]
    print(f'90/10 split: train={train_size:,}, heldout_test={len(test_idx):,}')

    # Extract scene_indices for heldout test samples from metadata
    heldout_scenes = set()
    heldout_scene_q_count = defaultdict(int)
    for i in test_idx:
        meta_str = metadata[i]
        if isinstance(meta_str, bytes):
            meta_str = meta_str.decode('utf-8')
        try:
            m = json.loads(meta_str)
            s = m.get('scene_index')
            if s is not None:
                heldout_scenes.add(s)
                heldout_scene_q_count[s] += 1
        except Exception:
            continue
    print(f'Heldout test samples span {len(heldout_scenes):,} unique scene_indices')
    print(f'  min scene_index: {min(heldout_scenes)}')
    print(f'  max scene_index: {max(heldout_scenes)}')
    return heldout_scenes


def accuracy_on_heldout(details_path, heldout_scenes):
    """Recompute accuracy restricted to heldout scenes only."""
    by_type = defaultdict(lambda: {'correct': 0, 'total': 0, 'correct_valid': 0, 'total_valid': 0})
    with open(details_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            scene_id = r.get('scene_id', '')
            # scene_id is like "annotation_14500"
            scene_num = None
            for tok in scene_id.replace('annotation_', '').split('_'):
                try:
                    scene_num = int(tok)
                    break
                except ValueError:
                    continue
            if scene_num is None or scene_num not in heldout_scenes:
                continue
            qt = r.get('clevrer_type', '?')
            correct = bool(r.get('correct'))
            has_correct_choice = bool(r.get('correct_choices'))
            by_type['ALL']['total'] += 1
            by_type[qt]['total'] += 1
            if correct:
                by_type['ALL']['correct'] += 1
                by_type[qt]['correct'] += 1
            if has_correct_choice:
                by_type['ALL']['total_valid'] += 1
                by_type[qt]['total_valid'] += 1
                if correct:
                    by_type['ALL']['correct_valid'] += 1
                    by_type[qt]['correct_valid'] += 1

    print(f'\n=== Phase 3 accuracy on heldout-only subset of CLEVRER val ===')
    print(f'{"type":<20} {"inclusive":>24} {"valid-only":>24}')
    for qt in ['ALL', 'explanatory', 'predictive', 'counterfactual']:
        d = by_type.get(qt)
        if d is None or d['total'] == 0:
            continue
        inc_acc = 100 * d['correct'] / d['total']
        v_acc = 100 * d['correct_valid'] / max(d['total_valid'], 1)
        print(f'{qt:<20} {inc_acc:>7.1f}% ({d["correct"]}/{d["total"]}) '
              f'{v_acc:>7.1f}% ({d["correct_valid"]}/{d["total_valid"]})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', type=Path, default=Path(DEFAULT_H5),
                        help=f'CLEVRER training H5 path (default: {DEFAULT_H5}; '
                             f'also reads CLEVRER_H5 env var).')
    parser.add_argument('--details', type=Path, default=DEFAULT_DETAILS,
                        help=f'Phase-3 details.jsonl path '
                             f'(default: {DEFAULT_DETAILS}).')
    args = parser.parse_args()
    if not args.details.exists():
        raise FileNotFoundError(
            f'Phase-3 details file not found: {args.details}. '
            f'Generate it via run_adapter_evaluation.py or pass --details PATH.'
        )
    heldout_scenes = identify_heldout_scenes(args.h5)
    accuracy_on_heldout(args.details, heldout_scenes)


if __name__ == '__main__':
    main()
