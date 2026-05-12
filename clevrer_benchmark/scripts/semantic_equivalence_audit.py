"""Scene-aware referent-equivalence audit of Phase 3 wrong predictions.

Question this audit answers
---------------------------
The substring scorer in ``run_adapter_evaluation.evaluate_answer()`` rejects
predictions like ``"the blue object's colliding with the metal cylinder"``
when the labeled correct choice is ``"the blue cylinder's colliding with the
metal cylinder"`` -- but in scenes where there is exactly one blue object and
that object IS the blue cylinder, the prediction unambiguously identifies the
same entity. CLEVRER also offers ``"the blue object"`` as a wrong choice in
some questions, which is exactly when the substring rule rejects this
prediction (because it matches a wrong choice).

This script implements a scene-aware referent-equivalence rule:

    Reclaim a wrong record iff, after loading the actual CLEVRER scene
    object inventory, the prediction's parsed event template and entity
    descriptors uniquely identify the same physical entities as at least
    one labeled correct choice -- and do NOT identify the same entities
    as any labeled wrong choice.

The reclaim is intentionally conservative: a prediction that maps to two or
more scene objects (ambiguous referent), or that maps to a different event
kind (collision vs presence vs entrance/exit), is left as a wrong.

The rule is **a robustness check, not a primary score**. The strict
substring number remains the apples-to-apples comparison with the LLM
baselines (which all run under the same strict scorer). The reclaim ceiling
shows the strict number isn't a substring-matching artifact.

Output format
-------------
A summary table with three numbers side-by-side:

  strict substring acc           : the paper's primary 69.2 / 79.6
  NLI bidirectional ceiling      : already +0.05 pp from paraphrase_audit.py
  scene-aware referent ceiling   : new -- this script

Plus per-question records of what got reclaimed for review.

Usage
-----
::

    python compsac_2026_code/clevrer_benchmark/scripts/semantic_equivalence_audit.py \\
        [--heldout] [--clevrer_dir $CLEVRER_DIR]

Snapshot-portable: paths anchor on ``__file__``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# Snapshot-portable defaults.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _SCRIPT_DIR.parent
_RESULTS_DIR = _BENCH_DIR / 'results'
DEFAULT_DETAILS = _RESULTS_DIR / 'phase3_BASELINE_SHUFFLE_FULL5000.details.jsonl'
DEFAULT_OUTPUT = _RESULTS_DIR / 'semantic_equivalence_audit.json'
DEFAULT_H5 = os.environ.get(
    'CLEVRER_H5',
    os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'),
)
DEFAULT_CLEVRER = Path(os.environ.get('CLEVRER_DIR', 'clevrer'))

sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_BENCH_DIR))

# Event-template parsing, descriptor parsing, scene-object referent
# resolution and the scene-loading cache live in the shared scoring
# package now -- they are reused by paraphrased-MCQ tests, free-form
# transfer tests, and the Phase 9 lenient rescore. Behaviour is byte
# identical to the pre-refactor inline definitions; the names below
# preserve the original public API so callers are unaffected.
from scoring.referent_equiv import (  # noqa: E402
    COLORS,
    MATERIALS,
    SHAPES,
    GENERIC_SHAPE,
    EVENT_PATTERNS,
    parse_event,
    parse_descriptor,
    matches_scene_object,
    referent_set,
    is_semantic_match,
    load_scene_objects,
    scene_id_to_num as _scene_id_to_num,
    scene_object_inventory as _scene_object_inventory,
)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Scene-aware referent-equivalence audit of Phase 3 wrongs.')
    parser.add_argument('--details', type=Path, default=DEFAULT_DETAILS,
                        help=f'Phase 3 details.jsonl (default: {DEFAULT_DETAILS.name}).')
    parser.add_argument('--heldout', action='store_true', default=True,
                        help='Restrict to held-out 501 scenes (default on).')
    parser.add_argument('--no_heldout', action='store_false', dest='heldout')
    parser.add_argument('--valid_only', action='store_true', default=True,
                        help='Drop zero-correct MCQ trap items (default on).')
    parser.add_argument('--no_valid_only', action='store_false', dest='valid_only')
    parser.add_argument('--clevrer_dir', type=Path, default=DEFAULT_CLEVRER)
    parser.add_argument('--h5', type=Path, default=Path(DEFAULT_H5))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--show_samples', type=int, default=15)
    args = parser.parse_args()

    if not args.details.exists():
        raise FileNotFoundError(f'Details file not found: {args.details}')

    heldout: Optional[Set[int]] = None
    if args.heldout:
        from compute_paper_stats import load_heldout_scenes  # type: ignore
        heldout = load_heldout_scenes(args.h5)
        print(f'[heldout] {len(heldout)} scenes loaded')

    # Pass 1: load and bucket records.
    print(f'[load] {args.details}')
    raw_records: List[Dict] = []
    with open(args.details, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if args.valid_only and not (r.get('correct_choices') or []):
                continue
            if heldout is not None:
                num = _scene_id_to_num(r.get('scene_id', ''))
                if num is None or num not in heldout:
                    continue
            raw_records.append(r)
    n_total = len(raw_records)
    n_strict_correct = sum(1 for r in raw_records if r.get('correct'))
    wrongs = [r for r in raw_records if not r.get('correct')]

    print(f'[pool] total={n_total:,}  strict_correct={n_strict_correct:,}  '
          f'strict_wrong={len(wrongs):,}')

    # Pass 2: for each wrong, attempt referent-equivalence reclaim.
    n_no_scene = 0
    n_no_pred_event = 0
    n_no_choice_event = 0
    n_reclaimed = 0
    n_blocked_by_wrong = 0  # would-have-reclaimed but matched a labeled wrong too
    reclaim_records: List[Dict] = []
    blocked_records: List[Dict] = []
    breakdown_by_qtype: Counter = Counter()

    for r in wrongs:
        scene_num = _scene_id_to_num(r.get('scene_id', ''))
        if scene_num is None:
            continue
        inventory = load_scene_objects(args.clevrer_dir, scene_num)
        if inventory is None:
            n_no_scene += 1
            continue

        pred = (r.get('predicted') or '').strip().lower()
        if parse_event(pred) is None:
            n_no_pred_event += 1
            continue

        correct_texts = [c['choice'] for c in r['choices'] if c.get('answer') == 'correct']
        wrong_texts = [c['choice'] for c in r['choices'] if c.get('answer') == 'wrong']

        matched_correct = [c for c in correct_texts
                           if is_semantic_match(pred, c, inventory)]
        matched_wrong = [w for w in wrong_texts
                         if is_semantic_match(pred, w, inventory)]

        if not matched_correct and not matched_wrong:
            # No semantic match at all -- this is a genuine wrong.
            continue
        if matched_correct and not matched_wrong:
            n_reclaimed += 1
            breakdown_by_qtype[r.get('clevrer_type', 'unknown')] += 1
            if len(reclaim_records) < args.show_samples:
                reclaim_records.append({
                    'scene_id': r.get('scene_id'),
                    'qtype': r.get('clevrer_type'),
                    'question': r.get('question_text'),
                    'predicted': pred,
                    'matched_correct': matched_correct,
                    'matched_wrong': matched_wrong,
                    'scene_object_count': len(inventory),
                })
            continue
        if matched_correct and matched_wrong:
            # Predicted descriptor referent-matches BOTH a correct and a wrong
            # choice -- ambiguous in this scene, so do not reclaim.
            n_blocked_by_wrong += 1
            if len(blocked_records) < args.show_samples:
                blocked_records.append({
                    'scene_id': r.get('scene_id'),
                    'qtype': r.get('clevrer_type'),
                    'question': r.get('question_text'),
                    'predicted': pred,
                    'matched_correct': matched_correct,
                    'matched_wrong': matched_wrong,
                    'scene_object_count': len(inventory),
                })

    n_lenient_correct = n_strict_correct + n_reclaimed
    print()
    print('=' * 72)
    print('Scene-aware referent-equivalence audit -- summary')
    print('=' * 72)
    print(f'  pool                                    n = {n_total:,}')
    print(f'  strict-substring correct                {n_strict_correct:,}'
          f'  ({100.0*n_strict_correct/max(n_total,1):.2f}%)')
    print(f'  + reclaimed under referent-equivalence  +{n_reclaimed:,}'
          f'  (+{100.0*n_reclaimed/max(n_total,1):.2f} pp)')
    print(f'  --> referent-equivalence ceiling        {n_lenient_correct:,}'
          f'  ({100.0*n_lenient_correct/max(n_total,1):.2f}%)')
    print()
    print('Diagnostic counts:')
    print(f'  scene file missing:                     {n_no_scene:,}')
    print(f'  pred didn\'t match any event template:   {n_no_pred_event:,}')
    print(f'  blocked (matched correct AND wrong):    {n_blocked_by_wrong:,}')
    print()
    print('Reclaim breakdown by question type:')
    for qt in sorted(breakdown_by_qtype):
        print(f'  {qt:<18} {breakdown_by_qtype[qt]:,}')
    print()

    if reclaim_records:
        print('=' * 72)
        print('Sample reclaimed records (pred semantically equals correct)')
        print('=' * 72)
        for s in reclaim_records:
            print(f'\n  [{s["qtype"]}] scene={s["scene_id"]} (n_objs={s["scene_object_count"]})')
            print(f'    Q:               {str(s["question"])[:90]}')
            print(f'    predicted:       {s["predicted"]!r}')
            print(f'    matched correct: {s["matched_correct"]}')
    if blocked_records:
        print()
        print('=' * 72)
        print('Sample BLOCKED records (matched both correct AND wrong -- ambiguous)')
        print('=' * 72)
        for s in blocked_records:
            print(f'\n  [{s["qtype"]}] scene={s["scene_id"]} (n_objs={s["scene_object_count"]})')
            print(f'    Q:               {str(s["question"])[:90]}')
            print(f'    predicted:       {s["predicted"]!r}')
            print(f'    matched correct: {s["matched_correct"]}')
            print(f'    matched wrong:   {s["matched_wrong"]}')

    summary = {
        'config': {
            'details': str(args.details),
            'heldout': args.heldout,
            'valid_only': args.valid_only,
            'clevrer_dir': str(args.clevrer_dir),
        },
        'pool': {
            'n_total': n_total,
            'strict_correct': n_strict_correct,
            'strict_wrong': len(wrongs),
            'strict_acc_pct': round(100.0 * n_strict_correct / max(n_total, 1), 2),
        },
        'reclaim': {
            'n_reclaimed': n_reclaimed,
            'n_blocked_by_ambiguity': n_blocked_by_wrong,
            'n_no_scene_file': n_no_scene,
            'n_no_pred_event_template': n_no_pred_event,
            'lenient_correct': n_lenient_correct,
            'lenient_acc_pct': round(100.0 * n_lenient_correct / max(n_total, 1), 2),
            'delta_pp': round(100.0 * n_reclaimed / max(n_total, 1), 2),
            'breakdown_by_qtype': dict(breakdown_by_qtype),
        },
        'reclaim_sample': reclaim_records,
        'blocked_sample': blocked_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n  full summary written to: {args.out}')


if __name__ == '__main__':
    main()
