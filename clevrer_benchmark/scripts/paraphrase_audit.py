"""NLI-based paraphrase audit for Phase 3 wrong predictions.

For every prediction the strict scorer flagged as wrong, runs a Natural Language
Inference (NLI) model in both directions (P -> C and C -> P) to test whether the
predicted text is a *paraphrase* of one of the correct CLEVRER choices that the
substring-based ``evaluate_answer()`` simply missed because of surface form.

The flip rule is intentionally conservative -- it requires bidirectional
entailment above a threshold AND that no wrong choice scores at least as high:

    flip if:  max_correct_score >= threshold
              AND  max_correct_score > max_wrong_score + margin

Default model: ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` -- a strong
small NLI checkpoint that runs comfortably on a single GPU and downloads once
(~180 MB). Override via ``--model``.

Default input: ``phase3_BASELINE_SHUFFLE_FULL5000.details.jsonl`` -- the file
behind the paper's primary macros (69.2 / 79.4 / 63.4 / 63.6). Override via
``--details``.

Usage::

    # paper-grade run on heldout valid-only pool:
    python compsac_2026_code/clevrer_benchmark/scripts/paraphrase_audit.py \\
        --heldout --valid_only

    # quick smoke test (skips the 99% verbatim-wrong picks, audits free-form only):
    python compsac_2026_code/clevrer_benchmark/scripts/paraphrase_audit.py \\
        --skip_verbatim_wrong --max 200

The script never overwrites the source ``.details.jsonl``; it only reports
candidate flips and an adjusted-accuracy ceiling. Pass ``--out PATH`` to
also dump every flipped record (with NLI scores) as a sidecar JSONL for
manual review.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the held-out scene loader from compute_paper_stats.py rather than
# duplicating the H5-walking logic. The two scripts live in the same dir so a
# sys.path insert is sufficient. The benchmark-root insert lets us import the
# shared ``scoring`` package without packaging gymnastics.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_BENCH_DIR))
from compute_paper_stats import load_heldout_scenes  # noqa: E402

# Scoring helpers were extracted into ``clevrer_benchmark/scoring/`` so the
# audit pipeline, free-form transfer tests, paraphrase MCQ test, and the
# Phase 9 lenient rescore all share one definition of ``substring_correct``
# / ``bucket`` / NLI paraphrase logic. The names below are the original
# underscore-prefixed helpers; they remain as thin aliases in this module so
# the rest of the file is unchanged.
from scoring.text_match import _norm, _bucket  # noqa: E402
from scoring.nli_paraphrase import (  # noqa: E402
    _build_pairs,
    _run_nli,
    _evaluate_flips,
)

# Snapshot-portable defaults (anchor on __file__ so any CWD works).
DEFAULT_DETAILS = (_SCRIPT_DIR.parent / 'results' /
                   'phase3_BASELINE_SHUFFLE_FULL5000.details.jsonl')
DEFAULT_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
DEFAULT_H5 = os.environ.get(
    'CLEVRER_H5',
    os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'),
)


def _load_records(details_path: Path,
                  valid_only: bool,
                  heldout_scenes: Optional[set]) -> List[Dict]:
    """Stream ``details.jsonl`` and keep only the records we need to audit."""
    out = []
    with open(details_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('correct'):
                continue
            if valid_only and not (r.get('correct_choices') or []):
                continue
            if heldout_scenes is not None:
                sid = r.get('scene_id', '')
                try:
                    num = int(sid.replace('annotation_', '').split('_')[-1])
                except ValueError:
                    continue
                if num not in heldout_scenes:
                    continue
            out.append(r)
    return out


def _summarise_buckets(records: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        b = r.get('_audit_bucket', 'unbucketed')
        out[b] = out.get(b, 0) + 1
    return out


def _print_flip_samples(flips: List[Dict], limit: int) -> None:
    if not flips:
        print('  (none)')
        return
    for r in flips[:limit]:
        pred = _norm(r.get('predicted', ''))
        correct = [_norm(c['choice']) for c in r['choices'] if c['answer'] == 'correct']
        wrong = [_norm(c['choice']) for c in r['choices'] if c['answer'] == 'wrong']
        print(f"  qid:    {r.get('question_id', '?')}  qtype: {r.get('clevrer_type', '?')}")
        print(f"  pred:   {pred!r}")
        print(f"  corr:   {correct}")
        print(f"  wrong:  {wrong[:3]}")
        print(f"  scores: max_correct={r['_audit_max_correct_paraphrase']:.3f}"
              f"   max_wrong={r['_audit_max_wrong_paraphrase']:.3f}"
              f"   bucket={r['_audit_bucket']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='NLI paraphrase audit of Phase 3 wrong predictions.')
    parser.add_argument('--details', type=Path, default=DEFAULT_DETAILS,
                        help=f'Phase 3 details.jsonl to audit '
                             f'(default: {DEFAULT_DETAILS.name}).')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'HuggingFace NLI model id (default: {DEFAULT_MODEL}).')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Min bidirectional entailment probability to count '
                             'as a paraphrase (default: 0.7).')
    parser.add_argument('--margin', type=float, default=0.05,
                        help='Required gap of max-correct-score over max-wrong-score '
                             'before flipping (default: 0.05).')
    parser.add_argument('--heldout', action='store_true',
                        help='Restrict audit to the 501 heldout scenes (paper primary pool).')
    parser.add_argument('--valid_only', action='store_true',
                        help='Drop questions where every choice is labeled wrong (matches '
                             'compute_paper_stats.py --valid_only).')
    parser.add_argument('--skip_verbatim_wrong', action='store_true',
                        help='Skip predictions that exact-match a labeled wrong choice. '
                             'These are confirmed reasoning errors -- skipping them is '
                             '~99%% faster and rarely changes the count of flips.')
    parser.add_argument('--max', type=int, default=None, dest='max_records',
                        help='Limit to the first N wrong records (smoke testing).')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--h5', type=Path, default=Path(DEFAULT_H5),
                        help=f'CLEVRER H5 path; only used with --heldout. '
                             f'Default: {DEFAULT_H5} (also reads CLEVRER_H5 env var).')
    parser.add_argument('--out', type=Path, default=None,
                        help='Optional sidecar JSONL to receive every flipped record + scores.')
    parser.add_argument('--show_samples', type=int, default=10,
                        help='Number of flipped samples to print to stdout (default: 10).')
    args = parser.parse_args()

    if not args.details.exists():
        raise FileNotFoundError(
            f'Details file not found: {args.details}. '
            f'Generate it via run_adapter_evaluation.py --save_details.'
        )

    # Pick device.
    if args.device == 'auto':
        try:
            import torch
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            args.device = 'cpu'

    heldout = load_heldout_scenes(args.h5) if args.heldout else None

    print(f'[load] {args.details}')
    records = _load_records(args.details, valid_only=args.valid_only,
                            heldout_scenes=heldout)
    n_total_wrong = len(records)
    if args.max_records is not None:
        records = records[:args.max_records]

    pairs, meta = _build_pairs(records, skip_verbatim_wrong=args.skip_verbatim_wrong)
    buckets = _summarise_buckets(records)

    print()
    print(f'=== Wrong-prediction audit (NLI paraphrase) ===')
    print(f'  details:        {args.details.name}')
    print(f'  filters:        '
          f'heldout={"on" if args.heldout else "off"}, '
          f'valid_only={"on" if args.valid_only else "off"}, '
          f'skip_verbatim_wrong={"on" if args.skip_verbatim_wrong else "off"}')
    print(f'  model:          {args.model}')
    print(f'  device:         {args.device}')
    print(f'  threshold:      paraphrase prob >= {args.threshold:.2f} '
          f'(margin >= {args.margin:.2f} over best wrong)')
    print()
    print(f'  total wrong (after filters):   {n_total_wrong:,}')
    if args.max_records is not None:
        print(f'  audited:                       {len(records):,} (capped via --max)')
    print(f'  bucket breakdown:')
    for b, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / max(len(records), 1)
        print(f'      {n:>6,}  {b:<32}  ({pct:5.1f}%)')
    n_to_score = sum(1 for r in records if not r.get('_audit_skipped'))
    n_pairs = len(pairs)
    print(f'  records sent to NLI:           {n_to_score:,}')
    print(f'  premise-hypothesis pairs:      {n_pairs:,} (bidirectional)')
    print()

    if not pairs:
        print('[done] Nothing to score (all wrong predictions filtered out).')
        return

    scores = _run_nli(pairs, model_id=args.model, device=args.device,
                      batch_size=args.batch_size)

    # The shared ``evaluate_flips`` dropped the unused ``pairs`` argument that
    # the original signature accepted -- ``meta`` + ``scores`` carry every bit
    # of information the flip rule needs. We pass through the same threshold
    # / margin here so the audit numbers stay byte-identical to pre-refactor.
    n_flips, flips = _evaluate_flips(records, meta, scores,
                                     threshold=args.threshold, margin=args.margin)

    print()
    print(f'=== Results ===')
    print(f'  semantic flips:                {n_flips:,} of {n_to_score:,} audited '
          f'({100.0 * n_flips / max(n_to_score, 1):.2f}%)')

    # Recompute the headline accuracy assuming every flip turns into a correct.
    # For interpretability, also report the strict-match correct count from
    # the same filter set so the reader can tell what changed.
    n_correct_strict = 0
    n_total = 0
    with open(args.details, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if args.valid_only and not (r.get('correct_choices') or []):
                continue
            if heldout is not None:
                sid = r.get('scene_id', '')
                try:
                    num = int(sid.replace('annotation_', '').split('_')[-1])
                except ValueError:
                    continue
                if num not in heldout:
                    continue
            n_total += 1
            if r.get('correct'):
                n_correct_strict += 1

    if n_total > 0:
        strict_acc = 100.0 * n_correct_strict / n_total
        adj_acc = 100.0 * (n_correct_strict + n_flips) / n_total
        print(f'  strict-match accuracy:         {strict_acc:.2f}%  '
              f'({n_correct_strict:,}/{n_total:,})')
        print(f'  +flips ceiling accuracy:       {adj_acc:.2f}%  '
              f'(+{n_flips:,} -> {n_correct_strict + n_flips:,}/{n_total:,})')
        print(f'  delta from semantic match:     +{adj_acc - strict_acc:.2f} pp')

    print()
    print(f'=== Top {min(args.show_samples, len(flips))} flip candidates ===')
    _print_flip_samples(flips, args.show_samples)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for r in flips:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + '\n')
        print(f'  flipped records written to: {args.out}')


if __name__ == '__main__':
    main()
