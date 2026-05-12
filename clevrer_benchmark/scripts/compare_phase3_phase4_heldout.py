"""Apples-to-apples Phase 3 vs Phase 4 held-out MCQ comparison.

Reuses ``compute_paper_stats.load_result`` so Phase 3's FULL5000 result is
filtered to held-out + valid-only on the fly (yielding the paper's 69.2%
primary). Phase 4 was already evaluated on the held-out subset directly, so
we just apply ``valid_only`` to its details file (also filtering 7%
malformed) for a clean comparison.

Reports accuracy, Wilson 95% CI, and Fisher's exact p-value per category.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from compute_paper_stats import (  # noqa: E402
    load_result, load_heldout_scenes, wilson_ci, cohen_h,
    fisher_pvalue, format_pvalue,
)

DEFAULT_H5 = Path(os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'))


def _row(label, d):
    if d is None or d.get('total', 0) == 0:
        return f'  {label:<28} ---'
    p, lo, hi = wilson_ci(d['correct'], d['total'])
    return (f'  {label:<28} {p*100:>5.1f}% '
            f'[{lo*100:>4.1f}, {hi*100:>4.1f}]  '
            f'({d["correct"]}/{d["total"]})')


def main() -> None:
    results_dir = _HERE.parent / 'results'
    p3_path = results_dir / 'phase3_BASELINE_SHUFFLE_FULL5000.json'
    p4_path = results_dir / 'phase4_heldout_BASELINE_SHUFFLE.json'
    if not p3_path.exists():
        raise SystemExit(f'Phase 3 reference missing: {p3_path}')
    if not p4_path.exists():
        raise SystemExit(f'Phase 4 result missing: {p4_path}')

    print('=' * 78)
    print('  Phase 3 vs Phase 4 held-out MCQ comparison (paper primary protocol)')
    print('  filter: --shuffle_choices, --filter_malformed, heldout-only, valid_only')
    print('=' * 78)

    heldout = load_heldout_scenes(DEFAULT_H5)
    print(f'  heldout pool: {len(heldout)} scenes (10% holdout from CLEVRER training H5)')
    print()

    # Phase 3: full 5000 -> filter to heldout + valid_only via the details sidecar.
    p3 = load_result(p3_path, valid_only=True, heldout_scenes=heldout)
    # Phase 4: already heldout-only at eval time, just apply valid_only post-hoc.
    p4 = load_result(p4_path, valid_only=True, heldout_scenes=None)

    cats = ['overall', 'explanatory', 'predictive', 'counterfactual']
    labels = {'overall': 'Overall', 'explanatory': 'Explanatory',
              'predictive': 'Predictive', 'counterfactual': 'Counterfactual'}

    print('  Phase 3 (paper primary):')
    for c in cats:
        print(_row(labels[c], p3.get(c)))
    print()
    print('  Phase 4 (mixed-format):')
    for c in cats:
        print(_row(labels[c], p4.get(c)))
    print()

    print('=' * 78)
    print('  Deltas (Phase 4 - Phase 3): pp, Cohen\'s h, Fisher\'s exact p')
    print('=' * 78)
    for c in cats:
        b = p3.get(c); a = p4.get(c)
        if not b or not a or b['total'] == 0 or a['total'] == 0:
            print(f'  {labels[c]:<20} ---')
            continue
        pb, pa = b['correct'] / b['total'], a['correct'] / a['total']
        delta_pp = (pa - pb) * 100
        h = cohen_h(pa, pb)
        p = fisher_pvalue(a['correct'], a['total'], b['correct'], b['total'])
        print(f'  {labels[c]:<20} {delta_pp:>+6.2f} pp   '
              f'h={h:>+5.2f}   p={format_pvalue(p):<10}   '
              f'(P3 n={b["total"]}, P4 n={a["total"]})')

    print()
    print('=' * 78)
    print('  Tier 1a Phase A gate (regression sentinel)')
    print('=' * 78)
    if p3 and p4 and p3['overall']['total'] > 0 and p4['overall']['total'] > 0:
        p3_acc = p3['overall']['correct'] / p3['overall']['total']
        p4_acc = p4['overall']['correct'] / p4['overall']['total']
        gate_lo, gate_hi = p3_acc - 0.02, p3_acc + 0.02
        print(f'  Phase 3 overall:  {p3_acc*100:.2f}%')
        print(f'  Phase 4 overall:  {p4_acc*100:.2f}%')
        print(f'  Gate window:      [{gate_lo*100:.2f}%, {gate_hi*100:.2f}%]')
        print(f'  Delta:            {(p4_acc - p3_acc)*100:+.2f} pp')
        if p4_acc < gate_lo:
            print(f'  Verdict: [FAIL] Phase 4 regressed below the lower gate.')
        elif p4_acc > gate_hi:
            print(f'  Verdict: [PASS+] Phase 4 ABOVE the upper gate -- improved beyond noise floor.')
            print(f'           (gate is "must not regress"; improvement passes the regression test)')
        else:
            print(f'  Verdict: [PASS] Phase 4 within \u00b12pp of Phase 3.')


if __name__ == '__main__':
    main()
