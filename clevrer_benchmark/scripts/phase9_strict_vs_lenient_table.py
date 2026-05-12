"""Print a single comparison table from a phase9_lenient_rescore.json.

Run after ``eval_phase9_lenient_rescore.py``; prints a paper-ready
strict-vs-lenient comparison for the Phase 9 evaluation suite.

Pure formatter -- no scoring logic, no I/O beyond reading the saved
summary JSON. Safe to run standalone or import as a function.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = _BENCH_DIR / 'results' / 'phase9_lenient_rescore.json'


def render(d: dict) -> None:
    print()
    print('=' * 78)
    print('Phase 9: STRICT vs LENIENT (final aggregation)')
    print('=' * 78)

    # 1. Categorical held-out types
    ht = d.get('heldout_type')
    if ht:
        print()
        print('1. Held-out question TYPES (categorical, n=200)')
        hdr = f'  {"qa_type":<26} {"strict":>9} {"synonym":>9} {"ord_tol1":>9} {"delta_max_pp":>14}'
        print(hdr)
        for qt, b in sorted(ht['per_qa_type'].items()):
            s = b['strict']['acc_pct']
            sy = b['synonym']['acc_pct']
            ot = b['ordinal_tol1']['acc_pct']
            print(f'  {qt:<26} {s:>8.2f}% {sy:>8.2f}% {ot:>8.2f}% {(ot - s):>13.2f}')
        oa = ht['overall']
        s = oa['strict']['acc_pct']
        sy = oa['synonym']['acc_pct']
        ot = oa['ordinal_tol1']['acc_pct']
        print(f'  {"OVERALL":<26} {s:>8.2f}% {sy:>8.2f}% {ot:>8.2f}% {(ot - s):>13.2f}')

    # 2. CLEVRER MCQ-style evals strict vs lenient
    print()
    print('2. CLEVRER MCQ-style evals')
    cols = ['n', 'strict', '+ref', 'NLI@.40', 'NLI@.50', 'NLI@.60', 'NLI@.70', 'NLI@.80']
    hdr = '  ' + f'{"eval":<22}' + ' '.join(f'{c:>9}' for c in cols)
    print(hdr)
    for kind in ('paraphrase', 'prefix_ablation', 'free_form_transfer'):
        s = d.get(kind)
        if s is None:
            continue
        n = s['n_flattened_records']
        strict = s['strict_acc_pct']
        ref = s['referent_equivalence']['lenient_acc_pct']
        nli = s['nli_paraphrase']['thresholds']
        cells = [f'{nli[t]["lenient_acc_pct"]:>8.2f}%' for t in ('0.40', '0.50', '0.60', '0.70', '0.80')]
        print(f'  {kind:<22} {n:>9} {strict:>8.2f}% {ref:>8.2f}% ' + ' '.join(cells))

    # 3. Per-condition strict accuracy
    print()
    print('3. Per-condition strict accuracy')
    for kind in ('paraphrase', 'prefix_ablation', 'free_form_transfer'):
        s = d.get(kind)
        if s is None:
            continue
        pc = s.get('per_condition_strict', {})
        if not pc:
            continue
        print(f'  {kind}:')
        for k, v in sorted(pc.items()):
            print(f'    {k:<28} n={v["n"]:>4}  strict={v["strict_acc_pct"]:>6.2f}%')

    # 4. NLI flips per-condition at thr=0.50
    print()
    print('4. NLI flips per-condition (thr=0.50)')
    for kind in ('paraphrase', 'prefix_ablation', 'free_form_transfer'):
        s = d.get(kind)
        if s is None:
            continue
        pcf = s.get('nli_paraphrase', {}).get('per_condition_flips', {})
        if not pcf:
            continue
        print(f'  {kind}:')
        for k, thr_map in sorted(pcf.items()):
            print(f'    {k:<28} flips_at_thr0.50={thr_map.get("0.50", 0)}')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    args = p.parse_args()
    with open(args.input, 'r', encoding='utf-8') as f:
        d = json.load(f)
    render(d)


if __name__ == '__main__':
    main()
