"""Dump per-type LLM baseline accuracies + Wilson CIs from FULL files.

Diagnostic for switching the article's primary numbers from 1K-pool to
held-out: which LLM baseline claims still hold once Ours moves to
held-out values?
"""
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / 'results'


def wilson(c, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = c / n
    z = 1.96
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, cen - half), min(1.0, cen + half)


def fmt(c, n):
    p, lo, hi = wilson(c, n)
    return f'{p * 100:5.1f}% [{lo * 100:5.1f},{hi * 100:5.1f}] (n={n})'


def main():
    print(f'{"file":<48} {"overall":<28} {"explan":<28} {"pred":<28} {"counter":<28}')
    print('-' * 168)
    for f in sorted(RESULTS.glob('*_with_scene*FULL.json')):
        if 'partial' in f.name or 'wrong' in f.name:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        n = d.get('total', 0)
        c = d.get('correct', 0)
        overall = fmt(c, n)
        parts = []
        for qt in ('explanatory', 'predictive', 'counterfactual'):
            b = d.get('by_type', {}).get(qt, {})
            c2 = b.get('correct', 0)
            n2 = b.get('total', 0)
            parts.append(fmt(c2, n2))
        print(f'{f.name:<48} {overall:<28} {parts[0]:<28} {parts[1]:<28} {parts[2]:<28}')


if __name__ == '__main__':
    main()
