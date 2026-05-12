"""Count held-out questions per type and report Wilson CIs.

Diagnostic: how many predictive questions are in the held-out 10% subset?
Used to decide whether the held-out predictive sample is large enough to
support tight CIs in the article's predictive headline comparison.
"""
import os
import json
import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from holdout_generalization_check import identify_heldout_scenes  # noqa: E402

H5 = os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5')
DETAILS = _SCRIPT_DIR.parent / 'results' / 'phase3_BASELINE_SHUFFLE_FULL5000.details.jsonl'


def wilson(c, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = c / n
    z = 1.96
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, cen - half), min(1.0, cen + half)


def main():
    held = identify_heldout_scenes(H5)
    counts = {qt: {'all': 0, 'valid': 0, 'correct': 0, 'correct_valid': 0}
              for qt in ('explanatory', 'predictive', 'counterfactual')}
    with open(DETAILS, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            sid = r.get('scene_id', '')
            try:
                num = int(sid.replace('annotation_', '').split('_')[-1])
            except ValueError:
                continue
            if num not in held:
                continue
            qt = r.get('clevrer_type', '')
            if qt not in counts:
                continue
            is_valid = bool(r.get('correct_choices'))
            is_corr = bool(r.get('correct'))
            counts[qt]['all'] += 1
            counts[qt]['correct'] += int(is_corr)
            if is_valid:
                counts[qt]['valid'] += 1
                counts[qt]['correct_valid'] += int(is_corr)

    print()
    print(f'{"Type":<16} {"all_n":>6} {"valid_n":>8} {"acc(valid)":>12} {"Wilson 95% CI":>22} {"half-width":>12}')
    print('-' * 80)
    for qt in ('explanatory', 'predictive', 'counterfactual'):
        c = counts[qt]
        p, lo, hi = wilson(c['correct_valid'], c['valid'])
        half = (hi - lo) / 2
        print(f'{qt:<16} {c["all"]:>6} {c["valid"]:>8} '
              f'{p * 100:>10.1f}% [{lo * 100:>5.1f}, {hi * 100:>5.1f}] '
              f'{half * 100:>10.1f}pp')


if __name__ == '__main__':
    main()
