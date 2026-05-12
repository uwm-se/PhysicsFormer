"""Inspect predictions across Phase 9 ablation runs and aggregate.

Reads eval_heldout_type_n200_<config>.json files and prints
side-by-side per-type accuracy + a sample of predictions per qa_type
for diagnosing what each layer is doing.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / 'results' / 'phase9_diverse_grounding'

CONFIGS = [
    ('baseline', 'eval_heldout_type_n200_baseline.json'),
    ('cue', 'eval_heldout_type_n200_cue.json'),
    ('cue+shots', 'eval_heldout_type_n200_cue_shots.json'),
    ('cd', 'eval_heldout_type_n200_cd.json'),
    ('cue+cd', 'eval_heldout_type_n200_cue_cd.json'),
]


def _load(name):
    p = ROOT / name
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def main() -> None:
    print()
    print('=' * 96)
    print('Phase 9 held-out-type ablation: per-type substring accuracy (%)')
    print('=' * 96)
    qa_types = ['collision_prediction', 'kinetic_energy', 'mass_comparison', 'speed_comparison', 'time_to_event']
    header = f'{"qa_type":<26}'
    for label, _ in CONFIGS:
        header += f'{label:>12}'
    print(header)
    rows = {label: _load(name) for label, name in CONFIGS}

    def _cell(v):
        if isinstance(v, (int, float)):
            return f'{v:>10.2f}% '
        return f'{"-":>12}'

    def _summary_of(d):
        # eval_phase9_heldout_type.py writes {summary: {overall, per_type, ...}, records}
        if d is None:
            return {}
        return d.get('summary', d)  # tolerate older flat-shape JSONs

    for qt in qa_types:
        line = f'{qt:<26}'
        for label, _ in CONFIGS:
            s = _summary_of(rows.get(label))
            v = s.get('per_type', {}).get(qt, {}).get('substring_acc_pct')
            line += _cell(v)
        print(line)
    line = f'{"OVERALL":<26}'
    for label, _ in CONFIGS:
        s = _summary_of(rows.get(label))
        v = s.get('overall', {}).get('substring_acc_pct')
        line += _cell(v)
    print(line)

    # Sample a few predictions per config for diagnosis.
    print()
    print('=' * 96)
    print('Sample predictions (first 3 per qa_type per config)')
    print('=' * 96)
    for qt in qa_types:
        print(f'\n--- {qt} ---')
        for label, _ in CONFIGS:
            d = rows.get(label)
            if d is None:
                continue
            picks = [r for r in d.get('records', []) if r.get('qa_type') == qt][:3]
            print(f'  [{label}]')
            for r in picks:
                pred = (r.get('predicted') or '').replace('\n', ' ')
                if len(pred) > 70:
                    pred = pred[:67] + '...'
                gold = r.get('gold_answer')
                ok = r.get('substring_correct')
                tag = 'OK ' if ok else '   '
                print(f'    {tag} gold={gold!r:<14} pred={pred!r}')


if __name__ == '__main__':
    main()
