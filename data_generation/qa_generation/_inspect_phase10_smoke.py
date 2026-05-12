"""One-off inspection of a Phase 10 smoke-test dataset.

Verifies that --include_answer_cue=mix produces a roughly 50/50 split
of cued and bare questions, with cues appearing only on qa_types that
have a registered cue in TRAINED_TYPE_CUES (free-form types pass
through unchanged).
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path


def main(path: Path) -> None:
    recs = json.load(open(path, 'r', encoding='utf-8'))
    counts: dict = defaultdict(lambda: {'cued': 0, 'bare': 0})
    for r in recs:
        qa_type = r.get('qa_type', '')
        q = r.get('question', '')
        has_cue = ('(' in q and q.endswith(')'))
        bucket = 'cued' if has_cue else 'bare'
        counts[qa_type][bucket] += 1
        tag = 'CUED' if has_cue else 'BARE'
        q_short = q[:96]
        print(f'  {tag}  [{qa_type:<24}]  q={q_short!r}  target={r.get("target")!r}')
    print()
    print(f'total: {len(recs)} records')
    print()
    print('per-qa_type (cued / bare):')
    for qt, c in sorted(counts.items()):
        print(f'  {qt:<28}  cued={c["cued"]:>3}  bare={c["bare"]:>3}')


if __name__ == '__main__':
    p = Path(sys.argv[1] if len(sys.argv) > 1
             else 'compsac_2026_code/data/phase9/_smoke_phase10.json')
    main(p)
