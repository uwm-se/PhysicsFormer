"""Tiny one-shot helper to print the first N free-form transfer records
from a results JSON file.

This exists because PowerShell + python -c heredoc is fragile and we
need a stable way to eyeball model outputs after each eval run. Keep it
small; it has no business growing into a real CLI tool.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--n', type=int, default=8)
args = parser.parse_args()

doc = json.loads(Path(args.path).read_text())
recs = doc.get('records', doc) if isinstance(doc, dict) else doc
print(f'file:    {args.path}')
print(f'records: {len(recs)} (showing first {min(args.n, len(recs))})')

# Try to fish the summary stats out of whichever schema the file uses.
summary = doc.get('summary') if isinstance(doc, dict) else None
if summary:
    print('summary:')
    for k, v in summary.items():
        print(f'  {k}: {v}')

for i, r in enumerate(recs[: args.n]):
    qt = r.get('question_type', r.get('q_type', '?'))
    q = r.get('question_text', r.get('question', ''))
    a = r.get('answer_text', r.get('expected_answer', r.get('answer', '')))
    ff = r.get('freeform_pred_text', r.get('freeform_pred',
              r.get('ff_pred', r.get('pred_text', ''))))
    mcq = r.get('mcq_pred_text', r.get('mcq_pred', r.get('mcq', '')))
    print(f'\n--- {i + 1} [{qt}] ---')
    print(f'  Q:        {q[:120]}')
    print(f'  Expected: {a[:140]}')
    print(f'  FF gen:   {str(ff)[:200]}')
    if mcq:
        print(f'  MCQ gen:  {str(mcq)[:200]}')
