"""Audit Ours' predictive supplement: count valid vs invalid MCQ items in
phase3_BASELINE_SHUFFLE_FULL5000.details.jsonl and compute valid-only accuracy.

A question is 'valid' when it has at least one choice labeled correct
(matches compute_paper_stats.py:163 valid_only filter).
"""
import json
from pathlib import Path

DETAILS = Path(__file__).resolve().parent.parent / "results" / "phase3_BASELINE_SHUFFLE_FULL5000.details.jsonl"

if not DETAILS.exists():
    raise SystemExit(f"Missing details file: {DETAILS}")

per_type_total = {"explanatory": 0, "predictive": 0, "counterfactual": 0, "descriptive": 0, "other": 0}
per_type_valid = {"explanatory": 0, "predictive": 0, "counterfactual": 0, "descriptive": 0, "other": 0}
per_type_correct_total = {"explanatory": 0, "predictive": 0, "counterfactual": 0, "descriptive": 0, "other": 0}
per_type_correct_valid = {"explanatory": 0, "predictive": 0, "counterfactual": 0, "descriptive": 0, "other": 0}

# Collect a few sample invalid predictive questions for inspection
invalid_samples = []

with open(DETAILS, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        qt = r.get("question_type", r.get("clevrer_type", "other"))
        if qt not in per_type_total:
            qt = "other"
        is_valid = bool(r.get("correct_choices") or [])
        is_correct = bool(r.get("correct"))

        per_type_total[qt] += 1
        if is_correct:
            per_type_correct_total[qt] += 1
        if is_valid:
            per_type_valid[qt] += 1
            if is_correct:
                per_type_correct_valid[qt] += 1
        elif qt == "predictive" and len(invalid_samples) < 3:
            invalid_samples.append(r)

print("=== Phase 3 FULL5000 details audit ===")
print(f"{'type':<16} {'total':>8} {'valid':>8} {'invalid':>8} | {'all-acc':>8} {'valid-acc':>10}")
print("-" * 72)
for qt in ("explanatory", "predictive", "counterfactual", "descriptive", "other"):
    n = per_type_total[qt]
    nv = per_type_valid[qt]
    if n == 0:
        continue
    inv = n - nv
    acc_all = per_type_correct_total[qt] / n * 100 if n else 0
    acc_valid = per_type_correct_valid[qt] / nv * 100 if nv else 0
    print(f"{qt:<16} {n:>8} {nv:>8} {inv:>8} | {acc_all:>7.2f}% {acc_valid:>9.2f}%")

print()
print("=== Summary for predictive (Ours supplement) ===")
n_pred = per_type_total["predictive"]
nv_pred = per_type_valid["predictive"]
c_pred = per_type_correct_total["predictive"]
cv_pred = per_type_correct_valid["predictive"]
print(f"  ALL predictive:    n={n_pred:,}  correct={c_pred:,}  acc={c_pred/n_pred*100:.2f}%")
print(f"  VALID predictive:  n={nv_pred:,}  correct={cv_pred:,}  acc={cv_pred/nv_pred*100:.2f}%")
print(f"  Invalid (excluded by validate_question filter): {n_pred - nv_pred:,} "
      f"({(n_pred - nv_pred)/n_pred*100:.1f}%)")

if invalid_samples:
    print()
    print("=== Sample invalid predictive questions (no correct choice) ===")
    for i, r in enumerate(invalid_samples, 1):
        print(f"  [{i}] scene={r.get('scene_id')}  q_idx={r.get('question_idx', '?')}")
        print(f"      question: {r.get('question', '')[:120]}")
        choices = r.get("choices", [])
        for c in choices[:4]:
            ans = c.get("answer", "?") if isinstance(c, dict) else "?"
            txt = c.get("choice", c) if isinstance(c, dict) else c
            print(f"        ({ans}) {str(txt)[:80]}")
