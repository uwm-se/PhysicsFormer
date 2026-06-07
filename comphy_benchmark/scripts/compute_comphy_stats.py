"""Wilson 95% CIs + markdown / LaTeX table for ComPhy zero-shot eval.

Consumes the JSON written by ``run_comphy_evaluation.py`` and (optionally) the
``.details.jsonl`` sidecar so it can compute charge-conditioned slices.

Example::

    python comphy_benchmark/scripts/compute_comphy_stats.py \\
        --result comphy_benchmark/results/phase3_comphy_zeroshot.json \\
        --emit_latex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the same Wilson interval the CLEVRER paper stats use.
_HERE = Path(__file__).resolve()
_CLEVRER_SCRIPTS = _HERE.parent.parent.parent / "clevrer_benchmark" / "scripts"
sys.path.insert(0, str(_CLEVRER_SCRIPTS))
from compute_paper_stats import wilson_ci  # type: ignore  # noqa: E402


def _fmt(p: float, lo: float, hi: float) -> str:
    return f"{p * 100:5.1f}% [{lo * 100:4.1f}, {hi * 100:4.1f}]"


def _load_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_details(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _bucket_from_summary(summary: Dict[str, Any], key: str) -> List[Tuple[str, int, int]]:
    """Return [(label, correct, total)] from a summary bucket like 'by_coarse_type'."""
    items = []
    for label, v in summary.get(key, {}).items():
        items.append((label, int(v["correct"]), int(v["total"])))
    return items


def _print_table(rows: List[Tuple[str, int, int]], title: str) -> None:
    print(f"\n## {title}")
    print(f"\n| {'Type':<24} | {'n':>5} | {'Accuracy':>9} | {'Wilson 95% CI':>18} |")
    print(f"| {'-'*24} | {'-'*5} | {'-'*9} | {'-'*18} |")
    for label, c, n in rows:
        p, lo, hi = wilson_ci(c, n)
        ci = f"[{lo*100:.1f}, {hi*100:.1f}]"
        print(f"| {label:<24} | {n:>5d} | {p*100:>8.1f}% | {ci:>18} |")


def _emit_latex(rows: List[Tuple[str, int, int]], caption: str, label: str) -> None:
    print("\n% --- LaTeX fragment ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(rf"\caption{{{caption}}}")
    print(rf"\label{{{label}}}")
    print(r"\begin{tabular}{lrcc}")
    print(r"\toprule")
    print(r"Type & $n$ & Accuracy & Wilson 95\% CI \\")
    print(r"\midrule")
    for lbl, c, n in rows:
        p, lo, hi = wilson_ci(c, n)
        print(rf"{lbl} & {n} & {p*100:.1f}\% & [{lo*100:.1f}, {hi*100:.1f}] \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def _charge_slices(details: List[Dict[str, Any]]) -> List[Tuple[str, int, int]]:
    """Slice predictions by whether the question text mentions charge.

    This is a coarse keyword filter, not a parser. It exists so we can
    *honestly report* how much of the OOD performance comes from
    charge-independent questions vs charge-dependent ones the model can't
    fully observe (no charge slot in the 35-D state).
    """
    if not details:
        return []
    cm_correct = cm_total = 0
    ncm_correct = ncm_total = 0
    for r in details:
        text = (r.get("question_text") or "").lower()
        is_charge = ("charge" in text or "attract" in text or "repel" in text
                     or "magnet" in text)
        if is_charge:
            cm_total += 1
            cm_correct += int(bool(r.get("correct")))
        else:
            ncm_total += 1
            ncm_correct += int(bool(r.get("correct")))
    return [
        ("charge-independent", ncm_correct, ncm_total),
        ("charge-dependent (partial-obs)", cm_correct, cm_total),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="ComPhy stats / CIs")
    p.add_argument("--result", type=str, required=True,
                   help="Path to phase3_comphy_zeroshot.json (summary).")
    p.add_argument("--details", type=str, default=None,
                   help="Optional .details.jsonl path; defaults to result + .details.jsonl.")
    p.add_argument("--emit_latex", action="store_true",
                   help="Also emit a LaTeX table fragment for the coarse-type CIs.")
    args = p.parse_args()

    result_path = Path(args.result)
    details_path = Path(args.details) if args.details else result_path.with_suffix(".details.jsonl")

    summary = _load_summary(result_path)
    details = _load_details(details_path)

    overall_c = int(summary["overall"]["correct"])
    overall_n = int(summary["overall"]["total"])
    p_, lo, hi = wilson_ci(overall_c, overall_n)
    print(f"\nComPhy zero-shot OOD  ({result_path.name})")
    print("=" * 60)
    print(f"Overall:                 {_fmt(p_, lo, hi)}  (n={overall_n})")

    coarse = _bucket_from_summary(summary, "by_coarse_type")
    _print_table(coarse, "By coarse type")

    native = _bucket_from_summary(summary, "by_native_type")
    if native:
        _print_table(native, "By native ComPhy type")

    if details:
        charge_rows = _charge_slices(details)
        if any(n > 0 for _, _, n in charge_rows):
            _print_table(charge_rows, "Charge dependence (keyword heuristic on question text)")

    if args.emit_latex and coarse:
        _emit_latex(
            coarse,
            caption=(
                "Zero-shot ComPhy results for the CLEVRER-trained Physics-LLM "
                "adapter (no retraining)."
            ),
            label="tab:comphy_ood",
        )


if __name__ == "__main__":
    main()
