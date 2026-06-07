"""Offline alpha-sweep analysis for the ComPhy ``alpha_sweep`` eval.

Consumes the ``.details.jsonl`` written by ``run_comphy_evaluation.py
--eval_method alpha_sweep`` (each MC record carries per-choice ``real_scores``
and ``zero_scores`` -- the choice log-likelihoods with the physics prefix ON
and OFF). From that single pass it reconstructs the accuracy of every physics
weighting without re-running the model:

    combined(alpha) = real + alpha * (real - zero)        # contrastive rule
    physics_blind   = argmax(zero)                        # zeroed prefix == zero_physics ablation
    physics_only    = argmax(real)                        # alpha = 0
    delta_only      = argmax(real - zero)                 # alpha -> inf, pure physics uplift

The theory ("the CLEVRER physics grounding carries signal on ComPhy") predicts
accuracy should *rise* with alpha and the ``delta_only`` rule should beat both
chance and ``physics_blind``. A flat/declining curve is an honest negative.

Example::

    python comphy_benchmark/scripts/analyze_alpha_sweep.py \\
        --details comphy_benchmark/results/phase3_comphy_alphasweep.details.jsonl \\
        --emit_latex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve()
_CLEVRER_SCRIPTS = _HERE.parent.parent.parent / "clevrer_benchmark" / "scripts"
sys.path.insert(0, str(_CLEVRER_SCRIPTS))
from compute_paper_stats import wilson_ci  # type: ignore  # noqa: E402

# Default alpha grid. 0.0 == physics_only (no contrastive amplification);
# 1.0 == the published contrastive recipe / baseline; larger == more weight on
# the physics-delta term.
DEFAULT_ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def _argmax(xs: List[float]) -> int:
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def _choice_is_correct(predicted: str, ground_truth: str,
                       choices: List[Dict[str, Any]]) -> bool:
    """Replicate ``run_adapter_evaluation.evaluate_answer`` MC semantics.

    Strict: prediction must match a ``correct`` choice and not a ``wrong`` one.
    """
    pred = (predicted or "").strip().lower()
    gt = (ground_truth or "").strip().lower()
    if pred == gt:
        return True
    if choices:
        correct = [(_c.get("choice") or "").lower().strip()
                   for _c in choices if _c.get("answer") == "correct"]
        wrong = [(_c.get("choice") or "").lower().strip()
                 for _c in choices if _c.get("answer") == "wrong"]
        matches_correct = any(pred == c or c in pred or pred in c for c in correct)
        matches_wrong = any(pred == c or c in pred or pred in c for c in wrong)
        if matches_correct and not matches_wrong:
            return True
        if matches_correct and matches_wrong:
            return False
    return False


class _Counter:
    """Accumulate correct/total overall and by coarse type."""

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0
        self.by_coarse: Dict[str, List[int]] = {}

    def add(self, ok: bool, coarse: str) -> None:
        self.total += 1
        self.correct += int(ok)
        c = self.by_coarse.setdefault(coarse, [0, 0])
        c[1] += 1
        c[0] += int(ok)


def _load(details_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(details_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("real_scores") and r.get("zero_scores") and r.get("choices"):
                rows.append(r)
    return rows


def _eval_rule(rows: List[Dict[str, Any]], pick) -> _Counter:
    """``pick(real, zero) -> chosen index``. Returns a populated counter."""
    ctr = _Counter()
    for r in rows:
        real = [float(x) for x in r["real_scores"]]
        zero = [float(x) for x in r["zero_scores"]]
        choices = r["choices"]
        n = min(len(real), len(zero), len(choices))
        if n == 0:
            continue
        real, zero, choices = real[:n], zero[:n], choices[:n]
        idx = pick(real, zero)
        choice_texts = [(c.get("choice", c) if isinstance(c, dict) else str(c))
                        for c in choices]
        ok = _choice_is_correct(str(choice_texts[idx]), r.get("ground_truth", ""),
                                choices)
        ctr.add(ok, r.get("coarse_type", "?"))
    return ctr


def _row_str(label: str, c: int, n: int) -> str:
    p, lo, hi = wilson_ci(c, n)
    return f"{label:<26} {p*100:6.2f}%  [{lo*100:5.1f}, {hi*100:5.1f}]   (n={n})"


def _diagnose_delta(rows: List[Dict[str, Any]]) -> None:
    """Tell us *which link* limits the physics delta, from the logged scores.

    - delta spread ~ 0           -> prefix barely moves the LLM (link 1/2:
                                    normalize states / multi-frame / MCQ head)
    - physics overrides the prior
      but is wrong               -> grounding quality (conversion/normalization)
    - overrides are right more
      often than the prior        -> signal is real; amplifying alpha helps
    """
    spreads = []           # per-question max-min of the delta across choices
    abs_deltas = []        # |real-zero| per choice
    override_n = 0         # questions where physics changes the argmax vs blind
    override_phys_correct = 0   # ...and the physics pick is correct
    override_blind_correct = 0  # ...and the blind pick it replaced was correct
    for r in rows:
        real = [float(x) for x in r["real_scores"]]
        zero = [float(x) for x in r["zero_scores"]]
        choices = r["choices"]
        n = min(len(real), len(zero), len(choices))
        if n < 2:
            continue
        real, zero, choices = real[:n], zero[:n], choices[:n]
        delta = [a - b for a, b in zip(real, zero)]
        spreads.append(max(delta) - min(delta))
        abs_deltas.extend(abs(d) for d in delta)
        i_phys = _argmax([a + (a - b) for a, b in zip(real, zero)])  # alpha=1 pick
        i_blind = _argmax(zero)
        if i_phys != i_blind:
            override_n += 1
            texts = [(c.get("choice", c) if isinstance(c, dict) else str(c))
                     for c in choices]
            gt = r.get("ground_truth", "")
            override_phys_correct += int(_choice_is_correct(str(texts[i_phys]), gt, choices))
            override_blind_correct += int(_choice_is_correct(str(texts[i_blind]), gt, choices))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\nDelta diagnostics (why the physics signal is/ isn't strong)")
    print("-" * 64)
    print(f"mean |real-zero| per choice:        {_mean(abs_deltas):.4f}")
    print(f"mean per-question delta spread:     {_mean(spreads):.4f}  "
          f"(max-min across choices; ~0 => prefix inert)")
    n_q = len(spreads)
    if n_q:
        print(f"physics overrides language prior:   {override_n}/{n_q} "
              f"({override_n/n_q*100:.1f}% of questions)")
    if override_n:
        pc = override_phys_correct / override_n
        bc = override_blind_correct / override_n
        print(f"  when it overrides -> physics pick correct: {pc*100:.1f}%   "
              f"(prior it replaced was correct: {bc*100:.1f}%)")
        verdict = ("HELPS (override beats the prior it replaced)" if pc > bc
                   else "HURTS (override is worse than the prior)" if pc < bc
                   else "NEUTRAL")
        print(f"  override net effect: {verdict}  ({(pc-bc)*100:+.1f} pp)")


def main() -> None:
    ap = argparse.ArgumentParser(description="ComPhy alpha-sweep analysis")
    ap.add_argument("--details", type=str, required=True,
                    help="Path to the alpha_sweep .details.jsonl file.")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help=f"Alpha grid for the combined rule. Default: {DEFAULT_ALPHAS}")
    ap.add_argument("--by_coarse", action="store_true",
                    help="Also break each rule down by coarse type.")
    ap.add_argument("--emit_latex", action="store_true")
    args = ap.parse_args()

    alphas = args.alphas if args.alphas is not None else list(DEFAULT_ALPHAS)
    rows = _load(Path(args.details))
    if not rows:
        print("No alpha_sweep records (with real_scores/zero_scores) found.")
        return

    print(f"\nComPhy alpha-sweep  ({Path(args.details).name})")
    print("=" * 64)
    print(f"MC questions with logged components: {len(rows)}")
    print("\nDecision rule                 Accuracy   Wilson 95% CI")
    print("-" * 64)

    # Reference rules.
    blind = _eval_rule(rows, lambda real, zero: _argmax(zero))
    print(_row_str("physics_blind (zero)", blind.correct, blind.total))

    # Combined rule across the alpha grid.
    grid: List[Tuple[float, _Counter]] = []
    for a in alphas:
        ctr = _eval_rule(
            rows, lambda real, zero, a=a: _argmax(
                [r + a * (r - z) for r, z in zip(real, zero)]))
        grid.append((a, ctr))
        tag = "physics_only" if a == 0.0 else f"alpha={a:g}"
        print(_row_str(f"combined {tag}", ctr.correct, ctr.total))

    delta = _eval_rule(
        rows, lambda real, zero: _argmax([r - z for r, z in zip(real, zero)]))
    print(_row_str("delta_only (alpha->inf)", delta.correct, delta.total))

    # Verdict: does weighting the physics delta beat the physics-blind rule?
    best_a, best_ctr = max(grid + [(float("inf"), delta)],
                           key=lambda t: t[1].correct / max(1, t[1].total))
    best_acc = best_ctr.correct / max(1, best_ctr.total)
    blind_acc = blind.correct / max(1, blind.total)
    print("-" * 64)
    b_p, b_lo, b_hi = wilson_ci(best_ctr.correct, best_ctr.total)
    z_p, z_lo, z_hi = wilson_ci(blind.correct, blind.total)
    best_tag = "delta_only" if best_a == float("inf") else f"alpha={best_a:g}"
    print(f"Best physics rule: {best_tag}  ->  {best_acc*100:.2f}% "
          f"[{b_lo*100:.1f}, {b_hi*100:.1f}]")
    print(f"Physics-blind:                 {blind_acc*100:.2f}% "
          f"[{z_lo*100:.1f}, {z_hi*100:.1f}]")
    sep = "DISJOINT (physics signal is real)" if b_lo > z_hi else \
        "OVERLAPPING (no significant physics effect)"
    print(f"Uplift: {(best_acc - blind_acc)*100:+.2f} pp   CIs: {sep}")
    print(f"Chance (2-choice): 50.00%   best vs chance: "
          f"{'ABOVE' if b_lo > 0.5 else 'NOT above'} (CI lo={b_lo*100:.1f})")

    _diagnose_delta(rows)

    if args.by_coarse:
        print("\nBy coarse type")
        print("-" * 64)
        for name, ctr in [("physics_blind", blind),
                          (f"best ({best_tag})", best_ctr),
                          ("delta_only", delta)]:
            print(f"\n  {name}")
            for ck in sorted(ctr.by_coarse):
                cc, cn = ctr.by_coarse[ck]
                print("    " + _row_str(ck, cc, cn))

    if args.emit_latex:
        print("\n% --- LaTeX fragment ---")
        print(r"\begin{table}[t]\centering")
        print(r"\caption{ComPhy physics-uplift sweep: accuracy of the "
              r"contrastive rule $s_{\text{real}}+\alpha(s_{\text{real}}-"
              r"s_{\text{zero}})$ as a function of the physics weight $\alpha$. "
              r"$\alpha{=}0$ ignores the contrastive term; physics-blind ranks "
              r"by the zeroed-prefix likelihood.}")
        print(r"\label{tab:comphy_alpha_sweep}")
        print(r"\begin{tabular}{lrc}\toprule")
        print(r"Rule & Accuracy & Wilson 95\% CI \\\midrule")
        p, lo, hi = wilson_ci(blind.correct, blind.total)
        print(rf"physics-blind & {p*100:.1f}\% & [{lo*100:.1f}, {hi*100:.1f}] \\")
        for a, ctr in grid:
            p, lo, hi = wilson_ci(ctr.correct, ctr.total)
            tag = r"$\alpha{=}0$" if a == 0.0 else rf"$\alpha{{=}}{a:g}$"
            print(rf"{tag} & {p*100:.1f}\% & [{lo*100:.1f}, {hi*100:.1f}] \\")
        p, lo, hi = wilson_ci(delta.correct, delta.total)
        print(rf"delta-only & {p*100:.1f}\% & [{lo*100:.1f}, {hi*100:.1f}] \\")
        print(r"\bottomrule\end{tabular}\end{table}")


if __name__ == "__main__":
    main()
