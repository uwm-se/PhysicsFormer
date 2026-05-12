"""Compute Wilson 95% CIs for all 1K results.

Importable API (used by plot_ci_forest.py):
    wilson_ci(k, n)          -> (lo, hi) on [0, 1]
    load_per_type(name, fname) -> dict with overall/explanatory/predictive/counterfactual,
                                   each containing {correct, total, p, lo, hi}
    FILES                    -> ordered model -> JSON-filename map for the 1K pool
    HELDOUT                  -> held-out Grounded-Physics LM counts (n=1998, recorded
                                from compute_paper_stats.py --heldout --valid_only)

Run as a script: prints LaTeX table rows + per-type n values (legacy behavior).
"""
import math, json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Sentinel filename used to mark the Grounded-Physics LM row as sourcing
# its values from the held-out 10% partition (HELDOUT dict below) rather
# than from a 1K-pool JSON. Matching this string in load_per_type returns
# heldout_per_type() instead of reading a file.
HELDOUT_SENTINEL = "__heldout__"

# Ordered map preserving the article's table row order. The first row
# (Grounded-Physics LM) sources its values from the held-out partition --
# the only training-disjoint subset for our model. LLM baselines come from
# the 1K-pool zero-shot runs (LLMs never train on CLEVRER, so any 1K
# subset is an unbiased zero-shot measurement for them).
FILES = {
    "Grounded-Physics LM (held-out)": HELDOUT_SENTINEL,
    "Llama-3.3-70B": "llama-3.3-70b_with_scene_FULL.json",
    "Qwen3-235B": "qwen-72b_with_scene_FULL.json",
    "Claude 4.5 Sonnet": "claude-4.5_with_scene_FULL.json",
    "GPT-4o": "gpt4_with_scene_FULL.json",
    "Gemini 2.0 Flash": "gemini_with_scene_FULL.json",
    "Qwen2.5-7B": "qwen-7b_with_scene_FULL.json",
    "Claude Sonnet 4": "claude_with_scene_FULL.json",
    "DeepSeek-V3": "deepseek-v3_with_scene_FULL.json",
    "GPT-4o (no-tools)": "gpt4_with_scene_notools_FULL.json",
    "Gemini 2.0 Flash (no-tools)": "gemini_with_scene_notools_FULL.json",
    "Claude 4.5 Sonnet (no-tools)": "claude-4.5_with_scene_notools_FULL.json",
    "Claude Sonnet 4 (no-tools)": "claude_with_scene_notools_FULL.json",
}

# 15-object 1K-pool stress-test results. Row order matches
# tab:main_results_15obj (overall accuracy descending; with-tools first,
# then no-tools dagger rows). Gemini 2.0 Flash no-tools is dropped from
# the paper: its run could not be completed due to sustained Gemini API
# 503 overload during the resume attempt (the partial 342/998 file
# remains on disk under results/15obj_1k/gemini_15obj_notool.json for
# transparency, but is not used).
FILES_15OBJ = {
    "Grounded-Physics LM (1K pool)": "15obj_1k/physics_llm_15obj_1k.json",
    "Llama-3.3-70B": "15obj_1k/llama-3.3-70b_15obj_withtool.json",
    "Gemini 2.0 Flash": "15obj_1k/gemini_15obj_withtool.json",
    "Claude Sonnet 4": "15obj_1k/claude_15obj_withtool.json",
    "Qwen3-235B": "15obj_1k/qwen-72b_15obj_withtool.json",
    "Claude 4.5 Sonnet": "15obj_1k/claude-4.5_15obj_withtool.json",
    "DeepSeek-V3": "15obj_1k/deepseek-v3_15obj_withtool.json",
    "Qwen2.5-7B": "15obj_1k/qwen-7b_15obj_withtool.json",
    "GPT-4o": "15obj_1k/gpt4_15obj_withtool.json",
    "Claude Sonnet 4 (no-tools)": "15obj_1k/claude_15obj_notool.json",
    "Claude 4.5 Sonnet (no-tools)": "15obj_1k/claude-4.5_15obj_notool.json",
    "GPT-4o (no-tools)": "15obj_1k/gpt4_15obj_notool.json",
}

# Held-out Grounded-Physics LM counts on the n=1998 valid-only pool (shuffled MCQ).
# Source: compute_paper_stats.py --heldout --valid_only PRIMARY row.
HELDOUT = {
    "overall":        {"correct": 1383, "total": 1998},
    "explanatory":    {"correct":  564, "total":  710},
    "predictive":     {"correct":  229, "total":  361},
    "counterfactual": {"correct":  590, "total":  927},
}


def wilson_ci(k, n, z=1.96):
    """Wilson 95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def binomial_se(k, n):
    """Binomial standard error of a proportion ($\\sqrt{p(1-p)/n}$).

    This is the per-eval-set standard error analogous to the SEM used in
    NeurIPS/ICML tables; for the multi-seed case it reduces to std/sqrt(n)
    which is what readers will assume the $\\pm$ refers to. Returned in
    the same units as p (i.e., on [0, 1]).
    """
    if n == 0:
        return 0.0
    p = k / n
    return math.sqrt(p * (1 - p) / n)


def _entry(correct, total):
    p = correct / total if total else 0.0
    lo, hi = wilson_ci(correct, total)
    se = binomial_se(correct, total)
    return {"correct": correct, "total": total, "p": p,
            "lo": lo, "hi": hi, "se": se}


def _predictive_supplement_path(fname: str) -> Path:
    """Map a 1K-pool result JSON filename to its _PREDICTIVE.json counterpart.

    The supplement contains a predictive-only run at higher n than the
    ~163 captured by the natural distribution of the 1K pool. Two naming
    conventions are supported:

        gpt4_with_scene_FULL.json           -> gpt4_with_scene_PREDICTIVE.json
        claude_with_scene_notools_FULL.json -> claude_with_scene_notools_PREDICTIVE.json
        adapter_phase3_1k_sampled.json      -> adapter_phase3_1k_sampled_PREDICTIVE.json

    The first two patterns are produced by ``run_predictive_supplement.py``
    (LLM baselines). The third pattern is the Ours-side supplement, which
    can be produced by ``extract_ours_predictive_supplement.py`` from the
    existing FULL5000 measurement (no re-eval needed because Ours already
    has all ~3557 predictive items in ``phase3_BASELINE_SHUFFLE_FULL5000.json``).
    """
    name = Path(fname).name
    if name.endswith("_FULL.json"):
        new = name[: -len("_FULL.json")] + "_PREDICTIVE.json"
    else:
        # Insert _PREDICTIVE before the .json extension.
        new = name[: -len(".json")] + "_PREDICTIVE.json"
    return RESULTS_DIR / new


def _read_predictive_supplement(fname: str):
    """Return ``(correct, total)`` for the predictive supplement, or None.

    Returns ``None`` when no supplement file exists for ``fname``; callers
    should fall back to the 1K-pool predictive subset in that case.
    """
    path = _predictive_supplement_path(fname)
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    if "by_clevrer_type" in d:
        bt = d["by_clevrer_type"]
    else:
        bt = d.get("by_type", {})
    pred = bt.get("predictive", {})
    correct = pred.get("correct", 0)
    total = pred.get("total", 0)
    if total == 0:
        return None
    return correct, total


def load_per_type(fname, *, predictive_supplement: bool = True):
    """Load a result JSON and return per-type {correct, total, p, lo, hi}.

    Routing:

    * ``fname == HELDOUT_SENTINEL`` -> return ``heldout_per_type()`` directly
      (used for the Grounded-Physics LM row, whose primary numbers come
      from the disjoint 10% held-out partition rather than a 1K-pool JSON).
      Predictive-supplement substitution does not apply here: the held-out
      predictive cell is bounded at $n{=}361$ by construction (going outside
      that pool would re-introduce training-data leakage).
    * Any other ``fname`` -> read the 1K-pool result JSON. When
      ``predictive_supplement`` is True (default) and a sibling
      ``_PREDICTIVE.json`` file exists, the ``predictive`` entry is replaced
      with the larger-n supplement measurement. The other entries (overall,
      explanatory, counterfactual) continue to come from the 1K pool. This
      means downstream plotting (``plot_ci_forest.py``) automatically picks
      up the tightened predictive CI whenever the supplementary evaluation
      has been run, without any other code changes.

    Pass ``predictive_supplement=False`` to force the historical 1K-pool-only
    behavior for LLM rows (e.g., to reproduce the originally submitted
    numbers); the flag is silently ignored for the held-out sentinel.
    """
    if fname == HELDOUT_SENTINEL:
        return heldout_per_type()

    with open(RESULTS_DIR / fname) as f:
        d = json.load(f)
    if "by_clevrer_type" in d:
        bt = d["by_clevrer_type"]
        total = d["overall"]["total_questions"]
        correct = d["overall"]["correct"]
    else:
        bt = d["by_type"]
        total = d["total"]
        correct = d["correct"]

    out = {"overall": _entry(correct, total)}
    for t in ("explanatory", "predictive", "counterfactual"):
        e = bt.get(t, {})
        out[t] = _entry(e.get("correct", 0), e.get("total", 0))

    if predictive_supplement:
        sup = _read_predictive_supplement(fname)
        if sup is not None:
            out["predictive"] = _entry(*sup)

    return out


def heldout_per_type():
    """Held-out Grounded-Physics LM per-type entries (with Wilson CIs)."""
    return {t: _entry(v["correct"], v["total"]) for t, v in HELDOUT.items()}


def _fmt(k, n):
    pct = k / n * 100
    lo, hi = wilson_ci(k, n)
    return f"{pct:.1f}\\% $[{lo*100:.1f}, {hi*100:.1f}]$"


def _fmt_sem(k, n, *, percent_sign=True):
    """Render a result as 'XX.X $\\pm$ Y.Y' using 1\\times binomial SE.

    Both the point estimate and the SE are reported in percentage points
    (matching the existing tables). The optional ``percent_sign`` toggle
    controls whether to append ``\\%`` after the point estimate to keep
    cell width down in dense tables.
    """
    pct = k / n * 100 if n else 0.0
    se_pp = binomial_se(k, n) * 100
    suffix = "\\%" if percent_sign else ""
    return f"{pct:.1f}{suffix} $\\pm$ {se_pp:.1f}"


def render_sem_rows(files_map, header, *, percent_sign=True):
    """Print mean $\\pm$ SE LaTeX rows for a {name -> filename} mapping."""
    suffix = "with %" if percent_sign else "no %"
    print(f"=== {header} (mean $\\pm$ 1\\times binomial SE, {suffix}) ===")
    for name, fname in files_map.items():
        per = load_per_type(fname)
        kw = {"percent_sign": percent_sign}
        ov = _fmt_sem(per["overall"]["correct"], per["overall"]["total"], **kw)
        ex = _fmt_sem(per["explanatory"]["correct"], per["explanatory"]["total"], **kw)
        pr = _fmt_sem(per["predictive"]["correct"], per["predictive"]["total"], **kw)
        cn = _fmt_sem(per["counterfactual"]["correct"], per["counterfactual"]["total"], **kw)
        dagger = "$^\\dagger$" if "(no-tools)" in name else ""
        clean = name.replace(" (no-tools)", "")
        print(f"{clean}{dagger} & {ov} & {ex} & {pr} & {cn} \\\\")
    print()


def main():
    print("=== Wilson 95% CIs (legacy bracket form) ===\n")
    print("LaTeX table rows (Grounded-Physics LM = held-out partition; LLMs = 1K-pool):\n")
    for name, fname in FILES.items():
        per = load_per_type(fname)
        ov = _fmt(per["overall"]["correct"], per["overall"]["total"])
        ex = _fmt(per["explanatory"]["correct"], per["explanatory"]["total"])
        pr = _fmt(per["predictive"]["correct"], per["predictive"]["total"])
        cn = _fmt(per["counterfactual"]["correct"], per["counterfactual"]["total"])
        dagger = "$^\\dagger$" if "(no-tools)" in name else ""
        clean = name.replace(" (no-tools)", "")
        print(f"{clean}{dagger} & {ov} & {ex} & {pr} & {cn} \\\\")

    print()
    print("=== Per-type n values ===")
    per = load_per_type("gpt4_with_scene_FULL.json")
    print("GPT-4o (1K-pool, with predictive-supplement substitution):")
    for t in ("explanatory", "predictive", "counterfactual"):
        print(f"  {t}: n={per[t]['total']}")
    per = load_per_type(HELDOUT_SENTINEL)
    print("Grounded-Physics LM (held-out partition):")
    for t in ("explanatory", "predictive", "counterfactual"):
        print(f"  {t}: n={per[t]['total']}")
    print()

    # SEM-rendered LaTeX rows for the article tables. The first row in each
    # table comes from FILES and uses HELDOUT_SENTINEL for Grounded-Physics
    # LM; LLM rows use their 1K-pool zero-shot files.
    render_sem_rows(FILES, "tab:ci_baselines (Ours=held-out, LLMs=1K-pool)",
                    percent_sign=True)
    render_sem_rows(FILES, "tab:main_results (Ours=held-out, LLMs=1K-pool, no %)",
                    percent_sign=False)
    # tab:main_results_15obj uses bare numbers (no %). 15-obj is a separate
    # stress test (the 15-object scene generation is not restricted to the
    # held-out partition); both Ours and LLMs are evaluated on the same
    # 15-obj 1K pool.
    render_sem_rows(FILES_15OBJ, "tab:main_results_15obj (15-obj 1K pool, no %)",
                    percent_sign=False)


if __name__ == "__main__":
    main()
