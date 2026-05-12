"""Generate Wilson 95% CI forest plot for CLEVRER benchmark results.

Reuses the wilson_ci / load_per_type / FILES / HELDOUT API from
``_compute_1k_cis.py`` so the figure values are guaranteed to match
``tab:ci_baselines`` in the article.

Outputs a 2x2 panel PDF (Overall / Explanatory / Predictive / Counterfactual)
to ``compsac_2026_code/article/ci_forest.pdf`` for inclusion in main.tex.

Usage::

    python compsac_2026_code/clevrer_benchmark/scripts/plot_ci_forest.py

The figure highlights:
  * Grounded-Physics LM (held-out) and (1K pool) with a distinct color
  * Wilson 95% CI as horizontal error bars
  * A vertical reference line at the 25% MCQ-random baseline
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

from _compute_1k_cis import FILES, FILES_15OBJ, load_per_type


# IEEE-conference-friendly typography (10pt body, 8pt for figure labels).
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
})

# Color scheme: Physics-LLM in distinct red/orange, LLMs in muted blue, no-tools in
# darker blue, random baseline in a desaturated red so it cannot be confused with
# the gray gridlines, ours-row highlight in a very light tint of COLOR_OURS.
COLOR_OURS = "#C73E1D"       # Grounded-Physics LM (held-out partition, primary row)
COLOR_OURS_BAND = "#FBE9E4"  # Light tint of COLOR_OURS for the Ours-row background highlight
COLOR_LLM = "#2E86AB"        # LLM with-tools baselines
COLOR_NOTOOLS = "#1F4E5F"    # LLM no-tools baselines
COLOR_RANDOM = "#A63D40"     # 25% MCQ random reference (desaturated red, distinct from gridlines)
COLOR_DIVIDER = "#B8B8B8"    # Horizontal divider between with-tools and no-tools groups

# Display labels in plot order (top -> bottom). Grounded-Physics LM (Ours)
# row sources its values from the held-out 10% partition (n=1,998 valid
# items, training-disjoint); LLM rows are from the 1K-pool zero-shot runs
# (with predictive-supplement substitution where present). LLMs are
# ordered by overall accuracy descending (matches tab:ci_baselines), then
# no-tools dagger rows.
ROWS = [
    ("Grounded-Physics LM (Ours)",      "ours"),
    ("Llama-3.3-70B",                   "llm"),
    ("Qwen3-235B",                      "llm"),
    ("Claude 4.5 Sonnet",               "llm"),
    ("GPT-4o",                          "llm"),
    ("Gemini 2.0 Flash",                "llm"),
    ("Qwen2.5-7B",                      "llm"),
    ("Claude Sonnet 4",                 "llm"),
    ("DeepSeek-V3",                     "llm"),
    ("GPT-4o$^\\dagger$",               "notools"),
    ("Gemini 2.0 Flash$^\\dagger$",     "notools"),
    ("Claude 4.5 Sonnet$^\\dagger$",    "notools"),
    ("Claude Sonnet 4$^\\dagger$",      "notools"),
]

# 15-object row order (matches tab:main_results_15obj, descending overall).
# Gemini 2.0 Flash$^\dagger$ is omitted because its no-tools 15-obj run
# could not be completed (sustained Gemini API 503 overload); the row is
# dropped from the paper rather than reported with partial counts. See
# FILES_15OBJ in `_compute_1k_cis.py` for the matching note. The 15-obj
# pool is a separate stress test (15-object scene generation is not
# restricted to the held-out partition); both Ours and LLMs are evaluated
# on the same 15-obj 1K pool, so this plot uses the 1K-pool source for
# Ours rather than the held-out source.
ROWS_15OBJ = [
    ("Grounded-Physics LM (Ours)",      "ours"),
    ("Llama-3.3-70B",                   "llm"),
    ("Gemini 2.0 Flash",                "llm"),
    ("Claude Sonnet 4",                 "llm"),
    ("Qwen3-235B",                      "llm"),
    ("Claude 4.5 Sonnet",               "llm"),
    ("DeepSeek-V3",                     "llm"),
    ("Qwen2.5-7B",                      "llm"),
    ("GPT-4o",                          "llm"),
    ("Claude Sonnet 4$^\\dagger$",      "notools"),
    ("Claude 4.5 Sonnet$^\\dagger$",    "notools"),
    ("GPT-4o$^\\dagger$",               "notools"),
]

# Map display-label -> _compute_1k_cis FILES / FILES_15OBJ key. For the
# 3--6 obj plot, "Grounded-Physics LM (Ours)" routes to the held-out entry
# (FILES["Grounded-Physics LM (held-out)"], whose value is HELDOUT_SENTINEL).
# For the 15-obj plot, the same label routes to the 15-obj 1K-pool entry
# (FILES_15OBJ["Grounded-Physics LM (1K pool)"]). The two key maps are
# kept separate to make the difference explicit.
KEY_MAP = {
    "Grounded-Physics LM (Ours)": "Grounded-Physics LM (held-out)",
    "Llama-3.3-70B": "Llama-3.3-70B",
    "Qwen3-235B": "Qwen3-235B",
    "Claude 4.5 Sonnet": "Claude 4.5 Sonnet",
    "GPT-4o": "GPT-4o",
    "Gemini 2.0 Flash": "Gemini 2.0 Flash",
    "Qwen2.5-7B": "Qwen2.5-7B",
    "Claude Sonnet 4": "Claude Sonnet 4",
    "DeepSeek-V3": "DeepSeek-V3",
    "GPT-4o$^\\dagger$": "GPT-4o (no-tools)",
    "Gemini 2.0 Flash$^\\dagger$": "Gemini 2.0 Flash (no-tools)",
    "Claude 4.5 Sonnet$^\\dagger$": "Claude 4.5 Sonnet (no-tools)",
    "Claude Sonnet 4$^\\dagger$": "Claude Sonnet 4 (no-tools)",
}

# Separate map for the 15-obj plot: there, "Grounded-Physics LM (Ours)"
# uses the 15-obj 1K-pool result (no held-out variant exists for 15-obj).
KEY_MAP_15OBJ = {
    **KEY_MAP,
    "Grounded-Physics LM (Ours)": "Grounded-Physics LM (1K pool)",
}

PANELS = [
    ("Overall",        "overall"),
    ("Explanatory",    "explanatory"),
    ("Predictive",     "predictive"),
    ("Counterfactual", "counterfactual"),
]


def _row_data(rows=ROWS, files=FILES, key_map=KEY_MAP):
    """Yield (display_label, kind, per_type_dict) for each row in plot order.

    All rows are loaded from `files` via `load_per_type` after mapping the
    display label through `key_map`. The Grounded-Physics LM row routes
    through ``HELDOUT_SENTINEL`` for the 3--6 obj plot (giving the held-out
    partition counts) and through the 15-obj 1K-pool file for the 15-obj
    plot (since 15-obj has no held-out variant -- it is a separate stress
    test).
    """
    cache: dict[str, dict] = {}
    for label, kind in rows:
        key = key_map[label]
        if key not in cache:
            cache[key] = load_per_type(files[key])
        yield label, kind, cache[key]


def _color_for(kind: str) -> str:
    if kind == "ours":
        return COLOR_OURS
    if kind == "notools":
        return COLOR_NOTOOLS
    return COLOR_LLM


def make_figure(
    out_path: Path,
    rows=ROWS,
    files=FILES,
    key_map=KEY_MAP,
    *,
    figsize: tuple[float, float] = (7.0, 2.6),
) -> None:
    """Render a 1x4 forest plot of Wilson 95% CIs for the given rows + files.

    Args:
        out_path: PDF path; a sibling .png is written alongside.
        rows: list of (display_label, kind) tuples in top-to-bottom plot order.
        files: model-name -> result-JSON-path dict (relative to results/).
        key_map: display-label -> ``files`` key map. Defaults to ``KEY_MAP``
            (3--6 obj). Pass ``KEY_MAP_15OBJ`` for the 15-obj plot.
        figsize: matplotlib figure size in inches. The default ``(7.0, 2.6)``
            is calibrated for the 13-row 3--6-obj plot (one less row than
            before since the held-out sub-row was merged into the primary
            Ours row). For the 12-row 15-obj plot a proportionally smaller
            height keeps the post-tight-bbox aspect ratio matched.
    """
    row_data = list(_row_data(rows=rows, files=files, key_map=key_map))
    n = len(row_data)
    # y positions: top row at the top of the plot.
    y_positions = list(range(n - 1, -1, -1))

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

    for ax, (panel_title, key) in zip(axes, PANELS):
        for y, (label, kind, per) in zip(y_positions, row_data):
            entry = per[key]
            p = entry["p"] * 100
            lo = entry["lo"] * 100
            hi = entry["hi"] * 100
            color = _color_for(kind)
            # Two visual tiers: primary Ours (heaviest) and LLM baselines
            # (standard). The previous held-out sub-row is gone now that
            # held-out IS the primary measurement for Ours.
            if kind == "ours":
                marker_size = 5
                lw = 1.6
                zorder = 5
            else:
                marker_size = 3.5
                lw = 1.2
                zorder = 3
            ax.errorbar(
                p, y,
                xerr=[[p - lo], [hi - p]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=lw,
                capsize=2.5,
                markersize=marker_size,
                zorder=zorder,
            )

        # Background highlight band for the primary Ours row.
        for y, (label, kind, _) in zip(y_positions, row_data):
            if kind == "ours":
                ax.axhspan(y - 0.45, y + 0.45, color=COLOR_OURS_BAND, zorder=0)

        # Horizontal divider between with-tools group and no-tools group.
        # Detected as the boundary where 'kind' transitions to 'notools'.
        for idx, (label, kind) in enumerate(rows):
            if kind == "notools":
                # idx is 0-based from top; y_position for this row is n-1-idx.
                # Divider goes between this row and the previous one.
                divider_y = (n - 1 - idx) + 0.5
                ax.axhline(divider_y, color=COLOR_DIVIDER, linewidth=0.6,
                           linestyle="-", zorder=1)
                break

        # 25% random reference for MCQ. Color and weight chosen so it reads
        # as a separate object from the gray gridlines.
        ax.axvline(25, color=COLOR_RANDOM, linestyle="--", linewidth=1.2,
                   alpha=0.85, zorder=2)
        ax.set_title(panel_title)
        ax.set_xlabel("Accuracy (%)")
        ax.set_xlim(0, 100)
        # Major ticks every 20pp (was 25pp): finer reference for reading off
        # CI endpoints, since the actual Wilson half-widths are ~3-7pp.
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        # Minor ticks every 5pp give a sub-major grid without label clutter.
        ax.set_xticks(range(5, 100, 5), minor=True)
        ax.tick_params(axis="x", which="major", direction="in", length=3)
        ax.tick_params(axis="x", which="minor", direction="in", length=1.5)
        ax.tick_params(axis="y", which="both", left=False)
        ax.grid(axis="x", which="minor", linestyle=":", alpha=0.12)
        ax.set_axisbelow(True)

    # Shared y-tick labels on the leftmost panel only. Bold the primary Ours
    # row so the headline finding lands the eye immediately.
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([label for label, _ in rows])
    for tick_label, (_, kind) in zip(axes[0].get_yticklabels(), rows):
        if kind == "ours":
            tick_label.set_color(COLOR_OURS)
            tick_label.set_fontweight("bold")

    # Legend (one entry per kind), placed under the panels.
    handles = [
        plt.Line2D([0], [0], marker="o", color=COLOR_OURS, lw=1.6,
                   label="Grounded-Physics LM (Ours)", markersize=5),
        plt.Line2D([0], [0], marker="o", color=COLOR_LLM, lw=1.2,
                   label="LLM baselines", markersize=3.5),
        plt.Line2D([0], [0], marker="o", color=COLOR_NOTOOLS, lw=1.2,
                   label=r"LLM baselines (no tools, $\dagger$)", markersize=3.5),
        plt.Line2D([0], [0], color=COLOR_RANDOM, lw=1.2, linestyle="--",
                   alpha=0.85, label="MCQ random baseline (25%)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    article_dir = Path(__file__).resolve().parents[2] / "article"
    # 3--6 object plot (matches tab:ci_baselines): Grounded-Physics LM
    # row sources from the held-out partition (FILES["Grounded-Physics LM
    # (held-out)"] == HELDOUT_SENTINEL); LLM rows from 1K-pool.
    out_pdf = article_dir / "ci_forest.pdf"
    make_figure(out_pdf, rows=ROWS, files=FILES, key_map=KEY_MAP)
    print(f"Saved {out_pdf}")
    # 15-object 1K pool stress-test (matches tab:main_results_15obj).
    # The figure height (2.30 inches, vs 2.60 for ci_forest.pdf) is tuned
    # empirically so both PDFs end up at the same aspect ratio after
    # ``bbox_inches="tight"`` trimming. Uses KEY_MAP_15OBJ because the
    # 15-obj pool is a separate stress test with no held-out variant; the
    # Grounded-Physics LM row sources from the 15-obj 1K-pool file there.
    out_pdf_15 = article_dir / "ci_forest_15obj.pdf"
    make_figure(out_pdf_15, rows=ROWS_15OBJ, files=FILES_15OBJ,
                key_map=KEY_MAP_15OBJ, figsize=(7.0, 2.30))
    print(f"Saved {out_pdf_15}")
