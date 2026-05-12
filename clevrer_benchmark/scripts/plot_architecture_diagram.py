"""Generate the Grounded-Physics LM architecture diagram for fig:architecture.

Produces a vector PDF showing the full inference + training pipeline,
including components the prior PNG omitted: LoRA on DistilGPT-2 attention
(Phase 2/3 trainable) and the auxiliary numerical / descriptive / MCQ
heads (training-only). Layout is two parallel input pipelines (physics +
question) converging at the prefix concatenation, then a single column
through DistilGPT-2 to the answer output.

Outputs ``compsac_2026_code/article/architecture_diagram_vector.pdf``
(parked vector replacement for the legacy raster
``architecture_diagram.png`` referenced from main.tex). Distinct filename
intentional: the legacy PNG must not be clobbered if this script is
rerun.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt


# IEEE-conference typography; slightly larger fonts than the forest plot
# because this figure carries lots of small inline labels.
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette -- matches the warm pink / cool blue split of the rest of
# the article. Trainable phases distinguished by hue:
#   Phase 1 trainable (adapter MLP + aux heads): pink
#   Phase 2 trainable (+ LoRA on attention):     purple
#   Phase 3 trainable (+ full DistilGPT-2):      light blue
#   Always frozen / data:                        light gray
COLOR_INPUT = "#E8E8E8"          # input/data boxes (gray)
COLOR_INPUT_EDGE = "#5A5A5A"
COLOR_PHYSICS_FROZEN = "#FBE0DA" # PhysicsFormer (frozen during adapter training)
COLOR_PHYSICS_FROZEN_EDGE = "#A35540"
COLOR_PHASE1 = "#F4D5E5"         # adapter + aux heads (Phase 1 trainable)
COLOR_PHASE1_EDGE = "#A03A6F"
COLOR_PHASE2 = "#E0CDF7"         # LoRA (Phase 2 trainable)
COLOR_PHASE2_EDGE = "#6B4FA0"
COLOR_PHASE3 = "#D5E5F4"         # DistilGPT-2 backbone (Phase 3 fine-tuned)
COLOR_PHASE3_EDGE = "#4A6F95"
COLOR_OUTPUT = "#888888"
COLOR_AUX = "#FFF5E0"             # aux heads pale background
COLOR_AUX_EDGE = "#B89544"
COLOR_FROZEN_BORDER = "#A0A0A0"   # dashed when component is frozen


def draw_box(ax, x, y, w, h, text, *, fc, ec, lw=1.0, fontsize=8,
             bold=False, italic=False, dashed=False, rounded=True):
    """Draw a rectangular component box with centered text."""
    style = "round,pad=0.05,rounding_size=0.10" if rounded else "square,pad=0.05"
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=style,
        facecolor=fc, edgecolor=ec,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    style_text = "italic" if italic else "normal"
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=weight,
            fontstyle=style_text,
            color=ec, zorder=4)


def draw_arrow(ax, x0, y0, x1, y1, *, color="#444444", lw=1.4,
               style="->", dashed=False, label=None, label_offset=(0.0, 0.0)):
    """Draw a clean arrow between two points."""
    arrow = patches.FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style,
        mutation_scale=14,
        color=color,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                fontsize=6.5, ha="center", va="center",
                color=color, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.9, pad=0.5))


def draw_phase_legend(ax, x, y):
    """Small inline legend showing what each color/style means."""
    items = [
        (COLOR_PHYSICS_FROZEN, COLOR_PHYSICS_FROZEN_EDGE,
         "PhysicsFormer (frozen)"),
        (COLOR_PHASE1, COLOR_PHASE1_EDGE,
         "Phase 1 trainable: adapter + aux heads"),
        (COLOR_PHASE2, COLOR_PHASE2_EDGE,
         "Phase 2 trainable: + LoRA"),
        (COLOR_PHASE3, COLOR_PHASE3_EDGE,
         "Phase 3 trainable: + full DistilGPT-2"),
    ]
    box_w, box_h = 0.45, 0.35
    line_h = 0.55
    # Frame
    frame = patches.FancyBboxPatch(
        (x - 0.15, y - line_h * len(items) - 0.10),
        4.65, line_h * len(items) + 0.30,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        facecolor="white", edgecolor="#888888",
        linewidth=0.8, zorder=5,
    )
    ax.add_patch(frame)
    for i, (fc, ec, label) in enumerate(items):
        cy = y - i * line_h
        sw = patches.FancyBboxPatch(
            (x, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=6,
        )
        ax.add_patch(sw)
        ax.text(x + box_w + 0.18, cy, label,
                ha="left", va="center", fontsize=6.5,
                color="#333333", zorder=6)


def make_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.5))
    ax.set_xlim(-0.5, 13.0)
    ax.set_ylim(-1.5, 12.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Physics input pipeline (left column) ----------------------
    # Object states
    draw_box(ax, 0.5, 11.0, 4.0, 0.7,
             r"Object States $[B, T, N, 35]$",
             fc=COLOR_INPUT, ec=COLOR_INPUT_EDGE, bold=True)

    # State encoder (frozen part of PhysicsFormer)
    draw_box(ax, 0.5, 9.6, 4.0, 0.8,
             "StateEncoder\n"
             r"$70$D augmented $\to 768$D",
             fc=COLOR_PHYSICS_FROZEN, ec=COLOR_PHYSICS_FROZEN_EDGE,
             fontsize=7.5)

    # Transformer stack
    draw_box(ax, 0.5, 8.0, 4.0, 1.0,
             "PhysicsFormer Transformer\n"
             r"8 layers, physics-biased attention",
             fc=COLOR_PHYSICS_FROZEN, ec=COLOR_PHYSICS_FROZEN_EDGE,
             fontsize=7.5)

    # Pool
    draw_box(ax, 1.0, 6.7, 3.0, 0.6,
             "Pool over objects (mean)",
             fc=COLOR_PHYSICS_FROZEN, ec=COLOR_PHYSICS_FROZEN_EDGE,
             fontsize=7.5)

    # Adapter (Phase 1 trainable)
    draw_box(ax, 0.5, 5.2, 4.0, 0.9,
             "Adapter MLP\n"
             r"$768 \to 64 \times 768$ (prefix tokens)",
             fc=COLOR_PHASE1, ec=COLOR_PHASE1_EDGE, fontsize=7.5)

    # --- Question input pipeline (right column) --------------------
    draw_box(ax, 5.5, 11.0, 3.5, 0.7,
             "Question Text",
             fc=COLOR_INPUT, ec=COLOR_INPUT_EDGE, bold=True)

    draw_box(ax, 5.5, 9.6, 3.5, 0.8,
             "DistilGPT-2 Tokenizer",
             fc=COLOR_INPUT, ec=COLOR_INPUT_EDGE, fontsize=7.5)

    draw_box(ax, 5.5, 8.0, 3.5, 0.8,
             r"Embeddings $[L, 768]$",
             fc=COLOR_INPUT, ec=COLOR_INPUT_EDGE, fontsize=7.5)

    # --- Concatenation ---------------------------------------------
    draw_box(ax, 2.5, 3.7, 5.0, 0.9,
             r"Concatenate: $[\,\mathrm{Prefix}\,(64) \;|\; \mathrm{Question}\,(L)\,] = [64 + L,\ 768]$",
             fc=COLOR_PHASE1, ec=COLOR_PHASE1_EDGE, fontsize=7.5)

    # --- DistilGPT-2 stack -----------------------------------------
    # Phase 3 main backbone
    draw_box(ax, 2.5, 2.0, 5.0, 1.1,
             "DistilGPT-2 backbone\n"
             "6 transformer layers, 768D, 82M params",
             fc=COLOR_PHASE3, ec=COLOR_PHASE3_EDGE, fontsize=7.5)
    # LoRA badge attached to the right side of the DistilGPT-2 box
    draw_box(ax, 7.7, 2.1, 2.4, 0.9,
             "LoRA on attention\n"
             r"rank 8, $\alpha=16$",
             fc=COLOR_PHASE2, ec=COLOR_PHASE2_EDGE, fontsize=7.0)
    # Connector showing LoRA modifies the backbone (small bidirectional arrow)
    ax.annotate("", xy=(7.7, 2.55), xytext=(7.5, 2.55),
                arrowprops=dict(arrowstyle="<->",
                                color=COLOR_PHASE2_EDGE, lw=1.0),
                zorder=4)

    # LM head + answer
    draw_box(ax, 3.5, 0.6, 3.0, 0.7,
             r"LM head $\to$ vocab",
             fc=COLOR_PHASE3, ec=COLOR_PHASE3_EDGE, fontsize=7.5)
    draw_box(ax, 4.0, -0.7, 2.0, 0.7,
             "Answer",
             fc=COLOR_INPUT, ec=COLOR_INPUT_EDGE, bold=True)

    # --- Auxiliary heads (training-only branch) --------------------
    # Group container - dashed border indicates training-only
    aux_x, aux_y, aux_w, aux_h = 9.4, 4.6, 3.4, 2.5
    aux_frame = patches.FancyBboxPatch(
        (aux_x, aux_y), aux_w, aux_h,
        boxstyle="round,pad=0.10,rounding_size=0.12",
        facecolor=COLOR_AUX, edgecolor=COLOR_AUX_EDGE,
        linewidth=1.0, linestyle="--", zorder=2,
    )
    ax.add_patch(aux_frame)
    ax.text(aux_x + aux_w / 2, aux_y + aux_h - 0.20,
            "Auxiliary heads (training only)",
            ha="center", va="top", fontsize=7.5, fontweight="bold",
            fontstyle="italic", color=COLOR_AUX_EDGE, zorder=4)
    # Three sub-heads inside the aux container
    draw_box(ax, aux_x + 0.20, aux_y + 1.45, aux_w - 0.40, 0.55,
             r"Numerical head ($6$-D regression)",
             fc=COLOR_PHASE1, ec=COLOR_PHASE1_EDGE, fontsize=6.8)
    draw_box(ax, aux_x + 0.20, aux_y + 0.75, aux_w - 0.40, 0.55,
             "Descriptive heads (count/exist/\ncolor/shape/material)",
             fc=COLOR_PHASE1, ec=COLOR_PHASE1_EDGE, fontsize=6.8)
    draw_box(ax, aux_x + 0.20, aux_y + 0.05, aux_w - 0.40, 0.55,
             "MCQ scoring head",
             fc=COLOR_PHASE1, ec=COLOR_PHASE1_EDGE, fontsize=6.8)

    # --- Arrows: physics pipeline ----------------------------------
    draw_arrow(ax, 2.5, 11.0, 2.5, 10.4)   # ObjStates -> StateEncoder
    draw_arrow(ax, 2.5, 9.6, 2.5, 9.0)     # StateEncoder -> Transformer
    draw_arrow(ax, 2.5, 8.0, 2.5, 7.3)     # Transformer -> Pool
    draw_arrow(ax, 2.5, 6.7, 2.5, 6.1)     # Pool -> Adapter
    draw_arrow(ax, 2.5, 5.2, 4.0, 4.6)     # Adapter -> Concat (left input)

    # --- Arrows: question pipeline ---------------------------------
    draw_arrow(ax, 7.25, 11.0, 7.25, 10.4)   # QuestionText -> Tokenizer
    draw_arrow(ax, 7.25, 9.6, 7.25, 8.8)     # Tokenizer -> Embeddings
    draw_arrow(ax, 7.25, 8.0, 6.0, 4.6)      # Embeddings -> Concat (right input)

    # --- Arrows: post-concat ---------------------------------------
    draw_arrow(ax, 5.0, 3.7, 5.0, 3.1)       # Concat -> DistilGPT-2
    draw_arrow(ax, 5.0, 2.0, 5.0, 1.3)       # DistilGPT-2 -> LM head
    draw_arrow(ax, 5.0, 0.6, 5.0, 0.0)       # LM head -> Answer

    # --- Arrows: aux heads (training-only, dashed) -----------------
    # Route from the Adapter MLP right edge up and over into the aux
    # container's top-left region so the label sits in whitespace between
    # the Adapter box and the aux container title (not on top of the
    # "Numerical head" sub-box).
    draw_arrow(
        ax, 4.5, 5.90, 9.4, 6.95,
        color=COLOR_AUX_EDGE, lw=1.2, dashed=True,
        label="train",
        label_offset=(0.0, 0.25),
    )

    # --- Phase legend (top-right corner, above aux heads) ----------
    draw_phase_legend(ax, 9.4, 11.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    article_dir = Path(__file__).resolve().parents[2] / "article"
    out_pdf = article_dir / "architecture_diagram_vector.pdf"
    make_figure(out_pdf)
    print(f"Saved {out_pdf}")
