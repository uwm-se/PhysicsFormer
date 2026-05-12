"""Generate a top-down 2D scene diagram for fig:example_prompt.

Renders the 3-object CLEVRER counterfactual instance used in the article's
worked example: blue rubber cube, brown rubber sphere, yellow rubber
cylinder. Shows current positions, velocity vectors, and the counterfactual
projected trajectory of the cylinder (with the blue cube removed) so a
reader can see immediately why GPT-4o was wrong: the cylinder's trajectory
passes the stationary sphere at $x \\approx 0$ vs sphere at $x = 0.86$,
missing by $>0.8$ m.

Outputs ``compsac_2026_code/article/scene_diagram.pdf`` (vector) plus a PNG
preview alongside.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


# IEEE-conference typography matching the rest of the article figures.
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
})

# Scene constants (frame 64 of the worked example).
# Positions are (x, y); z dropped for the top-down view.
BLUE_CUBE = {
    "pos": np.array([3.380, -1.140]),
    "vel": np.array([-2.960, 0.480]),
    "radius": 0.36,
    "color": "#3B6FB6",       # blue
    "edge": "#1F3F66",
    "label": "blue cube",
}
BROWN_SPHERE = {
    "pos": np.array([0.860, -0.750]),
    "vel": np.array([0.0, 0.0]),
    "radius": 0.30,
    "color": "#8C5A3C",       # brown
    "edge": "#4F3220",
    "label": "brown sphere",
}
YELLOW_CYLINDER = {
    "pos": np.array([0.670, 5.700]),
    "vel": np.array([-0.280, -2.700]),
    "radius": 0.33,
    "color": "#E8C547",       # yellow
    "edge": "#7E6818",
    "label": "yellow cylinder",
}

# Counterfactual closest-approach point of the cylinder to the sphere
# (computed analytically from the velocity vector).
# When y_cyl = y_sphere = -0.75, t = (-0.75 - 5.70) / -2.70 = 2.389 s.
# x_cyl at that time = 0.670 + (-0.280) * 2.389 = 0.001.
CLOSEST_X = 0.001
CLOSEST_Y = BROWN_SPHERE["pos"][1]   # -0.750
MISS_DISTANCE_M = abs(CLOSEST_X - BROWN_SPHERE["pos"][0])  # ~0.86 m


def _draw_object(ax, obj, *, draw_velocity=True):
    """Draw an object as a circle with optional velocity arrow + label."""
    x, y = obj["pos"]
    circ = patches.Circle(
        (x, y), obj["radius"],
        facecolor=obj["color"],
        edgecolor=obj["edge"],
        linewidth=1.0,
        zorder=4,
    )
    ax.add_patch(circ)
    # Object label, offset diagonally up-right of the marker.
    ax.text(
        x + obj["radius"] + 0.20, y + obj["radius"] + 0.15,
        obj["label"], fontsize=7, ha="left", va="bottom",
        color=obj["edge"], zorder=5,
    )
    if draw_velocity and np.linalg.norm(obj["vel"]) > 1e-3:
        ax.annotate(
            "", xytext=(x, y),
            xy=(x + obj["vel"][0] * 0.6, y + obj["vel"][1] * 0.6),
            arrowprops=dict(
                arrowstyle="->",
                color=obj["edge"],
                lw=1.4,
                shrinkA=obj["radius"] * 50,  # approx; visually OK
                shrinkB=0,
            ),
            zorder=5,
        )


def make_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 4.4))

    # Draw the brown sphere and yellow cylinder (kept in the counterfactual).
    _draw_object(ax, BROWN_SPHERE)
    _draw_object(ax, YELLOW_CYLINDER)
    # Draw the blue cube faded with a strikethrough to show it's removed.
    bc = BLUE_CUBE
    bx, by = bc["pos"]
    fade = patches.Circle(
        (bx, by), bc["radius"],
        facecolor=bc["color"], edgecolor=bc["edge"],
        linewidth=1.0, alpha=0.25, zorder=3,
    )
    ax.add_patch(fade)
    cross_size = bc["radius"] * 0.95
    ax.plot([bx - cross_size, bx + cross_size],
            [by - cross_size, by + cross_size],
            color=bc["edge"], linewidth=1.4, alpha=0.7, zorder=4)
    ax.plot([bx - cross_size, bx + cross_size],
            [by + cross_size, by - cross_size],
            color=bc["edge"], linewidth=1.4, alpha=0.7, zorder=4)
    ax.text(
        bx + bc["radius"] + 0.20, by + bc["radius"] + 0.15,
        "blue cube (removed)",
        fontsize=7, ha="left", va="bottom",
        color=bc["edge"], alpha=0.85, zorder=5, style="italic",
    )

    # Projected counterfactual trajectory of the yellow cylinder.
    cyl_pos = YELLOW_CYLINDER["pos"]
    cyl_vel = YELLOW_CYLINDER["vel"]
    # Extend the trajectory until it leaves the visible region at y = -2.5.
    t_end = (-2.5 - cyl_pos[1]) / cyl_vel[1]
    traj_t = np.linspace(0, t_end, 200)
    traj_x = cyl_pos[0] + cyl_vel[0] * traj_t
    traj_y = cyl_pos[1] + cyl_vel[1] * traj_t
    ax.plot(traj_x, traj_y, linestyle="--", color=YELLOW_CYLINDER["edge"],
            linewidth=1.2, alpha=0.7, zorder=2,
            label="cylinder projected trajectory")

    # Mark the closest-approach point and annotate the miss distance.
    ax.plot(CLOSEST_X, CLOSEST_Y, marker="x", color="#A63D40",
            markersize=9, mew=2.2, zorder=5)
    # Horizontal arrow showing the miss distance.
    arrow_y = CLOSEST_Y - 0.65
    ax.annotate(
        "", xy=(BROWN_SPHERE["pos"][0] - BROWN_SPHERE["radius"], arrow_y),
        xytext=(CLOSEST_X, arrow_y),
        arrowprops=dict(arrowstyle="<->,head_length=0.35,head_width=0.18",
                        color="#A63D40", lw=1.4),
        zorder=5,
    )
    ax.text(
        (CLOSEST_X + BROWN_SPHERE["pos"][0]) / 2, arrow_y - 0.30,
        f"miss $\\approx {MISS_DISTANCE_M:.2f}$ m",
        fontsize=7, ha="center", va="top", color="#A63D40", zorder=5,
        fontweight="bold",
    )

    # Cosmetic: axis limits, equal aspect, grid, labels.
    ax.set_xlim(-1.2, 4.6)
    ax.set_ylim(-3.0, 7.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_title("Top-down scene at frame 64 (counterfactual)")

    # Legend (compact; only the trajectory line needs an entry since the
    # objects label themselves inline). Lower-right keeps it clear of the
    # yellow cylinder label at the top and the brown sphere / miss-arrow
    # cluster around y=-1.
    ax.legend(
        loc="lower right",
        frameon=True, framealpha=0.95,
        edgecolor="#888888",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    article_dir = Path(__file__).resolve().parents[2] / "article"
    out_pdf = article_dir / "scene_diagram.pdf"
    make_figure(out_pdf)
    print(f"Saved {out_pdf}")
