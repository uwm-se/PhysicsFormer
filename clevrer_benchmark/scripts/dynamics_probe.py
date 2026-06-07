"""In-distribution statics-vs-dynamics probe for the Phase 3 physics prefix.

The ComPhy OOD study found the physics prefix is loud but non-discriminative,
and that fixing the *dynamics* channels (velocity/mass/...) barely changed its
effect -- suggesting the encoder leans on statics (position/color/shape) rather
than transferable dynamics. This probe tests that *in-distribution*, on CLEVRER.

It reuses the exact contrastive delta instrument: for each CLEVRER descriptive
question with a closed answer set (exist/shape/material/color/count) it scores
the candidate answers with the physics prefix ON (``real``) and OFF (``zero``),
logs both, and tags the question as STATICS- or DYNAMICS-dependent by its text.
The output is the same schema ``scripts/analyze_alpha_sweep.py`` consumes, so:

    python clevrer_benchmark/scripts/dynamics_probe.py --n 3000 \\
        --output clevrer_benchmark/results/dynamics_probe.details.jsonl
    python comphy_benchmark/scripts/analyze_alpha_sweep.py \\
        --details clevrer_benchmark/results/dynamics_probe.details.jsonl --by_coarse

In the analyzer, ``coarse_type`` is ``statics`` / ``dynamics``. The question is
whether physics_blind (zeroed prefix) loses ground to the physics rule *more on
dynamics than on statics*. If physics is load-bearing for dynamics, the prefix
should help dynamics questions specifically; if the gap is flat, the prefix is
statics-leaning even in-distribution.

CAVEAT: the only CLEVRER data with states co-located is the TRAINING h5, so
absolute accuracy is inflated by memorization. The physics-on-vs-off
*differential by group* is the interpretable signal, not the absolute level.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from clevrer_benchmark.run_adapter_evaluation import (  # noqa: E402
    load_adapter_model,
    _contrastive_score_choices,
)

DEFAULT_H5 = (r"C:\Users\jpoko\source\repos\homework\CascadeProjects\windsurf-project"
              r"\compsac_2026_code\data\clevrer_training_expanded.h5")
DEFAULT_CKPT = (r"C:\Users\jpoko\source\repos\homework\CascadeProjects\windsurf-project"
                r"\compsac_2026_code\checkpoints\adapter_phase3.pt")

COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
SHAPES = ["cube", "sphere", "cylinder"]
MATERIALS = ["metal", "rubber"]
EXISTS = ["yes", "no"]
COUNTS = [str(i) for i in range(0, 11)]

# Candidate set per descriptive subtype. Subtypes not listed are skipped
# (no clean closed answer set, e.g. causal_chain / counterfactual phrases).
SUBTYPE_CHOICES = {
    "descriptive_color": COLORS,
    "descriptive_shape": SHAPES,
    "descriptive_material": MATERIALS,
    "descriptive_exist": EXISTS,
    "descriptive_count": COUNTS,
}

# A question is dynamics-dependent if it references genuine motion / collision /
# entrance-exit events. Deliberately EXCLUDES ubiquitous temporal anchors
# ("when", "ends", "begins") -- nearly every CLEVRER descriptive question has
# one, so they don't separate statics from dynamics. Motion-state, collision,
# and entrance/exit verbs are what actually require trajectory reasoning.
DYN_KW = {"moving", "move", "moves", "stationary", "enter", "enters", "entering",
          "collide", "collides", "collision", "collisions", "responsible",
          "cause", "causes", "fall", "falls", "exit", "exits", "hit", "hits",
          "rotate", "rotating", "bounce", "bounces", "stop", "stops"}


def _dec(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)


def _is_dynamics(q: str) -> bool:
    toks = set(q.lower().replace("?", " ").replace(",", " ").split())
    return len(toks & DYN_KW) > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--adapter_checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=3000, help="Questions to sample.")
    ap.add_argument("--single_frame", type=int, default=-1,
                    help="-1 = full trajectory (training format); >=0 picks one frame.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print("Loading adapter...")
    adapter = load_adapter_model(args.adapter_checkpoint, "", device=args.device)
    adapter.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = skipped = 0
    grp = {"statics": 0, "dynamics": 0}
    with h5py.File(args.h5, "r") as f, open(out_path, "w", encoding="utf-8") as out:
        N = min(args.n, f["states"].shape[0])
        for i in range(N):
            subtype = _dec(f["question_types"][i])
            choices = SUBTYPE_CHOICES.get(subtype)
            if not choices:
                skipped += 1
                continue
            q = _dec(f["questions"][i])
            gold = _dec(f["answers"][i]).strip().lower()
            if gold not in [c.lower() for c in choices]:
                skipped += 1
                continue

            states = f["states"][i]   # [T, N, 35]
            masks = f["masks"][i]     # [T, N]
            if args.single_frame is not None and args.single_frame >= 0:
                fi = min(args.single_frame, states.shape[0] - 1)
                states = states[fi:fi + 1]
            st = torch.from_numpy(np.asarray(states)).float().unsqueeze(0).to(args.device)
            mask2d = (np.asarray(masks).max(axis=0) if masks.ndim == 2 else np.asarray(masks))
            mk = torch.from_numpy(mask2d).float().unsqueeze(0).to(args.device)

            try:
                _, real_s, zero_s = _contrastive_score_choices(
                    adapter, st, mk, q, list(choices),
                    alpha=1.0, device=args.device, return_components=True)
            except Exception as e:
                print(f"[skip] q{i}: {e}")
                skipped += 1
                continue

            coarse = "dynamics" if _is_dynamics(q) else "statics"
            grp[coarse] += 1
            rec = {
                "question_id": f"h5_{i}",
                "question_text": q,
                "comphy_type": subtype,
                "coarse_type": coarse,
                "is_mcq": True,
                "ground_truth": gold,
                "choices": [{"choice": c, "answer": ("correct" if c.lower() == gold else "wrong")}
                            for c in choices],
                "real_scores": [float(x) for x in real_s[0].tolist()],
                "zero_scores": [float(x) for x in zero_s[0].tolist()],
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 200 == 0:
                print(f"  kept {kept} (statics={grp['statics']}, dynamics={grp['dynamics']})")

    print(f"\nDone. kept={kept} skipped={skipped}  "
          f"statics={grp['statics']} dynamics={grp['dynamics']}")
    print(f"Details -> {out_path}")
    print("Analyze with: python comphy_benchmark/scripts/analyze_alpha_sweep.py "
          f"--details {out_path} --by_coarse")


if __name__ == "__main__":
    main()
