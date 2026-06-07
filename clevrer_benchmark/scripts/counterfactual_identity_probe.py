"""Counterfactual probe: do predictions track physical dynamics or identity labels?

On the training-disjoint CLEVRER held-out partition, for each causal MC question
we recompute the model's predicted choice under two scene edits:

  velocity-reversal : multiply every velocity (linear + angular) by -1, keeping
                      positions and identities fixed. Physically this inverts the
                      dynamics -- collisions that would happen now don't. A model
                      that reasons over dynamics must change its answer; a model
                      that reads object identity should not.
  identity-permute  : roll the object-identity channels (color/shape/mass/size/
                      friction) across objects, keeping positions and velocities
                      fixed. This moves the referents ("the red object"), so any
                      model that uses the scene at all should change its answer.
                      Positive control that the model is not simply inert.

We report the prediction FLIP RATE (argmax choice changes vs. baseline) under
each edit. flip(velocity) ~ 0 with flip(identity) large == predictions track
identity, not physical state.

Single-frame (frame 64) to match the eval protocol; velocity reversal on a
single frame is clean (no trajectory for the encoder to re-derive motion from).

    python clevrer_benchmark/scripts/counterfactual_identity_probe.py --n 1500
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
    load_adapter_model, _contrastive_score_choices,
)
from clevrer_benchmark.scripts.compute_paper_stats import wilson_ci  # noqa: E402
from clevrer_benchmark.scripts.heldout_channel_ablation import (  # noqa: E402
    POS_CHANNELS, VEL_CHANNELS, STATIC_CHANNELS, CAUSAL, _dec, _argmax,
    _heldout_indices,
)

DEFAULT_H5 = (r"C:\Users\jpoko\source\repos\homework\CascadeProjects\windsurf-project"
              r"\compsac_2026_code\data\clevrer_training_expanded.h5")
DEFAULT_CKPT = (r"C:\Users\jpoko\source\repos\homework\CascadeProjects\windsurf-project"
                r"\compsac_2026_code\checkpoints\adapter_phase3.pt")


def _pred(adapter, st_np, mk, q, choices, device):
    st = torch.from_numpy(np.ascontiguousarray(st_np)).float().unsqueeze(0).to(device)
    _, real_s, _z = _contrastive_score_choices(
        adapter, st, mk, q, choices, alpha=1.0, device=device, return_components=True)
    return _argmax([float(x) for x in real_s[0].tolist()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--adapter_checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--frame", type=int, default=64)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    print("Loading adapter...")
    adapter = load_adapter_model(args.adapter_checkpoint, "", device=args.device)
    adapter.eval()

    flips = {"velocity": 0, "identity": 0}
    total = 0
    with h5py.File(args.h5, "r") as f:
        meta_all = f["metadata"][:]
        ctype = []
        for m in meta_all:
            try:
                ctype.append(json.loads(_dec(m)).get("clevrer_type", "?"))
            except Exception:
                ctype.append("?")
        held = _heldout_indices(ctype)[:args.n]
        print(f"held-out causal questions: {len(held)}")

        for i in held:
            try:
                meta = json.loads(_dec(meta_all[i]))
            except Exception:
                continue
            choices_meta = meta.get("choices")
            if not choices_meta:
                continue
            choice_texts = [c.get("choice", "") for c in choices_meta]
            gold = _dec(f["answers"][i]).strip().lower()
            if not any(c.strip().lower() == gold for c in choice_texts):
                continue
            q = _dec(f["questions"][i])
            states = np.asarray(f["states"][i])
            masks = np.asarray(f["masks"][i])
            fi = min(args.frame, states.shape[0] - 1)
            base = states[fi:fi + 1].copy()                       # [1, N, 35]
            n_obj = base.shape[1]
            if n_obj < 2:
                continue
            mask2d = masks.max(axis=0) if masks.ndim == 2 else masks
            mk = torch.from_numpy(mask2d).float().unsqueeze(0).to(args.device)

            # velocity-reversal (dynamics edit, identity + position fixed)
            vel = base.copy(); vel[:, :, VEL_CHANNELS] *= -1.0
            # identity-permute (identity edit, position + velocity fixed)
            idp = base.copy(); idp[:, :, STATIC_CHANNELS] = np.roll(
                base[:, :, STATIC_CHANNELS], shift=1, axis=1)

            try:
                p0 = _pred(adapter, base, mk, q, choice_texts, args.device)
                pv = _pred(adapter, vel, mk, q, choice_texts, args.device)
                pi = _pred(adapter, idp, mk, q, choice_texts, args.device)
            except Exception as e:
                print(f"[skip] {i}: {e}")
                continue
            flips["velocity"] += int(pv != p0)
            flips["identity"] += int(pi != p0)
            total += 1
            if total % 200 == 0:
                print(f"  {total} | vel-flip {100*flips['velocity']/total:.1f}%  "
                      f"id-flip {100*flips['identity']/total:.1f}%")

    print(f"\nCounterfactual sensitivity on held-out causal MC (n={total})")
    print("=" * 60)
    for k in ["velocity", "identity"]:
        p, lo, hi = wilson_ci(flips[k], total)
        label = ("velocity-reversal (DYNAMICS edit)" if k == "velocity"
                 else "identity-permute (referent edit)")
        print(f"{label:<36} flip {p*100:5.1f}%  [{lo*100:4.1f}, {hi*100:4.1f}]")
    print("-" * 60)
    vf = flips["velocity"] / max(1, total); idf = flips["identity"] / max(1, total)
    print(f"Predictions are {idf/max(vf,1e-9):.1f}x more sensitive to identity "
          f"than to inverted dynamics.")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n": total, "flips": flips}, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
