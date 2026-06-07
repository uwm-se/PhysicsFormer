"""Held-out statics-vs-dynamics ablation: what kind of grounding does the
Phase 3 physics prefix carry?

Runs on the TRAINING-DISJOINT held-out partition (causal questions, last 10% of
the h5 in dataset order -- the same split the training notebook used), so the
result speaks to *generalization*, not memorization. For each held-out MC
question it scores the choices (contrastive ``real`` likelihood) under four
input conditions and reports argmax accuracy by CLEVRER type:

  full        : full trajectory, all 35 channels (dynamics available)
  single_frame: one frame (#64) -> encoder gets NO derivable velocity (statics)
  zero_static : full trajectory, object-identity channels zeroed
                (color/shape/mass/radius/friction/restitution/quat); motion kept
  zero_physics: all channels zeroed (the repo's -62pp ablation)

Key contrasts:
  full vs single_frame  -> does multi-frame DYNAMICS add anything over a static
                           snapshot? (repo headline uses single_frame, so if
                           full ~= single, the load-bearing grounding is static)
  single_frame vs zero  -> the static snapshot's contribution
  zero_static vs full   -> does removing object identity collapse it?

Usage:
  python clevrer_benchmark/scripts/heldout_channel_ablation.py --n 2000 \\
      --output clevrer_benchmark/results/heldout_channel_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from clevrer_benchmark.run_adapter_evaluation import (  # noqa: E402
    load_adapter_model, _contrastive_score_choices,
)
from clevrer_benchmark.scripts.compute_paper_stats import wilson_ci  # noqa: E402

# --- reproducibility: resolve data/checkpoint paths from env vars (see REPRODUCTION.md) ---
import os as _os_repro
from pathlib import Path as _Path_repro
_REPRO_ROOT = _Path_repro(__file__).resolve().parents[2]

DEFAULT_H5 = _os_repro.environ.get("CLEVRER_H5", str(_REPRO_ROOT / "data" / "clevrer_training_expanded.h5"))
DEFAULT_CKPT = _os_repro.environ.get("ADAPTER_CKPT", str(_REPRO_ROOT / "checkpoints" / "adapter_phase3.pt"))

CAUSAL = {"explanatory", "predictive", "counterfactual"}
# Channel groups. Velocity = linear + angular (the dynamics channels).
POS_CHANNELS = [0, 1, 2]
VEL_CHANNELS = [3, 4, 5, 10, 11, 12]
# Object-identity / static-attribute channels (quat, mass, radius, color, shape,
# is_static, friction, inside, inertia, bbox, restitution).
STATIC_CHANNELS = [6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                   23, 24, 25, 26, 27, 34]


def _dec(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)


def _argmax(xs):
    bi, bv = 0, xs[0]
    for i, v in enumerate(xs):
        if v > bv:
            bi, bv = i, v
    return bi


def _heldout_indices(qtypes_meta):
    """Replicate the notebook split: filter to causal types, last 10% by order."""
    causal = [i for i, ct in enumerate(qtypes_meta) if ct in CAUSAL]
    train_size = int(0.9 * len(causal))
    return causal[train_size:]


def _score(adapter, states_np, mask_np, q, choices, device, conds):
    """Return {cond: real_scores list} for each input condition."""
    out = {}
    mask2d = mask_np.max(axis=0) if mask_np.ndim == 2 else mask_np
    mk = torch.from_numpy(mask2d).float().unsqueeze(0).to(device)
    for cond, builder in conds.items():
        st_np = builder(states_np)
        st = torch.from_numpy(np.ascontiguousarray(st_np)).float().unsqueeze(0).to(device)
        _, real_s, _zero = _contrastive_score_choices(
            adapter, st, mk, q, choices, alpha=1.0, device=device,
            return_components=True)
        out[cond] = [float(x) for x in real_s[0].tolist()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--adapter_checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=2000, help="Cap held-out questions.")
    ap.add_argument("--frame", type=int, default=64)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    print("Loading adapter...")
    adapter = load_adapter_model(args.adapter_checkpoint, "", device=args.device)
    adapter.eval()

    # All channel-zeroing conditions are SINGLE-FRAME: on a multi-frame input
    # the encoder re-derives velocity from the position trajectory, so zeroing
    # the velocity channel would not actually remove velocity. Single frame =>
    # the velocity channel is the only velocity signal, so zeroing it is clean.
    def b_full(s):                       # full trajectory, all channels (upper ref)
        return s
    def _single(s):
        fi = min(args.frame, s.shape[0] - 1)
        return s[fi:fi + 1].copy()
    def b_single(s):                     # one frame, all channels (instantaneous state)
        return _single(s)
    def b_no_vel(s):                     # pos + identity, NO velocity  <-- key test
        x = _single(s); x[:, :, VEL_CHANNELS] = 0.0; return x
    def b_no_pos(s):                     # velocity + identity, no position
        x = _single(s); x[:, :, POS_CHANNELS] = 0.0; return x
    def b_identity_only(s):              # identity only (no pos, no vel)
        x = _single(s); x[:, :, POS_CHANNELS + VEL_CHANNELS] = 0.0; return x
    def b_kinematics_only(s):            # pos + vel, NO identity
        x = _single(s); x[:, :, STATIC_CHANNELS] = 0.0; return x
    def b_zero(s):
        return np.zeros_like(s)
    CONDS = {"full": b_full, "single_all": b_single, "no_velocity": b_no_vel,
             "no_position": b_no_pos, "identity_only": b_identity_only,
             "kinematics_only": b_kinematics_only, "zero_physics": b_zero}

    # tallies[cond][ctype] = [correct, total]
    tallies = {c: defaultdict(lambda: [0, 0]) for c in CONDS}

    with h5py.File(args.h5, "r") as f:
        meta_all = f["metadata"][:]
        ctype = []
        for m in meta_all:
            try:
                ctype.append(json.loads(_dec(m)).get("clevrer_type", "?"))
            except Exception:
                ctype.append("?")
        held = _heldout_indices(ctype)
        print(f"held-out causal questions: {len(held)} (capping at {args.n})")
        held = held[:args.n]

        done = 0
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
            gold_idx = next((k for k, c in enumerate(choice_texts)
                             if c.strip().lower() == gold), None)
            if gold_idx is None:
                continue
            q = _dec(f["questions"][i])
            ct = ctype[i]
            states = np.asarray(f["states"][i])   # [T,N,35]
            masks = np.asarray(f["masks"][i])
            try:
                scored = _score(adapter, states, masks, q, choice_texts, args.device, CONDS)
            except Exception as e:
                print(f"[skip] {i}: {e}")
                continue
            for cond, real in scored.items():
                pred = _argmax(real)
                ok = int(pred == gold_idx)
                tallies[cond][ct][0] += ok
                tallies[cond][ct][1] += 1
                tallies[cond]["ALL"][0] += ok
                tallies[cond]["ALL"][1] += 1
            done += 1
            if done % 200 == 0:
                a = tallies["full"]["ALL"]; z = tallies["zero_physics"]["ALL"]
                print(f"  {done} | full {100*a[0]/max(1,a[1]):.1f}%  "
                      f"zero {100*z[0]/max(1,z[1]):.1f}%")

    print(f"\nHeld-out channel/frame ablation  (n={tallies['full']['ALL'][1]} questions)")
    print("=" * 78)
    order = ["full", "single_all", "no_velocity", "no_position",
             "identity_only", "kinematics_only", "zero_physics"]
    types = ["ALL", "explanatory", "predictive", "counterfactual"]
    hdr = f"{'condition':<14}" + "".join(f"{t[:13]:>16}" for t in types)
    print(hdr)
    print("-" * 78)
    for cond in order:
        row = f"{cond:<14}"
        for t in types:
            c, n = tallies[cond][t]
            if n == 0:
                row += f"{'-':>16}"
            else:
                p, lo, hi = wilson_ci(c, n)
                row += f"{p*100:>7.1f}% ({n:>4}) "
        print(row)
    print("-" * 78)
    def acc(name):
        t = tallies[name]["ALL"]; return 100 * t[0] / max(1, t[1])
    print(f"total grounding   full - zero_physics:        {acc('full')-acc('zero_physics'):+.1f} pp")
    print(f"trajectory/accel  full - single_all:          {acc('full')-acc('single_all'):+.1f} pp")
    print(f"VELOCITY          single_all - no_velocity:    {acc('single_all')-acc('no_velocity'):+.1f} pp   <-- dynamics test")
    print(f"position          single_all - no_position:    {acc('single_all')-acc('no_position'):+.1f} pp")
    print(f"identity-only     identity_only - zero:        {acc('identity_only')-acc('zero_physics'):+.1f} pp")
    print(f"kinematics-only   kinematics_only - zero:      {acc('kinematics_only')-acc('zero_physics'):+.1f} pp")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as out:
            json.dump({c: {t: tallies[c][t] for t in tallies[c]} for c in CONDS},
                      out, indent=2)
        print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
