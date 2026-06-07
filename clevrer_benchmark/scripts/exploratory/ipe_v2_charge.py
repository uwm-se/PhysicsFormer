"""Charge test for IPEv2: meta-train on CLEVRER + ComPhy (charged upweighted) and
check whether the inverse-square force-prior LEARNS charge -- i.e. whether the
charged-subset rollout improves (the IPE got +27%) and the force-prior scale
switches on (learns a nonzero magnitude).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from clevrer_benchmark.scripts.ipe_v2_transfer import IPEv2, ballistic_mse  # noqa: E402
from physics_llm_adapter.intuitive_physics_engine import rollout_loss  # noqa: E402

# --- reproducibility: resolve data/checkpoint paths from env vars (see REPRODUCTION.md) ---
import os as _os_repro
from pathlib import Path as _Path_repro
_REPRO_ROOT = _Path_repro(__file__).resolve().parents[3]

CLEVRER_H5 = _os_repro.environ.get("CLEVRER_H5", str(_REPRO_ROOT / "data" / "clevrer_training_expanded.h5"))
COMPHY_CACHE = "comphy_benchmark/results/ipe_comphy_cache.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    with h5py.File(CLEVRER_H5, "r") as f:
        Sc = np.asarray(f["states"][:16000]); Mc = np.asarray(f["masks"][:16000, 0, :])
    t0c = Sc.shape[1] // 2
    Sc_t, Mc_t = torch.from_numpy(Sc).float(), torch.from_numpy(Mc).float()

    d = np.load(COMPHY_CACHE)
    S, M, CH = d["states"], d["masks"], d["charged"]
    perm = np.random.default_rng(0).permutation(len(S))
    S, M, CH = S[perm], M[perm], CH[perm]
    t0p, hor = 45, 8
    ntr = int(0.85 * len(S))
    Sp_t = torch.from_numpy(S[:ntr]).float(); Mp_t = torch.from_numpy(M[:ntr, t0p, :]).float()
    ch_tr = CH[:ntr]; pos = np.where(ch_tr == 1)[0]; neg = np.where(ch_tr == 0)[0]
    es = torch.from_numpy(S[ntr:]).float().to(dev); em = torch.from_numpy(M[ntr:, t0p, :]).float().to(dev)
    ech = CH[ntr:]

    model = IPEv2().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(0); rng = np.random.default_rng(1)
    print(f"meta-train IPEv2 on CLEVRER + ComPhy (charged upweighted). "
          f"ComPhy train charged={len(pos)} neutral={len(neg)}")
    model.train()
    for step in range(1, args.steps + 1):
        if step % 2 == 0:                       # CLEVRER batch
            bi = torch.randint(0, Sc.shape[0], (args.batch,), generator=g)
            s, m, t0 = Sc_t[bi].to(dev), Mc_t[bi].to(dev), t0c
        else:                                   # ComPhy batch, 50% charged
            h = args.batch // 2
            bi = np.concatenate([rng.choice(pos, h), rng.choice(neg, args.batch - h)])
            bi = torch.from_numpy(bi)
            s, m, t0 = Sp_t[bi].to(dev), Mp_t[bi].to(dev), t0p
        pred, _ = model(s, m, t0, 8, 8)
        loss = rollout_loss(pred, s, m, t0, 8)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            print(f"  step {step}  MSE {loss.item():.4f}  force_scale={model.cell.force.scale.item():+.3f}")

    model.eval()
    print(f"\nForce-prior scale (0=off): {model.cell.force.scale.item():+.4f}")
    print(f"{'subset':<10}{'n':>6}{'ballistic':>12}{'IPEv2':>10}{'gain':>8}  (IPE charged was +27%)")
    with torch.no_grad():
        for name, sel in [("neutral", ech == 0), ("charged", ech == 1), ("all", np.ones_like(ech, bool))]:
            idx = np.where(sel)[0]
            s, m = es[idx], em[idx]
            bal = ballistic_mse(s, m, t0p, hor, dev)
            pred, _ = model(s, m, t0p, 8, hor)
            ipe = rollout_loss(pred, s, m, t0p, hor).item()
            print(f"{name:<10}{len(idx):>6}{bal:>12.4f}{ipe:>10.4f}{(1-ipe/bal)*100:>7.0f}%")


if __name__ == "__main__":
    main()
