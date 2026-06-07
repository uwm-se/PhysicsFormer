"""Module 1 v2: a relative-invariant relational simulator with a charge force-prior,
tested on the generalization go/no-go (CLEVRER -> ComPhy zero-shot).

Two upgrades over the IPE, each targeting a measured failure:
  - RELATIVE-INVARIANT edges (pairwise relative position/velocity; node embedding
    has NO absolute position) -> laws are translation-invariant, so they should
    transfer across distributions with different coordinates. Fix for the IPE's
    -1% CLEVRER->ComPhy-neutral transfer.
  - INVERSE-SQUARE FORCE PRIOR modulated by a per-object charge inferred from
    motion -> an inductive bias for long-range attraction/repulsion. Fix for the
    IPE's failure on charge.

Train on CLEVRER, evaluate rollout MSE vs a constant-velocity baseline on
held-out ComPhy (neutral and charged), ZERO-SHOT. Go = beats ballistic on
neutral (IPE got -1%); charged gain is the harder secondary signal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from physics_llm_adapter.intuitive_physics_engine import (  # noqa: E402
    LatentInference, mlp, POS, VEL, STATIC_IDX, DYN_DIM, rollout_loss)

# --- reproducibility: resolve data/checkpoint paths from env vars (see REPRODUCTION.md) ---
import os as _os_repro
from pathlib import Path as _Path_repro
_REPRO_ROOT = _Path_repro(__file__).resolve().parents[2]

CLEVRER_H5 = _os_repro.environ.get("CLEVRER_H5", str(_REPRO_ROOT / "data" / "clevrer_training_expanded.h5"))
COMPHY_CACHE = "comphy_benchmark/results/ipe_comphy_cache.npz"
DT = 1.0 / 25.0


class RelInteraction(nn.Module):
    """Translation-invariant message passing: edges from RELATIVE pos/vel."""
    def __init__(self, node_dim, hidden):
        super().__init__()
        self.edge = mlp(2 * node_dim + 7, hidden, hidden)   # ni,nj,relpos(3),relvel(3),dist(1)
        self.node = mlp(node_dim + hidden, hidden, node_dim)

    def forward(self, h, pos, vel, mask):
        B, N, d = h.shape
        rp = pos.unsqueeze(2) - pos.unsqueeze(1)            # [B,N,N,3] (i from j)
        rv = vel.unsqueeze(2) - vel.unsqueeze(1)
        dist = rp.norm(dim=-1, keepdim=True)
        ni = h.unsqueeze(2).expand(B, N, N, d)
        nj = h.unsqueeze(1).expand(B, N, N, d)
        e = self.edge(torch.cat([ni, nj, rp, rv, dist], -1))
        m = mask.unsqueeze(1).unsqueeze(-1)
        e = e * m
        agg = e.sum(2) / m.sum(2).clamp(min=1.0)
        return h + self.node(torch.cat([h, agg], -1)) * mask.unsqueeze(-1)


class ForcePrior(nn.Module):
    """Inverse-square pairwise force ~ q_i q_j / r^2 along the line of centers,
    with per-object charge q inferred from the latent. Same sign -> repel."""
    def __init__(self, latent):
        super().__init__()
        self.charge = nn.Linear(latent, 1)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, pos, latent, mask):
        q = torch.tanh(self.charge(latent)).squeeze(-1)    # [B,N] in [-1,1]
        rp = pos.unsqueeze(2) - pos.unsqueeze(1)           # [B,N,N,3]  (force on i from j)
        dist = rp.norm(dim=-1, keepdim=True).clamp(min=0.3)
        unit = rp / dist
        qq = (q.unsqueeze(2) * q.unsqueeze(1)).unsqueeze(-1)   # same sign>0 -> repel along +unit
        force = self.scale * qq / (dist ** 2) * unit
        m = mask.unsqueeze(1).unsqueeze(-1)
        return (force * m).sum(2) * mask.unsqueeze(-1)


class DynCellV2(nn.Module):
    def __init__(self, static_dim, latent, hidden):
        super().__init__()
        self.embed = mlp(3 + static_dim + latent, hidden, hidden)   # vel+static+latent (NO abs pos)
        self.inter = RelInteraction(hidden, hidden)
        self.out = mlp(hidden, hidden, 3)
        self.force = ForcePrior(latent)

    def forward(self, pos, vel, static, latent, mask):
        h = self.embed(torch.cat([vel, static, latent], -1))
        h = self.inter(h, pos, vel, mask)
        accel = self.out(h) * mask.unsqueeze(-1)
        return accel + self.force(pos, latent, mask)


class IPEv2(nn.Module):
    def __init__(self, hidden=128, latent=8):
        super().__init__()
        sdim = len(STATIC_IDX)
        self.latent_inf = LatentInference(DYN_DIM + sdim, hidden, latent)
        self.cell = DynCellV2(sdim, latent, hidden)
        self.dt = DT

    def forward(self, states, mask, t0, obs_w, horizon):
        dyn = torch.cat([states[..., POS], states[..., VEL]], -1)
        static = states[..., STATIC_IDX]
        obs = torch.cat([dyn[:, t0 - obs_w:t0], static[:, t0 - obs_w:t0]], -1)
        latent = self.latent_inf(obs, mask)
        st = static[:, t0]
        pos, vel = dyn[:, t0, :, :3], dyn[:, t0, :, 3:]
        preds = []
        for _ in range(horizon):
            accel = self.cell(pos, vel, st, latent, mask)
            vel = vel + accel
            pos = pos + vel * self.dt
            preds.append(torch.cat([pos, vel], -1))
        return torch.stack(preds, 1), latent


def ballistic_mse(states, mask, t0, hor, dev):
    dyn = torch.cat([states[..., POS], states[..., VEL]], -1)
    p0, v0 = dyn[:, t0, :, :3], dyn[:, t0, :, 3:]
    ks = torch.arange(1, hor + 1, device=dev).view(1, -1, 1, 1)
    cvpos = p0.unsqueeze(1) + v0.unsqueeze(1) * ks * DT
    cvvel = v0.unsqueeze(1).expand(-1, hor, -1, -1)
    return rollout_loss(torch.cat([cvpos, cvvel], -1), states, mask, t0, hor).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool", type=int, default=20000)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    # CLEVRER training data
    with h5py.File(CLEVRER_H5, "r") as f:
        pool = min(args.pool, f["states"].shape[0])
        Sc = np.asarray(f["states"][:pool])
        Mc = np.asarray(f["masks"][:pool, 0, :])
    Tc = Sc.shape[1]; t0c = Tc // 2
    Sc_t = torch.from_numpy(Sc).float(); Mc_t = torch.from_numpy(Mc).float()

    model = IPEv2().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(0)
    print(f"IPEv2 train on CLEVRER (t0={t0c}, obs=8, hor=8), params="
          f"{sum(p.numel() for p in model.parameters()):,}")
    model.train()
    for step in range(1, args.steps + 1):
        bi = torch.randint(0, pool, (args.batch,), generator=g)
        s = Sc_t[bi].to(dev); m = Mc_t[bi].to(dev)
        pred, _ = model(s, m, t0c, 8, 8)
        loss = rollout_loss(pred, s, m, t0c, 8)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 300 == 0:
            print(f"  step {step}  CLEVRER rollout MSE {loss.item():.4f}")

    # Transfer eval on ComPhy (zero-shot)
    d = np.load(COMPHY_CACHE)
    S, M, CH = d["states"], d["masks"], d["charged"]
    t0p, hor = 45, 8
    s_all = torch.from_numpy(S).float().to(dev)
    m_all = torch.from_numpy(M[:, t0p, :]).float().to(dev)
    print(f"\nZero-shot transfer: CLEVRER-trained IPEv2 -> ComPhy ({len(S)} scenes)")
    print(f"{'subset':<12}{'n':>6}{'ballistic':>12}{'IPEv2':>10}{'gain':>8}  (IPE was -1% neutral)")
    model.eval()
    with torch.no_grad():
        for name, sel in [("neutral", CH == 0), ("charged", CH == 1),
                          ("all", np.ones_like(CH, bool))]:
            idx = np.where(sel)[0]
            s, m = s_all[idx], m_all[idx]
            bal = ballistic_mse(s, m, t0p, hor, dev)
            pred, _ = model(s, m, t0p, 8, hor)
            ipe = rollout_loss(pred, s, m, t0p, hor).item()
            print(f"{name:<12}{len(idx):>6}{bal:>12.4f}{ipe:>10.4f}{(1-ipe/bal)*100:>7.0f}%")


if __name__ == "__main__":
    main()
