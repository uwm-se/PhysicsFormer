"""Intuitive Physics Engine (IPE): a relational forward-dynamics simulator over
the 35-D CLEVRER/ComPhy state. Implements the four architecture changes the
ablations call for, in one model:

  (1) RELATIONAL representation -- an Interaction Network over object nodes and
      pairwise edges, replacing the mean-pooled static prefix (preserves the
      object<->state binding the counterfactual probe showed was lost).
  (2) FORWARD-DYNAMICS rollout  -- a learned transition cell integrated k steps,
      so a downstream consumer reads a *simulated trajectory*, not a snapshot.
  (3) NEXT-STATE objective      -- self-supervised rollout MSE that cannot be
      minimized without using velocity + interactions (removes the static
      shortcut that made velocity contribute 0 to accuracy).
  (4) LATENT-PROPERTY inference -- a system-ID head that infers per-object hidden
      parameters (a mass/charge proxy) from an observed window and conditions the
      rollout. This is what ComPhy's hidden attributes need; a schema slot alone
      cannot supply the inference.

This file is the simulator core. It trains standalone on CLEVRER trajectories
(``train`` below) with the rollout objective; the LLM adapter consumes the
rollout (per-object tokens) in place of ``create_prefix_tokens`` as the
integration step (see ``rollout_prefix``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

POS = slice(0, 3)
VEL = slice(3, 6)
DYN_DIM = 6                       # [pos(3), vel(3)]
STATIC_IDX = [13, 14, 15, 16, 17, 18]  # mass, radius, color(3), shape


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers += [nn.Linear(d, o)]
    return nn.Sequential(*layers)


class InteractionNet(nn.Module):
    """(1) One round of relational message passing: pairwise edges -> node update.

    Mask-aware: messages only flow from valid sender objects, and updates are
    gated by the receiver's validity. No pooling to a single vector -- per-object
    structure (and the identity<->state binding) is preserved throughout.
    """

    def __init__(self, node_dim, hidden):
        super().__init__()
        self.edge = mlp(2 * node_dim, hidden, hidden)
        self.node = mlp(node_dim + hidden, hidden, node_dim)

    def forward(self, nodes, mask):           # nodes [B,N,d], mask [B,N]
        B, N, d = nodes.shape
        ni = nodes.unsqueeze(2).expand(B, N, N, d)
        nj = nodes.unsqueeze(1).expand(B, N, N, d)
        e = self.edge(torch.cat([ni, nj], -1))        # [B,N,N,h]
        m = mask.unsqueeze(1).unsqueeze(-1)           # [B,1,N,1]  (sender j)
        e = e * m
        agg = e.sum(2) / m.sum(2).clamp(min=1.0)      # [B,N,h]
        upd = self.node(torch.cat([nodes, agg], -1))
        return nodes + upd * mask.unsqueeze(-1)


class LatentInference(nn.Module):
    """(4) System ID: infer a per-object latent (mass/charge proxy) from an
    observed motion window. A GRU summarizes each object's own motion; one
    interaction round lets the latent reflect how objects respond to each other
    (the only observable signature of hidden mass/charge)."""

    def __init__(self, in_dim, hidden, latent):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.inter = InteractionNet(hidden, hidden)
        self.head = mlp(hidden, hidden, latent)

    def forward(self, obs, mask):             # obs [B,Tobs,N,in_dim]
        B, T, N, F = obs.shape
        x = obs.permute(0, 2, 1, 3).reshape(B * N, T, F)
        _, h = self.gru(x)                    # [1, B*N, hidden]
        h = h[0].view(B, N, -1)
        h = self.inter(h, mask)
        return self.head(h)                   # [B,N,latent]


class DynamicsCell(nn.Module):
    """(2) One transition step: predict each object's ACCELERATION from its own
    state, static attributes, inferred latent, and relational interactions.
    Position is then integrated semi-implicitly from velocity (see IPE.forward),
    so the kinematic prior is built in and the net only learns forces/collisions
    -- which is stable and makes velocity load-bearing by construction."""

    def __init__(self, static_dim, latent, hidden):
        super().__init__()
        self.embed = mlp(DYN_DIM + static_dim + latent, hidden, hidden)
        self.inter = InteractionNet(hidden, hidden)
        self.out = mlp(hidden, hidden, 3)         # acceleration (delta-velocity)

    def forward(self, dyn, static, latent, mask):
        h = self.embed(torch.cat([dyn, static, latent], -1))
        h = self.inter(h, mask)
        return self.out(h) * mask.unsqueeze(-1)   # accel [B,N,3]


class IPE(nn.Module):
    def __init__(self, hidden=128, latent=8, fps=25.0):
        super().__init__()
        sdim = len(STATIC_IDX)
        self.latent_inf = LatentInference(DYN_DIM + sdim, hidden, latent)
        self.cell = DynamicsCell(sdim, latent, hidden)
        self.dt = 1.0 / fps          # CLEVRER velocity = per-frame delta * fps

    def _split(self, states):
        dyn = torch.cat([states[..., POS], states[..., VEL]], -1)   # [B,T,N,6]
        static = states[..., STATIC_IDX]                            # [B,T,N,s]
        return dyn, static

    def forward(self, states, mask, t0, obs_w, horizon):
        dyn, static = self._split(states)
        obs = torch.cat([dyn[:, t0 - obs_w:t0], static[:, t0 - obs_w:t0]], -1)
        latent = self.latent_inf(obs, mask)                # (4) infer hidden props
        st = static[:, t0]
        pos, vel = dyn[:, t0, :, :3], dyn[:, t0, :, 3:]
        preds = []
        for _ in range(horizon):                           # (2) semi-implicit rollout
            accel = self.cell(torch.cat([pos, vel], -1), st, latent, mask)
            vel = vel + accel
            pos = pos + vel * self.dt
            preds.append(torch.cat([pos, vel], -1))
        return torch.stack(preds, 1), latent               # [B,horizon,N,6]

    @torch.no_grad()
    def rollout_prefix(self, states, mask, t0, obs_w, horizon):
        """Integration hook: per-object simulated-trajectory features the LLM
        adapter consumes in place of the pooled prefix (one token per object,
        carrying its predicted future)."""
        preds, latent = self.forward(states, mask, t0, obs_w, horizon)
        # [B,N, horizon*6 + latent]: each object's predicted future + its latent
        B, H, N, D = preds.shape
        feat = preds.permute(0, 2, 1, 3).reshape(B, N, H * D)
        return torch.cat([feat, latent], -1)


def rollout_loss(pred, states, mask, t0, horizon):
    """(3) Next-state objective: rollout MSE over valid objects. Velocity and
    interactions are *required* to minimize it -- a static read cannot."""
    tgt = torch.cat([states[..., POS], states[..., VEL]], -1)[:, t0 + 1:t0 + 1 + horizon]
    m = mask.unsqueeze(1).unsqueeze(-1)
    return (((pred - tgt) ** 2) * m).sum() / (m.sum() * pred.size(-1)).clamp(min=1.0)


# ---------------------------------------------------------------------------
# Standalone training on CLEVRER trajectories (validates the simulator learns).
# ---------------------------------------------------------------------------

DEFAULT_H5 = (r"C:\Users\jpoko\source\repos\homework\CascadeProjects\windsurf-project"
              r"\compsac_2026_code\data\clevrer_training_expanded.h5")


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--obs_w", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    f = h5py.File(args.h5, "r")
    N_total, T, N_obj, _ = f["states"].shape
    # Use a fixed pool in memory for speed (first 20k samples).
    pool = min(20000, N_total)
    states_np = np.asarray(f["states"][:pool])
    masks_np = np.asarray(f["masks"][:pool])
    f.close()
    # Per-frame mask is constant in CLEVRER; use frame 0.
    obj_mask = masks_np[:, 0, :]                          # [pool, N]
    t0 = T // 2
    lo, hi = args.obs_w, T - args.horizon - 1
    if not (lo <= t0 <= hi):
        t0 = (lo + hi) // 2

    model = IPE(hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(0)
    train_hi = pool - 4096                          # reserve a fixed eval split

    # Fixed held-out eval batch (never trained on) + its static-frozen baseline.
    ev = np.arange(pool - 4096, pool - 4096 + 512)
    es = torch.from_numpy(states_np[ev]).float().to(dev)
    em = torch.from_numpy(obj_mask[ev]).float().to(dev)
    with torch.no_grad():
        edyn = torch.cat([es[..., POS], es[..., VEL]], -1)
        frozen = edyn[:, t0:t0 + 1].expand(-1, args.horizon, -1, -1)
        base_ref = rollout_loss(frozen, es, em, t0, args.horizon).item()

    def evaluate():
        model.eval()
        with torch.no_grad():
            p, _ = model(es, em, t0, args.obs_w, args.horizon)
            v = rollout_loss(p, es, em, t0, args.horizon).item()
        model.train()
        return v

    print(f"IPE training | pool={pool} T={T} N_obj={N_obj} t0={t0} "
          f"obs_w={args.obs_w} horizon={args.horizon} params="
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"  static-frozen (no-motion) eval baseline: {base_ref:.4f}")
    model.train()
    best = float("inf")
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, train_hi, (args.batch,), generator=g).numpy()
        s = torch.from_numpy(states_np[idx]).float().to(dev)
        m = torch.from_numpy(obj_mask[idx]).float().to(dev)
        pred, _ = model(s, m, t0, args.obs_w, args.horizon)
        loss = rollout_loss(pred, s, m, t0, args.horizon)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            ev_mse = evaluate(); best = min(best, ev_mse)
            print(f"  step {step:4d}  EVAL rollout MSE {ev_mse:.4f}   "
                  f"({(1-ev_mse/base_ref)*100:+.0f}% vs no-motion baseline)")

    final = evaluate()
    print(f"\nHeld-out eval rollout MSE {final:.4f} (best {best:.4f}) "
          f"vs static-frozen {base_ref:.4f}\n"
          f"-> simulator predicts dynamics {(1-best/base_ref)*100:.0f}% better than "
          f"the no-motion baseline (the regime the current model is stuck in).")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    train()
