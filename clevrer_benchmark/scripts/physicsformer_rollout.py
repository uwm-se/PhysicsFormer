"""Future-work build: PhysicsFormer's physics-biased attention in the rollout
cell, with a controlled isolation of WHERE translation-invariance has to live.

Three configs (same ~0.27M attention rollout, trained from scratch on CLEVRER,
evaluated zero-shot on held-out ComPhy rollout MSE vs a constant-velocity
baseline), each over several seeds:

  abs-full   : attention bias from ABSOLUTE pos/vel  + absolute pos in token
               (fully non-invariant; expected to fail, cf. IPEv2 abs = -1%)
  abs-token  : attention bias from RELATIVE pos/vel  + absolute pos in token
  rel        : attention bias from RELATIVE pos/vel  + no absolute pos
               (fully translation-invariant; cf. IPEv2 rel = +43%)

GATE: beat ballistic on neutral ComPhy. The abs-full vs {abs-token, rel}
contrast isolates whether the relative *attention bias* is the load-bearing
invariance, independent of whether the token embedding sees absolute position.
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
from clevrer_benchmark.scripts.ipe_v2_transfer import (  # noqa: E402
    CLEVRER_H5, COMPHY_CACHE, ballistic_mse, DT)

CONFIGS = {
    "abs-full":  dict(bias="abs", use_abs=True),
    "abs-token": dict(bias="rel", use_abs=True),
    "rel":       dict(bias="rel", use_abs=False),
}


class PhysAttnCell(nn.Module):
    """PhysicsFormer-style physics-biased multi-head self-attention.

    `bias`='rel' computes the additive attention bias from RELATIVE pos/vel
    (translation-invariant); 'abs' from the two objects' ABSOLUTE pos/vel
    (not invariant). `use_abs` toggles absolute position in the token embedding.
    """
    def __init__(self, static_dim, latent, hidden=128, heads=4,
                 bias="rel", use_abs=False):
        super().__init__()
        self.bias_mode, self.use_abs = bias, use_abs
        self.heads, self.hd = heads, hidden // heads
        in_dim = (3 if use_abs else 0) + 3 + static_dim + latent
        self.embed = mlp(in_dim, hidden, hidden)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.bias = mlp(12 if bias == "abs" else 7, hidden, heads)
        self.proj = nn.Linear(hidden, hidden)
        self.out = mlp(hidden, hidden, 3)

    def forward(self, pos, vel, static, latent, mask):
        B, N, _ = vel.shape
        feat = [vel, static, latent]
        if self.use_abs:
            feat = [pos] + feat
        h = self.embed(torch.cat(feat, -1))
        q = self.q(h).view(B, N, self.heads, self.hd).transpose(1, 2)
        k = self.k(h).view(B, N, self.heads, self.hd).transpose(1, 2)
        v = self.v(h).view(B, N, self.heads, self.hd).transpose(1, 2)
        if self.bias_mode == "abs":
            pi = pos.unsqueeze(2).expand(B, N, N, 3)
            pj = pos.unsqueeze(1).expand(B, N, N, 3)
            vi = vel.unsqueeze(2).expand(B, N, N, 3)
            vj = vel.unsqueeze(1).expand(B, N, N, 3)
            bf = torch.cat([pi, pj, vi, vj], -1)              # absolute, 12-d
        else:
            rp = pos.unsqueeze(2) - pos.unsqueeze(1)
            rv = vel.unsqueeze(2) - vel.unsqueeze(1)
            bf = torch.cat([rp, rv, rp.norm(dim=-1, keepdim=True)], -1)   # relative, 7-d
        b = self.bias(bf).permute(0, 3, 1, 2)                 # [B,heads,N,N]
        logits = (q @ k.transpose(-1, -2)) / (self.hd ** 0.5) + b
        logits = logits.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)
        o = (logits.softmax(-1) @ v).transpose(1, 2).reshape(B, N, self.heads * self.hd)
        return self.out(self.proj(o)) * mask.unsqueeze(-1)


class PFRollout(nn.Module):
    def __init__(self, hidden=128, latent=8, heads=4, bias="rel", use_abs=False):
        super().__init__()
        sdim = len(STATIC_IDX)
        self.latent_inf = LatentInference(DYN_DIM + sdim, hidden, latent)
        self.cell = PhysAttnCell(sdim, latent, hidden, heads, bias, use_abs)
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


def train_eval(cfg, seed, Sc_t, Mc_t, t0c, S, M, CH, args, dev):
    torch.manual_seed(seed)
    model = PFRollout(bias=cfg["bias"], use_abs=cfg["use_abs"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(1, args.steps + 1):
        bi = torch.randint(0, Sc_t.shape[0], (args.batch,), generator=g)
        s, m = Sc_t[bi].to(dev), Mc_t[bi].to(dev)
        pred, _ = model(s, m, t0c, 8, 8)
        loss = rollout_loss(pred, s, m, t0c, 8)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    t0p, hor = 45, 8
    s_all = torch.from_numpy(S).float().to(dev)
    m_all = torch.from_numpy(M[:, t0p, :]).float().to(dev)
    model.eval()
    out = {}
    with torch.no_grad():
        for name, sel in [("neutral", CH == 0), ("charged", CH == 1), ("all", np.ones_like(CH, bool))]:
            idx = np.where(sel)[0]
            s, m = s_all[idx], m_all[idx]
            bal = ballistic_mse(s, m, t0p, hor, dev)
            mse = rollout_loss(model(s, m, t0p, 8, hor)[0], s, m, t0p, hor).item()
            out[name] = (1 - mse / bal) * 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    with h5py.File(CLEVRER_H5, "r") as f:
        pool = min(args.pool, f["states"].shape[0])
        Sc = np.asarray(f["states"][:pool]); Mc = np.asarray(f["masks"][:pool, 0, :])
    t0c = Sc.shape[1] // 2
    Sc_t, Mc_t = torch.from_numpy(Sc).float(), torch.from_numpy(Mc).float()
    d = np.load(COMPHY_CACHE); S, M, CH = d["states"], d["masks"], d["charged"]
    print(f"CLEVRER pool={pool} (t0={t0c}); ComPhy n={len(S)} "
          f"(neutral={int((CH==0).sum())} charged={int((CH==1).sum())}); "
          f"seeds={args.seeds}, steps={args.steps}", flush=True)

    results = {}
    for name, cfg in CONFIGS.items():
        per_seed = [train_eval(cfg, sd, Sc_t, Mc_t, t0c, S, M, CH, args, dev)
                    for sd in range(args.seeds)]
        results[name] = per_seed
        for sub in ["neutral", "charged", "all"]:
            v = np.array([r[sub] for r in per_seed])
            print(f"  {name:<10} {sub:<8} {v.mean():+6.1f}% +/- {v.std():4.1f}  "
                  f"(seeds: {', '.join(f'{x:+.0f}' for x in v)})", flush=True)

    print("\n========== GATE: neutral rollout vs ballistic (mean over seeds) ==========", flush=True)
    for name in CONFIGS:
        v = np.array([r["neutral"] for r in results[name]])
        print(f"  {name:<10}: {v.mean():+5.1f}%  ->  "
              f"{'PASS' if v.mean() > 0 else 'FAIL'}", flush=True)
    nf = np.mean([r['neutral'] for r in results['abs-full']])
    nt = np.mean([r['neutral'] for r in results['abs-token']])
    nr = np.mean([r['neutral'] for r in results['rel']])
    print(f"\n  Isolation: absolute bias {nf:+.0f}%  vs  relative bias "
          f"(abs-token {nt:+.0f}%, rel {nr:+.0f}%).", flush=True)
    print(f"  -> the relative attention bias is {'load-bearing' if nt - nf > 10 else 'not clearly load-bearing'} "
          f"for reaching the rollout cell.", flush=True)


if __name__ == "__main__":
    main()
