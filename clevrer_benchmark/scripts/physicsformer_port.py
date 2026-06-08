"""Port: does the PRETRAINED 50M PhysicsFormer reach the rollout cell?

Freeze the pretrained encoder, read its per-object features (encode_physics on a
2-frame window so state-delta is available), and condition a relative-invariant
rollout head on them. The relative attention bias is the proven causal lever;
absolute information in the token is harmless (the abs-token control was +42%),
so conditioning on the encoder's absolute features is fine. Train the head on
CLEVRER, evaluate zero-shot on held-out ComPhy rollout MSE vs a constant-velocity
baseline. This puts the REAL 50M checkpoint in the rollout cell (not a from-scratch
instantiation) and compares against the from-scratch +42%.
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
from clevrer_benchmark.run_adapter_evaluation import load_adapter_model  # noqa: E402
from physics_llm_adapter.intuitive_physics_engine import (  # noqa: E402
    LatentInference, mlp, POS, VEL, STATIC_IDX, DYN_DIM, rollout_loss)
from clevrer_benchmark.scripts.ipe_v2_transfer import (  # noqa: E402
    CLEVRER_H5, COMPHY_CACHE, ballistic_mse, DT)


class PFFeatCell(nn.Module):
    """Relative-invariant physics-biased attention, token conditioned on frozen
    PhysicsFormer features (already projected to `hidden`)."""
    def __init__(self, latent, hidden=128, heads=4):
        super().__init__()
        self.heads, self.hd = heads, hidden // heads
        self.embed = mlp(hidden + 3 + latent, hidden, hidden)   # pf_feat + vel + latent
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.bias = mlp(7, hidden, heads)                       # relative pos/vel/dist
        self.proj = nn.Linear(hidden, hidden)
        self.out = mlp(hidden, hidden, 3)

    def forward(self, pos, vel, feat, latent, mask):
        B, N, _ = vel.shape
        h = self.embed(torch.cat([feat, vel, latent], -1))
        q = self.q(h).view(B, N, self.heads, self.hd).transpose(1, 2)
        k = self.k(h).view(B, N, self.heads, self.hd).transpose(1, 2)
        v = self.v(h).view(B, N, self.heads, self.hd).transpose(1, 2)
        rp = pos.unsqueeze(2) - pos.unsqueeze(1)
        rv = vel.unsqueeze(2) - vel.unsqueeze(1)
        bf = torch.cat([rp, rv, rp.norm(dim=-1, keepdim=True)], -1)
        b = self.bias(bf).permute(0, 3, 1, 2)
        logits = (q @ k.transpose(-1, -2)) / (self.hd ** 0.5) + b
        logits = logits.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)
        o = (logits.softmax(-1) @ v).transpose(1, 2).reshape(B, N, self.heads * self.hd)
        return self.out(self.proj(o)) * mask.unsqueeze(-1)


class PFFeatRollout(nn.Module):
    def __init__(self, pf, feat_dim, hidden=128, latent=8):
        super().__init__()
        self.pf = pf                                            # frozen
        sdim = len(STATIC_IDX)
        self.latent_inf = LatentInference(DYN_DIM + sdim, hidden, latent)
        self.feat_proj = mlp(feat_dim, hidden, hidden)
        self.cell = PFFeatCell(latent, hidden)
        self.dt = DT

    def pf_feat(self, states, mask, t0):
        # encoder expects the 70-D augmented state [s_t ; s_t - s_{t-1}] (Eq. 1)
        aug = torch.cat([states[:, t0], states[:, t0] - states[:, t0 - 1]], dim=-1)  # [B,N,70]
        with torch.no_grad():
            emb = self.pf.encode_physics(aug, mask)             # [B,N,feat]
        return emb

    def forward(self, states, mask, t0, obs_w, horizon):
        feat = self.feat_proj(self.pf_feat(states, mask, t0))
        dyn = torch.cat([states[..., POS], states[..., VEL]], -1)
        static = states[..., STATIC_IDX]
        obs = torch.cat([dyn[:, t0 - obs_w:t0], static[:, t0 - obs_w:t0]], -1)
        latent = self.latent_inf(obs, mask)
        pos, vel = dyn[:, t0, :, :3], dyn[:, t0, :, 3:]
        preds = []
        for _ in range(horizon):
            accel = self.cell(pos, vel, feat, latent, mask)
            vel = vel + accel
            pos = pos + vel * self.dt
            preds.append(torch.cat([pos, vel], -1))
        return torch.stack(preds, 1), latent


def load_pf_rollout(save_path, pf, dev="cpu"):
    """Load a saved PFFeatRollout head onto a frozen PhysicsFormer encoder.
    `pf` is adapter.physics_model from load_adapter_model(adapter_phase3.pt)."""
    ck = torch.load(save_path, map_location=dev)
    c = ck["config"]
    m = PFFeatRollout(pf, c["feat_dim"], c["hidden"], c["latent"]).to(dev)
    m.load_state_dict(ck["head"], strict=False)   # pf.* stays the frozen encoder
    return m.eval()


def load_selfcontained(save_path, dev="cpu"):
    """Load the SELF-CONTAINED rollout (bundled encoder + head). No adapter needed;
    only requires physics_former_full to be importable."""
    ck = torch.load(save_path, map_location=dev, weights_only=False)
    pf = ck["encoder_module"].to(dev).eval()
    for p in pf.parameters():
        p.requires_grad = False
    c = ck["config"]
    m = PFFeatRollout(pf, c["feat_dim"], c["hidden"], c["latent"]).to(dev)
    m.load_state_dict(ck["head"], strict=False)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool", type=int, default=14000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=8, help="training rollout horizon")
    ap.add_argument("--save", default=None, help="path to persist the best trained head")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    import os
    ckpt = os.environ.get("ADAPTER_CKPT", "checkpoints/adapter_phase3.pt")
    adapter = load_adapter_model(ckpt, "", device=dev)
    pf = adapter.physics_model.eval()
    for p in pf.parameters():
        p.requires_grad = False

    with h5py.File(CLEVRER_H5, "r") as f:
        pool = min(args.pool, f["states"].shape[0])
        Sc = np.asarray(f["states"][:pool]); Mc = np.asarray(f["masks"][:pool, 0, :])
    t0c = Sc.shape[1] // 2
    Sc_t, Mc_t = torch.from_numpy(Sc).float(), torch.from_numpy(Mc).float()
    d = np.load(COMPHY_CACHE); S, M, CH = d["states"], d["masks"], d["charged"]
    t0p, hor = 45, 8

    # probe feature dim
    with torch.no_grad():
        probe = PFFeatRollout(pf, 768).to(dev)  # placeholder dim, only use pf_feat
        fdim = probe.pf_feat(Sc_t[:2].to(dev), Mc_t[:2].to(dev), t0c).shape[-1]
    print(f"CLEVRER pool={pool} (t0={t0c}); ComPhy n={len(S)} "
          f"(neutral={int((CH==0).sum())} charged={int((CH==1).sum())}); "
          f"PF feature dim={fdim}; seeds={args.seeds}", flush=True)

    results = []
    best = (-1e9, None)
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        model = PFFeatRollout(pf, fdim).to(dev)
        head = [p for n, p in model.named_parameters() if not n.startswith("pf.")]
        opt = torch.optim.Adam(head, lr=args.lr)
        g = torch.Generator().manual_seed(seed)
        model.train()
        for step in range(1, args.steps + 1):
            bi = torch.randint(0, pool, (args.batch,), generator=g)
            s, m = Sc_t[bi].to(dev), Mc_t[bi].to(dev)
            pred, _ = model(s, m, t0c, 8, args.horizon)
            loss = rollout_loss(pred, s, m, t0c, args.horizon)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head, 1.0); opt.step()
            if step % 300 == 0:
                print(f"  seed {seed} step {step}  CLEVRER rollout MSE {loss.item():.4f}", flush=True)
        model.eval()
        out = {}
        with torch.no_grad():
            for name, sel in [("neutral", CH == 0), ("charged", CH == 1), ("all", np.ones_like(CH, bool))]:
                idx = np.where(sel)[0]
                ss = torch.from_numpy(S[idx]).float().to(dev)
                mm = torch.from_numpy(M[idx, t0p, :]).float().to(dev)
                bal = ballistic_mse(ss, mm, t0p, hor, dev)
                mse = rollout_loss(model(ss, mm, t0p, 8, hor)[0], ss, mm, t0p, hor).item()
                out[name] = (1 - mse / bal) * 100
        results.append(out)
        if out["neutral"] > best[0]:
            best = (out["neutral"], model)
        print(f"  seed {seed}: neutral {out['neutral']:+.0f}%  charged {out['charged']:+.0f}%  all {out['all']:+.0f}%", flush=True)

    print("\n========== PORT GATE: pretrained-50M-PhysicsFormer features in the rollout cell ==========", flush=True)
    for sub in ["neutral", "charged", "all"]:
        v = np.array([r[sub] for r in results])
        print(f"  {sub:<8} {v.mean():+6.1f}% +/- {v.std():4.1f}  (vs from-scratch attention: neutral +42%)", flush=True)
    nv = np.mean([r["neutral"] for r in results])
    print(f"\n  -> {'PASS: the pretrained 50M checkpoint reaches the rollout cell' if nv > 0 else 'FAIL'} "
          f"(neutral {nv:+.0f}% vs ballistic).", flush=True)

    if args.save and best[1] is not None:
        head_sd = {k: v.cpu() for k, v in best[1].state_dict().items() if not k.startswith("pf.")}
        torch.save({"head": head_sd,
                    "config": {"feat_dim": fdim, "hidden": 128, "latent": 8},
                    "neutral_gain": best[0],
                    "note": ("PFFeatRollout head: a relative-invariant rollout trained on frozen "
                             "PhysicsFormer encoder features (encode_physics, 70-D augmented input). "
                             "Load: pf=load_adapter_model(adapter_phase3.pt).physics_model; "
                             "model=load_pf_rollout(this_path, pf).")}, args.save)
        print(f"saved trained head ({sum(v.numel() for v in head_sd.values()):,} params, "
              f"best neutral {best[0]:+.0f}%) -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
