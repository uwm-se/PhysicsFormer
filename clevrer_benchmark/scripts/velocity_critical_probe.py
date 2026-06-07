"""Is velocity in the prefix, or is the LM just not trained to use it?

The QA velocity ablation can't separate these. Here we pose two velocity-critical
tasks whose ground truth is computable from position+velocity alone, and ask what
is decodable from the *exact pooled prefix the LM consumes*:

  mean-speed (regression, R^2)   : pure velocity magnitude, non-relational.
  will-collide (binary, acc/AUC) : ballistic rollout from t0 -> any pair within
                                   threshold within K frames; velocity + relational.

Feature conditions:
  raw_posvel   : per-object [pos,vel] flattened  (upper bound: task is solvable)
  raw_posonly  : per-object [pos]                 (lower bound: pos alone can't)
  prefix_full  : frozen-encoder pooled prefix from the real state  (what the LM sees)
  prefix_novel : frozen-encoder pooled prefix from velocity-zeroed state (control)

Reading:
  prefix_full mean-speed R^2 high  -> velocity magnitude IS preserved in the prefix;
      the QA zero-contribution is a TRAINING-PRESSURE problem (CLEVRER never required
      it) -- velocity-critical data would move it off zero.
  prefix_full will-collide ~ base while raw_posvel high -> the RELATIONAL dynamics
      are destroyed by pooling; need object-centric prefix, not just data pressure.
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

DEFAULT_H5 = (r"C:\Users\jpoko\source\repos\homework\CascadeProjects\windsurf-project"
              r"\compsac_2026_code\data\clevrer_training_expanded.h5")
DEFAULT_CKPT = (r"C:\Users\jpoko\source\repos\homework\CascadeProjects\windsurf-project"
                r"\compsac_2026_code\checkpoints\adapter_phase3.pt")
DT = 1.0 / 25.0


def _labels(states, masks, t0, K, thresh):
    """mean-speed (per scene) and will-collide (ballistic, any pair) at t0."""
    pos = states[:, t0, :, 0:3]          # [B,N,3]
    vel = states[:, t0, :, 3:6]
    m = masks                            # [B,N]
    speed = np.linalg.norm(vel, axis=-1) # [B,N]
    mean_speed = (speed * m).sum(1) / np.clip(m.sum(1), 1, None)   # [B]
    B, N, _ = pos.shape
    collide = np.zeros(B, dtype=np.float32)
    for b in range(B):
        idx = np.where(m[b] > 0)[0]
        if len(idx) < 2:
            continue
        hit = False
        for a_i in range(len(idx)):
            for b_i in range(a_i + 1, len(idx)):
                i, j = idx[a_i], idx[b_i]
                d0 = np.linalg.norm(pos[b, i] - pos[b, j])
                if d0 < thresh:
                    continue                       # already touching at t0
                mind = d0
                for k in range(1, K + 1):
                    pi = pos[b, i] + vel[b, i] * k * DT
                    pj = pos[b, j] + vel[b, j] * k * DT
                    mind = min(mind, np.linalg.norm(pi - pj))
                if mind < thresh:
                    hit = True
        collide[b] = 1.0 if hit else 0.0
    return mean_speed.astype(np.float32), collide


def _raw_feats(states, masks, t0, use_vel):
    pos = states[:, t0, :, 0:3]; vel = states[:, t0, :, 3:6]; m = masks[..., None]
    if use_vel:
        f = np.concatenate([pos * m, vel * m], axis=-1)   # [B,N,6]
    else:
        f = pos * m                                       # [B,N,3]
    return f.reshape(f.shape[0], -1)                      # flatten objects


@torch.no_grad()
def _prefix_feats(adapter, states, masks, t0, dev, zero_vel, bs=256):
    out = []
    s = states[:, t0:t0 + 1].copy()              # [B,1,N,35]
    if zero_vel:
        s[:, :, :, 3:6] = 0.0; s[:, :, :, 10:13] = 0.0
    m2 = masks                                   # [B,N]
    for i in range(0, s.shape[0], bs):
        st = torch.from_numpy(s[i:i + bs]).float().to(dev)         # [b,1,N,35]
        mk = torch.from_numpy(m2[i:i + bs]).float().to(dev)        # [b,N]
        feat = adapter.extract_physics_features(st, mk)            # [b, dim]
        out.append(feat.float().cpu().numpy())
    return np.concatenate(out, 0)


def _standardize(tr, te):
    mu, sd = tr.mean(0, keepdims=True), tr.std(0, keepdims=True) + 1e-6
    return (tr - mu) / sd, (te - mu) / sd


def _train_head(Xtr, ytr, Xte, yte, dev, task, steps=600):
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
    out_dim = 1 if task == "reg" else 2
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.SiLU(),
                        nn.Linear(128, out_dim)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    if task == "reg":
        ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev).view(-1, 1)
        lossf = nn.MSELoss()
    else:
        ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
        lossf = nn.CrossEntropyLoss()
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(steps):
        bi = torch.randint(0, Xtr_t.shape[0], (256,), generator=g).to(dev)
        out = net(Xtr_t[bi])
        loss = lossf(out if task != "reg" else out, ytr_t[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte_t).cpu().numpy()
    if task == "reg":
        ss_res = ((yte - pred[:, 0]) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-9
        return 1 - ss_res / ss_tot                        # R^2
    p = pred.argmax(1)
    return (p == yte).mean()                              # accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--adapter_checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--t0", type=int, default=64)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--thresh", type=float, default=0.9)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    with h5py.File(args.h5, "r") as f:
        n = min(args.n, f["states"].shape[0])
        states = np.asarray(f["states"][:n])
        masks = np.asarray(f["masks"][:n, 0, :])         # frame-0 object mask
    print(f"scenes={n} t0={args.t0} K={args.K} thresh={args.thresh}")

    mean_speed, collide = _labels(states, masks, args.t0, args.K, args.thresh)
    print(f"will-collide base rate: {collide.mean()*100:.1f}%   "
          f"mean-speed range [{mean_speed.min():.2f},{mean_speed.max():.2f}]")

    print("Loading adapter (frozen encoder)...")
    adapter = load_adapter_model(args.adapter_checkpoint, "", device=dev); adapter.eval()
    feats = {
        "raw_posvel":  _raw_feats(states, masks, args.t0, True),
        "raw_posonly": _raw_feats(states, masks, args.t0, False),
        "prefix_full": _prefix_feats(adapter, states, masks, args.t0, dev, False),
        "prefix_novel": _prefix_feats(adapter, states, masks, args.t0, dev, True),
    }
    ntr = int(0.8 * n)
    print(f"\n{'condition':<14}{'mean-speed R2':>16}{'will-collide acc':>18}")
    print("-" * 48)
    print(f"{'(base rate)':<14}{'--':>16}{max(collide.mean(),1-collide.mean())*100:>17.1f}%")
    for name, X in feats.items():
        Xtr, Xte = _standardize(X[:ntr], X[ntr:])
        r2 = _train_head(Xtr, mean_speed[:ntr], Xte, mean_speed[ntr:], dev, "reg")
        acc = _train_head(Xtr, collide[:ntr].astype(int), Xte, collide[ntr:].astype(int), dev, "clf")
        print(f"{name:<14}{r2:>15.2f}{acc*100:>17.1f}%")


if __name__ == "__main__":
    main()
