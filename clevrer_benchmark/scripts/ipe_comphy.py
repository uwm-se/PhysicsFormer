"""IPE where ballistic MUST fail: ComPhy hidden charge -> curved trajectories.

Trains a fresh IPE on ComPhy trajectories (so its latent-inference head can
recover charge from observed motion) and tests rollout error vs. a
constant-velocity (ballistic) baseline, SPLIT by whether the scene contains
charged objects. The hypothesis CLEVRER could not test:
  charged scenes : IPE << ballistic   (ballistic ignores attraction/repulsion)
  neutral scenes : IPE ~= ballistic   (as on CLEVRER)
A large charged-scene gap is the proof the learned simulator earns its keep.

First run builds a state cache from the ComPhy annotations (slow); later runs
reuse it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from physics_llm_adapter.intuitive_physics_engine import IPE, POS, VEL, rollout_loss  # noqa: E402
from comphy_benchmark.scene_converter import comphy_scene_to_state_tensor  # noqa: E402

COMPHY_DIR = os.environ.get("COMPHY_DIR", r"D:\comphy")
TF, NMAX = 100, 6
T0, OBS, HOR = 45, 10, 18


def build_cache(n, cache):
    allf = sorted(glob.glob(os.path.join(COMPHY_DIR, "target_annotation", "*", "*.json")))
    stride = max(1, len(allf) // n)             # spread across the full range
    files = allf[::stride][:n]                  # (charged scenes are later in the index)
    S, M, CH = [], [], []
    for k, fp in enumerate(files):
        try:
            ann = json.load(open(fp, "r", encoding="utf-8"))
            states, masks, meta = comphy_scene_to_state_tensor(ann)   # [T,N,35],[T,N]
        except Exception:
            continue
        T, N, _ = states.shape
        if T < T0 + HOR + 1:
            continue
        st = np.zeros((TF, NMAX, 35), np.float32)
        mk = np.zeros((TF, NMAX), np.float32)
        t, nn = min(T, TF), min(N, NMAX)
        st[:t, :nn] = states[:t, :nn]
        mk[:t, :nn] = masks[:t, :nn]
        charged = any(abs(float(o.get("comphy_charge") or 0)) > 0
                      for o in meta["objects"])
        S.append(st); M.append(mk); CH.append(1 if charged else 0)
        if (k + 1) % 1000 == 0:
            print(f"  converted {k+1} ({len(S)} usable)")
    S, M, CH = np.stack(S), np.stack(M), np.array(CH, np.int64)
    np.savez(cache, states=S, masks=M, charged=CH)
    print(f"cached {len(S)} scenes -> {cache}")
    return S, M, CH


def ballistic_mse(states, mask, dev):
    dyn = torch.cat([states[..., POS], states[..., VEL]], -1)
    p0, v0 = dyn[:, T0, :, :3], dyn[:, T0, :, 3:]
    ks = torch.arange(1, HOR + 1, device=dev).view(1, -1, 1, 1)
    cvpos = p0.unsqueeze(1) + v0.unsqueeze(1) * ks * (1.0 / 25.0)
    cvvel = v0.unsqueeze(1).expand(-1, HOR, -1, -1)
    return rollout_loss(torch.cat([cvpos, cvvel], -1), states, mask, T0, HOR).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--cache", default="comphy_benchmark/results/ipe_comphy_cache.npz")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    if os.path.exists(args.cache):
        d = np.load(args.cache)
        S, M, CH = d["states"], d["masks"], d["charged"]
        print(f"loaded cache: {len(S)} scenes")
    else:
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        S, M, CH = build_cache(args.n, args.cache)
    print(f"charged scenes: {CH.mean()*100:.1f}%")

    perm = np.random.default_rng(0).permutation(len(S))   # mix charged into train+eval
    S, M, CH = S[perm], M[perm], CH[perm]
    ntr = int(0.85 * len(S))
    Str = torch.from_numpy(S[:ntr]).float()
    Mtr = torch.from_numpy(M[:ntr, T0, :]).float()     # mask at t0
    ev_s = torch.from_numpy(S[ntr:]).float().to(dev)
    ev_m = torch.from_numpy(M[ntr:, T0, :]).float().to(dev)
    ev_ch = CH[ntr:]

    model = IPE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    ch_tr = CH[:ntr]
    pos = np.where(ch_tr == 1)[0]
    neg = np.where(ch_tr == 0)[0]
    rng_np = np.random.default_rng(0)
    print(f"training IPE on ComPhy | train={ntr} charged={len(pos)} neutral={len(neg)} "
          f"(charged upweighted to 50%) t0={T0} obs={OBS} hor={HOR}")
    model.train()
    for step in range(1, args.steps + 1):
        h = args.batch // 2
        bi = torch.from_numpy(np.concatenate([
            rng_np.choice(pos, h), rng_np.choice(neg, args.batch - h)]))
        s = Str[bi].to(dev); m = Mtr[bi].to(dev)
        pred, _ = model(s, m, T0, OBS, HOR)
        loss = rollout_loss(pred, s, m, T0, HOR)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 300 == 0:
            print(f"  step {step}  train rollout MSE {loss.item():.4f}")

    model.eval()
    print("\n--- held-out rollout MSE (lower=better), split by charge ---")
    print(f"{'subset':<18}{'n':>6}{'ballistic':>12}{'IPE':>10}{'IPE gain':>12}")
    with torch.no_grad():
        for name, sel in [("charged", ev_ch == 1), ("neutral", ev_ch == 0),
                          ("all", np.ones_like(ev_ch, bool))]:
            idx = np.where(sel)[0]
            if len(idx) == 0:
                continue
            s = ev_s[idx]; m = ev_m[idx]
            bal = ballistic_mse(s, m, dev)
            pred, _ = model(s, m, T0, OBS, HOR)
            ipe = rollout_loss(pred, s, m, T0, HOR).item()
            print(f"{name:<18}{len(idx):>6}{bal:>12.4f}{ipe:>10.4f}"
                  f"{(1-ipe/bal)*100:>11.0f}%")


if __name__ == "__main__":
    main()
