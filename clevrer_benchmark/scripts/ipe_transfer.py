"""Do learned physics laws generalize? Train forward model on CLEVRER, test its
rollout on ComPhy ZERO-SHOT (no ComPhy training), split by charge.

  neutral ComPhy (shared rigid-body physics): IPE > ballistic  -> laws generalize
  charged ComPhy (a force never seen in CLEVRER): IPE ~ ballistic or worse

Compares the CLEVRER-trained IPE (checkpoints/ipe_clevrer.pt) against a
constant-velocity baseline on the cached ComPhy scenes. Reference upper bound:
a ComPhy-TRAINED IPE gets ~58% gain on neutral (ipe_comphy.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from physics_llm_adapter.intuitive_physics_engine import IPE, POS, VEL, rollout_loss  # noqa: E402

T0, OBS, HOR = 45, 8, 8          # match how ipe_clevrer.pt was trained (obs=8, hor=8)


def ballistic_mse(states, mask, dev):
    dyn = torch.cat([states[..., POS], states[..., VEL]], -1)
    p0, v0 = dyn[:, T0, :, :3], dyn[:, T0, :, 3:]
    ks = torch.arange(1, HOR + 1, device=dev).view(1, -1, 1, 1)
    cvpos = p0.unsqueeze(1) + v0.unsqueeze(1) * ks * (1.0 / 25.0)
    cvvel = v0.unsqueeze(1).expand(-1, HOR, -1, -1)
    return rollout_loss(torch.cat([cvpos, cvvel], -1), states, mask, T0, HOR).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipe", default="checkpoints/ipe_clevrer.pt")
    ap.add_argument("--cache", default="comphy_benchmark/results/ipe_comphy_cache.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    d = np.load(args.cache)
    S, M, CH = d["states"], d["masks"], d["charged"]
    print(f"ComPhy cache: {len(S)} scenes, {CH.mean()*100:.1f}% charged "
          f"(CLEVRER-trained IPE has seen NONE of it)")

    model = IPE().to(dev)
    model.load_state_dict(torch.load(args.ipe, map_location=dev))
    model.eval()

    s_all = torch.from_numpy(S).float().to(dev)
    m_all = torch.from_numpy(M[:, T0, :]).float().to(dev)

    print(f"\n{'subset':<16}{'n':>6}{'ballistic':>12}{'IPE(CLEVRER)':>14}{'IPE gain':>11}")
    print("-" * 60)
    with torch.no_grad():
        for name, sel in [("neutral", CH == 0), ("charged", CH == 1),
                          ("all", np.ones_like(CH, bool))]:
            idx = np.where(sel)[0]
            if len(idx) == 0:
                continue
            s, m = s_all[idx], m_all[idx]
            bal = ballistic_mse(s, m, dev)
            pred, _ = model(s, m, T0, OBS, HOR)
            ipe = rollout_loss(pred, s, m, T0, HOR).item()
            print(f"{name:<16}{len(idx):>6}{bal:>12.4f}{ipe:>14.4f}{(1-ipe/bal)*100:>10.0f}%")


if __name__ == "__main__":
    main()
