"""The IPE path, tested at its crux: is the IPE a useful dynamics ORACLE?

Option 2 showed the LLM uses dynamics when handed them as text. So the IPE path
(IPE simulate -> text readout -> LLM) succeeds iff the IPE can predict dynamics
well. We test the hardest honest version: predict ACTUAL future collisions (from
the real CLEVRER trajectory), and ask whether the learned IPE beats a naive
constant-velocity (ballistic) predictor -- because a ballistic baseline already
captures straight-line approaches; the IPE only earns its keep on the
interaction-driven (post-collision / chain) events a static rollout misses.

  IPE acc > ballistic acc  -> the simulator adds dynamics value; the path is live.
  IPE acc ~= ballistic     -> at CLEVRER's scale the learned sim isn't beating
                              straight lines yet (needs a stronger simulator).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from physics_llm_adapter.intuitive_physics_engine import IPE, POS, VEL  # noqa: E402

# --- reproducibility: resolve data/checkpoint paths from env vars (see REPRODUCTION.md) ---
import os as _os_repro
from pathlib import Path as _Path_repro
_REPRO_ROOT = _Path_repro(__file__).resolve().parents[3]

DEFAULT_H5 = _os_repro.environ.get("CLEVRER_H5", str(_REPRO_ROOT / "data" / "clevrer_training_expanded.h5"))
DEFAULT_IPE = r"checkpoints\ipe_clevrer.pt"
T0, HOR, DT, THRESH = 64, 8, 1.0 / 25.0, 0.9


def _any_collide(pos_seq, mask):
    """pos_seq [H,N,3]; any valid pair within THRESH at any step (excluding pairs
    already within THRESH at the first step)."""
    idx = np.where(mask > 0)[0]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            d = np.linalg.norm(pos_seq[:, i] - pos_seq[:, j], axis=-1)
            if d[0] < THRESH:
                continue
            if d.min() < THRESH:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--ipe", default=DEFAULT_IPE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    with h5py.File(args.h5, "r") as f:
        n = min(args.n, f["states"].shape[0])
        states = np.asarray(f["states"][:n])
        masks = np.asarray(f["masks"][:n, 0, :])

    model = IPE().to(dev)
    model.load_state_dict(torch.load(args.ipe, map_location=dev))
    model.eval()

    base = ball_c = ipe_c = pos_rate = tot = 0
    with torch.no_grad():
        for i in range(n):
            m = masks[i]
            if (m > 0).sum() < 2:
                continue
            # ground truth: actual future trajectory
            actual = states[i, T0 + 1:T0 + 1 + HOR, :, 0:3]              # [H,N,3]
            y = _any_collide(actual, m)
            # ballistic prediction (constant velocity from t0)
            p0, v0 = states[i, T0, :, 0:3], states[i, T0, :, 3:6]
            ks = np.arange(1, HOR + 1)[:, None, None]
            ball = p0[None] + v0[None] * ks * DT                        # [H,N,3]
            yb = _any_collide(ball, m)
            # IPE prediction
            s = torch.from_numpy(states[i:i + 1]).float().to(dev)
            mk = torch.from_numpy(masks[i:i + 1]).float().to(dev)
            pred, _ = model(s, mk, T0, model_obs_w(), HOR)
            ipe_pos = pred[0, :, :, 0:3].cpu().numpy()                  # [H,N,3]
            yi = _any_collide(ipe_pos, m)

            tot += 1
            pos_rate += int(y)
            base += int(False == y)            # majority "no" baseline placeholder
            ball_c += int(yb == y)
            ipe_c += int(yi == y)

    br = pos_rate / max(tot, 1)
    majority = max(br, 1 - br) * 100
    print(f"n={tot}  actual-collision rate={br*100:.1f}%  "
          f"majority-class baseline={majority:.1f}%")
    print(f"ballistic predicts actual collision: {100*ball_c/max(tot,1):.1f}%")
    print(f"IPE       predicts actual collision: {100*ipe_c/max(tot,1):.1f}%")
    print(f"IPE - ballistic: {100*(ipe_c-ball_c)/max(tot,1):+.1f} pp")


def model_obs_w():
    return 8


if __name__ == "__main__":
    main()
