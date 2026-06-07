"""Wire PhysicsFormer's OWN prediction head into the oracle test.

PhysicsFormer was pretrained on next-state/delta prediction (a GNS-style forward
model) but is deployed as a single-frame encoder. Here we run it as the simulator
it was trained to be -- via its built-in predict_autoregressive rollout -- and
measure whether it predicts actual future collisions better than a constant-
velocity (ballistic) baseline, on CLEVRER. Comparable to ipe_collision_oracle.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from clevrer_benchmark.run_adapter_evaluation import load_adapter_model  # noqa: E402

# --- reproducibility: resolve data/checkpoint paths from env vars (see REPRODUCTION.md) ---
import os as _os_repro
from pathlib import Path as _Path_repro
_REPRO_ROOT = _Path_repro(__file__).resolve().parents[3]

DEFAULT_H5 = _os_repro.environ.get("CLEVRER_H5", str(_REPRO_ROOT / "data" / "clevrer_training_expanded.h5"))
DEFAULT_CKPT = _os_repro.environ.get("ADAPTER_CKPT", str(_REPRO_ROOT / "checkpoints" / "adapter_phase3.pt"))
T0, HOR, WARM, DT, THRESH = 64, 8, 3, 1.0 / 25.0, 0.9


def _any_collide(pos_seq, mask):
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
    ap.add_argument("--adapter_checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    with h5py.File(args.h5, "r") as f:
        n = min(args.n, f["states"].shape[0])
        states = np.asarray(f["states"][:n])
        masks = np.asarray(f["masks"][:n, 0, :])

    adapter = load_adapter_model(args.adapter_checkpoint, "", device=dev)
    pf = adapter.physics_model.eval()

    ball_c = pf_c = tot = 0
    pf_mse = ball_mse = 0.0
    with torch.no_grad():
        for s0 in range(0, n, args.batch):
            sl = slice(s0, min(s0 + args.batch, n))
            init = torch.from_numpy(states[sl, T0 - WARM + 1:T0 + 1].copy()).float().to(dev)
            mk = torch.from_numpy(masks[sl]).float().to(dev)
            try:
                traj, _ = pf.predict_autoregressive(init, mk, num_steps=HOR, num_warmup_frames=2)
                pf_pos = traj[..., 0:3].cpu().numpy()                 # [B,HOR,N,3]
            except Exception:
                import traceback; traceback.print_exc(); return
            for b in range(pf_pos.shape[0]):
                gi = s0 + b
                m = masks[gi]
                if (m > 0).sum() < 2:
                    continue
                actual = states[gi, T0 + 1:T0 + 1 + HOR, :, 0:3]
                y = _any_collide(actual, m)
                p0, v0 = states[gi, T0, :, 0:3], states[gi, T0, :, 3:6]
                ks = np.arange(1, HOR + 1)[:, None, None]
                ball = p0[None] + v0[None] * ks * DT
                tot += 1
                ball_c += int(_any_collide(ball, m) == y)
                pf_c += int(_any_collide(pf_pos[b], m) == y)
                vm = m > 0
                pf_mse += float(((pf_pos[b][:, vm] - actual[:, vm]) ** 2).mean())
                ball_mse += float(((ball[:, vm] - actual[:, vm]) ** 2).mean())

    print(f"\nPhysicsFormer's own prediction head as oracle (n={tot})")
    print(f"  position rollout MSE:  ballistic {ball_mse/tot:.4f}   PF {pf_mse/tot:.4f}")
    print(f"  collision-prediction:  ballistic {100*ball_c/tot:.1f}%   PF {100*pf_c/tot:.1f}%")
    print(f"  PF - ballistic (collision): {100*(pf_c-ball_c)/tot:+.1f} pp")


if __name__ == "__main__":
    main()
