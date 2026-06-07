"""Cheap data-only check: are ComPhy converted states in CLEVRER's trained range?

The Phase 3 encoder was trained on CLEVRER converter output. If ComPhy states
land outside the per-channel distribution the encoder saw, the physics prefix
is OOD-garbage -- which would explain an active-but-non-discriminative (even
counterproductive) prefix on ComPhy, independent of any frame-count issue.

This script needs ONLY ComPhy data. It converts a sample of ComPhy scenes and
reports per-channel stats next to the CLEVRER reference (known from the CLEVRER
converter constants), flagging channels that are out-of-distribution. The
velocity-units test is self-contained: CLEVRER velocity is a per-frame position
delta, so we compare ComPhy's annotated velocity magnitude against ComPhy's own
per-frame position-delta magnitude. A large ratio == different units == OOD.

    python comphy_benchmark/scripts/check_state_distribution.py --n 40
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from comphy_benchmark.scene_converter import comphy_scene_to_state_tensor  # noqa: E402

DEFAULT_COMPHY_DIR = os.environ.get("COMPHY_DIR", r"D:\comphy")

# CLEVRER reference per-channel (from clevrer_benchmark/scene_converter.py
# constants). None == data-dependent (no fixed reference).
CLEVRER_REF = {
    "pos[0:3]": "data (CLEVRER coord scale)",
    "vel[3:6]": "per-frame position delta",
    "quat[6:10]": "IDENTITY [0,0,0,1] (constant)",
    "angvel[10:13]": "0 (constant)",
    "mass[13]": "{1.0 rubber, 2.0 metal}",
    "radius[14]": "0.3 (constant)",
    "friction[20]": "{0.4, 0.8}",
    "restitution[34]": "{0.3, 0.9}",
}


def _annotation_files(comphy_dir: Path, n: int):
    files = sorted(glob.glob(str(comphy_dir / "target_annotation" / "*" / "*.json")))
    return files[:n]


def _valid_rows(states, masks):
    """Return [K, 35] of object-frame state rows where mask > 0."""
    m = masks > 0
    return states[m]  # [K, 35]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comphy_dir", default=DEFAULT_COMPHY_DIR)
    ap.add_argument("--n", type=int, default=40, help="Number of scenes to sample.")
    ap.add_argument("--align", action="store_true",
                    help="Convert with clevrer_align=True (verify the fix).")
    args = ap.parse_args()

    comphy_dir = Path(args.comphy_dir)
    files = _annotation_files(comphy_dir, args.n)
    if not files:
        print(f"No annotation JSONs under {comphy_dir}\\target_annotation")
        return

    rows = []
    vel_mag = []          # ||annotated velocity|| per valid object-frame
    posdelta_mag = []     # ||pos[t]-pos[t-1]|| per valid object-frame (CLEVRER convention)
    masses = set()
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                ann = json.load(f)
            states, masks, meta = comphy_scene_to_state_tensor(
                ann, clevrer_align=args.align)
        except Exception as e:
            print(f"[skip] {os.path.basename(fp)}: {e}")
            continue
        rows.append(_valid_rows(states, masks))
        for o in meta.get("objects", []):
            if o.get("comphy_mass") is not None:
                masses.add(float(o["comphy_mass"]))
        # velocity vs position-delta (per object, consecutive valid frames)
        T, N, _ = states.shape
        for oi in range(N):
            valid_t = np.where(masks[:, oi] > 0)[0]
            for k in range(1, len(valid_t)):
                t0, t1 = valid_t[k - 1], valid_t[k]
                if t1 - t0 != 1:
                    continue
                vel_mag.append(float(np.linalg.norm(states[t1, oi, 3:6])))
                posdelta_mag.append(
                    float(np.linalg.norm(states[t1, oi, 0:3] - states[t0, oi, 0:3])))

    if not rows:
        print("No convertible scenes.")
        return
    allrows = np.concatenate(rows, axis=0)  # [K, 35]

    def chan(lo, hi=None):
        sl = allrows[:, lo:(hi if hi else lo + 1)]
        return float(sl.min()), float(sl.mean()), float(sl.max()), float(sl.std())

    print(f"\nComPhy converted-state distribution  ({len(files)} scenes, "
          f"{allrows.shape[0]} valid object-frames)")
    print("=" * 78)
    print(f"{'channel':<16}{'min':>9}{'mean':>9}{'max':>9}{'std':>9}   CLEVRER reference")
    print("-" * 78)
    table = [
        ("pos[0:3]", 0, 3), ("vel[3:6]", 3, 6), ("quat[6:10]", 6, 10),
        ("angvel[10:13]", 10, 13), ("mass[13]", 13, 14), ("radius[14]", 14, 15),
        ("friction[20]", 20, 21), ("restitution[34]", 34, 35),
    ]
    for name, lo, hi in table:
        mn, me, mx, sd = chan(lo, hi)
        print(f"{name:<16}{mn:>9.3f}{me:>9.3f}{mx:>9.3f}{sd:>9.3f}   {CLEVRER_REF.get(name, '')}")

    print("-" * 78)
    print(f"ComPhy mass values seen: {sorted(masses)}   "
          f"(CLEVRER trained on {{1.0, 2.0}})")

    if vel_mag:
        vel_mag = np.array(vel_mag)
        posdelta_mag = np.array(posdelta_mag)
        nz = posdelta_mag > 1e-6
        ratio = vel_mag[nz] / posdelta_mag[nz]
        print("\nVelocity-units test (ComPhy annotated vel vs ComPhy per-frame pos delta)")
        print("-" * 78)
        print(f"  mean ||annotated velocity||:  {vel_mag.mean():.4f}")
        print(f"  mean ||pos[t]-pos[t-1]||:     {posdelta_mag.mean():.4f}  "
              f"(== CLEVRER's velocity convention)")
        print(f"  median ratio vel/posdelta:    {np.median(ratio):.2f}")
        # NOTE: CLEVRER's derive_velocity = per-frame delta * fps, so its
        # training velocity is ~fps x the bare position delta. ComPhy's
        # annotated velocity is already per-second, so a ratio ~= fps (~25) is
        # EXPECTED and CORRECT -- it means ComPhy velocity matches CLEVRER's
        # training scale. (Earlier this was misread as an OOD bug; the training
        # h5 shows CLEVRER vel std 0.87 vs ComPhy 0.72 -- a match.)
        med = np.median(ratio)
        verdict = (f"ratio ~{med:.0f}x position-delta == ~fps: matches CLEVRER's "
                   f"derive_velocity (per-second) scale -> IN-RANGE"
                   if 10 < med < 40 else
                   f"ratio {med:.1f}x: unexpected -- check against training h5")
        print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
