"""One-off: per-channel stats of the CLEVRER TRAINING states (the exact tensors
the Phase 3 encoder saw), to close the position-scale caveat in the ComPhy OOD
analysis. Reads the training h5 and prints masked per-channel ranges.
"""
import sys
import h5py
import numpy as np

H5 = (r"C:\Users\jpoko\source\repos\homework\CascadeProjects\windsurf-project"
      r"\compsac_2026_code\data\clevrer_training_expanded.h5")

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
with h5py.File(H5, "r") as f:
    S = f["states"][:n]   # [n, T, Nobj, 35]
    M = f["masks"][:n]    # [n, T, Nobj]
rows = S[M > 0]           # [K, 35]
print(f"CLEVRER TRAINING states ({n} samples, {rows.shape[0]} valid obj-frames)")
print(f"{'channel':<16}{'min':>9}{'mean':>9}{'max':>9}{'std':>9}")
print("-" * 52)
for name, lo, hi in [("pos[0:3]", 0, 3), ("vel[3:6]", 3, 6), ("quat[6:10]", 6, 10),
                     ("angvel[10:13]", 10, 13), ("mass[13]", 13, 14),
                     ("radius[14]", 14, 15), ("friction[20]", 20, 21),
                     ("restitution[34]", 34, 35)]:
    sl = rows[:, lo:hi]
    print(f"{name:<16}{float(sl.min()):>9.3f}{float(sl.mean()):>9.3f}"
          f"{float(sl.max()):>9.3f}{float(sl.std()):>9.3f}")
