"""Inspect the CLEVRER training h5: question/answer/type structure, to see if
the contrastive delta diagnostic (needs MC choices) can run on it."""
import h5py
import numpy as np
from collections import Counter

H5 = (r"C:\Users\jpoko\source\repos\homework\CascadeProjects\windsurf-project"
      r"\compsac_2026_code\data\clevrer_training_expanded.h5")

with h5py.File(H5, "r") as f:
    qt = f["question_types"][:5000]
    qs = f["questions"][:20]
    ans = f["answers"][:20]
    meta = f["metadata"][:3]

    def dec(x):
        return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)

    print("question_types distribution (first 5000):")
    for k, v in Counter(dec(x) for x in qt).most_common():
        print(f"  {k:<24} {v}")
    print("\nsample questions / answers:")
    for i in range(12):
        print(f"  [{dec(f['question_types'][i])}] Q: {dec(qs[i])[:90]}")
        print(f"      A: {dec(ans[i])[:90]}")
    print("\nsample metadata[0]:")
    print(" ", dec(meta[0])[:400])
