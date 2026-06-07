"""Option 2: does the LM use velocity when it arrives as TEXT instead of a
continuous prefix? Localizes the wall to encoding vs. reasoning.

Same "which is moving faster" task, pooled prefix unchanged, but the two objects'
speeds are injected into the prompt as text. We fine-tune LoRA and evaluate with
the speed text present vs. removed:
  acc(text) high, acc(no-text) ~ chance  -> the LM CAN use velocity as text;
      the continuous-prefix encoding was the wall (not reasoning).
  acc(text) also ~ chance                -> even textual velocity isn't usable
      (a reasoning / model-capacity limit, not an encoding one).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from clevrer_benchmark.run_adapter_evaluation import load_adapter_model  # noqa: E402
from clevrer_benchmark.scripts.velocity_finetune import (  # noqa: E402
    _desc, T0, score_choices, DEFAULT_H5, DEFAULT_CKPT)


def gen(state, mask, rng):
    frame = state[T0]
    idx = np.where(mask > 0)[0]
    if len(idx) < 2:
        return None
    i, j = rng.choice(idx, size=2, replace=False)
    oi, oj = frame[i], frame[j]
    di, dj = _desc(oi), _desc(oj)
    if di == dj:
        return None
    si, sj = float(np.linalg.norm(oi[3:6])), float(np.linalg.norm(oj[3:6]))
    if abs(si - sj) < 0.15:
        return None
    speeds = f"Speeds -- {di}: {si:.1f}, {dj}: {sj:.1f}. "
    q = f"Which is moving faster, the {di} or the {dj}?"
    return speeds, q, [di, dj], (0 if si > sj else 1)


def evaluate(adapter, states_np, masks_np, idx, dev, tag):
    """acc with speed-text vs without (text removed)."""
    wt = nt = tot = 0
    for i in idx:
        ex = gen(states_np[i], masks_np[i], np.random.default_rng(int(i)))
        if ex is None:
            continue
        speeds, q, ch, gold = ex
        s = torch.from_numpy(states_np[i, T0:T0 + 1].copy()).float().to(dev)
        m = torch.from_numpy(masks_np[i]).float().unsqueeze(0).to(dev)
        wt += int(score_choices(adapter, s, m, speeds + q, ch, dev).argmax() == gold)
        nt += int(score_choices(adapter, s, m, q, ch, dev).argmax() == gold)
        tot += 1
    print(f"[{tag}] n={tot}  acc(speed-text)={100*wt/max(tot,1):.1f}%  "
          f"acc(no-text)={100*nt/max(tot,1):.1f}%  "
          f"text-velocity-contribution={100*(wt-nt)/max(tot,1):+.1f} pp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--adapter_checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool", type=int, default=12000)
    ap.add_argument("--steps", type=int, default=2800)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    with h5py.File(args.h5, "r") as f:
        pool = min(args.pool, f["states"].shape[0])
        states_np = np.asarray(f["states"][:pool])
        masks_np = np.asarray(f["masks"][:pool, 0, :])
    split = int(0.85 * pool)
    train_idx, eval_idx = np.arange(split), np.arange(split, min(split + 800, pool))

    adapter = load_adapter_model(args.adapter_checkpoint, "", device=dev)
    for n, p in adapter.named_parameters():
        p.requires_grad = "lora" in n.lower()
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    print(f"trainable (LoRA) {sum(p.numel() for p in trainable):,}")

    adapter.eval()
    evaluate(adapter, states_np, masks_np, eval_idx, dev, "BEFORE")
    if args.steps <= 0:
        return

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    rng = np.random.default_rng(0)
    step = 0; opt.zero_grad(); losses = []
    while step < args.steps:
        i = int(rng.choice(train_idx))
        ex = gen(states_np[i], masks_np[i], rng)
        if ex is None:
            continue
        speeds, q, ch, gold = ex
        s = torch.from_numpy(states_np[i, T0:T0 + 1].copy()).float().to(dev)
        m = torch.from_numpy(masks_np[i]).float().unsqueeze(0).to(dev)
        sc = score_choices(adapter, s, m, speeds + q, ch, dev, grad=True)
        loss = F.cross_entropy(sc.unsqueeze(0), torch.tensor([gold], device=dev))
        (loss / args.accum).backward(); losses.append(loss.item()); step += 1
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt.zero_grad()
        if step % 400 == 0:
            print(f"  step {step}  CE {np.mean(losses[-400:]):.3f}")

    adapter.eval()
    evaluate(adapter, states_np, masks_np, eval_idx, dev, "AFTER")


if __name__ == "__main__":
    main()
