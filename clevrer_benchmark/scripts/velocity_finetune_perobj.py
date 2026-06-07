"""Option 1: does an OBJECT-CENTRIC prefix let the LM extract velocity, where the
pooled prefix could not?

The pooled-prefix fine-tunes never moved the velocity ablation off zero. Here we
replace the pooled prefix with ONE token per object (a fresh MLP on the frozen
encoder's per-object embeddings) and train on a per-object velocity task that
*requires* binding each object to its motion: "which of two objects is moving
faster?". Pooling destroys that binding; per-object tokens preserve it.

Frozen: encoder + base LLM. Trained: the new per-object prefix MLP + LoRA.
Success = high accuracy AND a large velocity-contribution (zeroing velocity
tanks it), vs. the pooled pathway's ~0 contribution.

    python clevrer_benchmark/scripts/velocity_finetune_perobj.py --steps 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from clevrer_benchmark.run_adapter_evaluation import load_adapter_model  # noqa: E402
from clevrer_benchmark.scripts.velocity_finetune import _desc, T0, DEFAULT_H5, DEFAULT_CKPT  # noqa: E402


def gen_faster(state, mask, rng):
    """Which of two distinct objects is moving faster (per-object, velocity-only)."""
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
        return None                                    # need a clear winner
    return f"Which is moving faster, the {di} or the {dj}?", [di, dj], (0 if si > sj else 1)


def score(adapter, obj_mlp, s, m, q, choices, dev, zero_vel=False, grad=False):
    if zero_vel:
        s = s.clone(); s[..., 3:6] = 0.0; s[..., 10:13] = 0.0
    obj_emb, flat_mask = adapter.extract_object_embeddings(s, m)   # [1,N,D], [1,N]
    prefix = obj_mlp(obj_emb) * flat_mask.unsqueeze(-1)            # [1,N,llm_dim]
    N = prefix.size(1)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    scores = []
    for c in choices:
        tok = adapter.tokenizer([q + " Answer: " + c], return_tensors="pt",
                                truncation=True, max_length=64).to(dev)
        emb = adapter.llm.transformer.wte(tok.input_ids)
        comb = torch.cat([prefix.to(emb.dtype), emb], dim=1)
        amask = torch.cat([flat_mask, tok.attention_mask], 1)
        qlen = len(adapter.tokenizer.encode(q + " Answer:", add_special_tokens=False))
        labels = tok.input_ids.clone(); labels[:, :qlen] = -100
        labels = torch.cat([torch.full((1, N), -100, dtype=torch.long, device=dev), labels], 1)
        with ctx:
            out = adapter.llm(inputs_embeds=comb, attention_mask=amask, labels=labels)
        scores.append(-out.loss)
    return torch.stack(scores)


def evaluate(adapter, obj_mlp, states_np, masks_np, idx, dev, tag):
    rc = ac = tot = 0
    for i in idx:
        ex = gen_faster(states_np[i], masks_np[i], np.random.default_rng(int(i)))
        if ex is None:
            continue
        q, ch, gold = ex
        s = torch.from_numpy(states_np[i, T0:T0 + 1].copy()).float().to(dev)
        m = torch.from_numpy(masks_np[i]).float().unsqueeze(0).to(dev)
        rc += int(score(adapter, obj_mlp, s, m, q, ch, dev).argmax() == gold)
        ac += int(score(adapter, obj_mlp, s, m, q, ch, dev, zero_vel=True).argmax() == gold)
        tot += 1
    ra, aa = 100 * rc / max(tot, 1), 100 * ac / max(tot, 1)
    print(f"[{tag}] n={tot}  acc(real)={ra:.1f}%  acc(vel-zeroed)={aa:.1f}%  "
          f"velocity-contribution={ra-aa:+.1f} pp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--adapter_checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool", type=int, default=12000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
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
        p.requires_grad = "lora" in n.lower()                     # base + encoder frozen
    # Build the per-object prefix MLP from the encoder's per-object embedding dim.
    with torch.no_grad():
        s0 = torch.from_numpy(states_np[0, T0:T0 + 1].copy()).float().to(dev)
        m0 = torch.from_numpy(masks_np[0]).float().unsqueeze(0).to(dev)
        D = adapter.extract_object_embeddings(s0, m0)[0].shape[-1]
    obj_mlp = nn.Sequential(nn.Linear(D, adapter.llm_dim), nn.SiLU(),
                            nn.Linear(adapter.llm_dim, adapter.llm_dim)).to(dev)
    trainable = [p for p in adapter.parameters() if p.requires_grad] + list(obj_mlp.parameters())
    print(f"per-object prefix: D={D}->{adapter.llm_dim}; "
          f"trainable {sum(p.numel() for p in trainable):,}")

    adapter.eval()
    evaluate(adapter, obj_mlp, states_np, masks_np, eval_idx, dev, "BEFORE (untrained per-obj)")
    if args.steps <= 0:
        return

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    rng = np.random.default_rng(0)
    step = 0; opt.zero_grad(); losses = []
    while step < args.steps:
        i = int(rng.choice(train_idx))
        ex = gen_faster(states_np[i], masks_np[i], rng)
        if ex is None:
            continue
        q, ch, gold = ex
        s = torch.from_numpy(states_np[i, T0:T0 + 1].copy()).float().to(dev)
        m = torch.from_numpy(masks_np[i]).float().unsqueeze(0).to(dev)
        sc = score(adapter, obj_mlp, s, m, q, ch, dev, grad=True)
        loss = F.cross_entropy(sc.unsqueeze(0), torch.tensor([gold], device=dev))
        (loss / args.accum).backward(); losses.append(loss.item()); step += 1
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt.zero_grad()
        if step % 400 == 0:
            print(f"  step {step}  CE {np.mean(losses[-400:]):.3f}")

    adapter.eval()
    evaluate(adapter, obj_mlp, states_np, masks_np, eval_idx, dev, "AFTER")


if __name__ == "__main__":
    main()
