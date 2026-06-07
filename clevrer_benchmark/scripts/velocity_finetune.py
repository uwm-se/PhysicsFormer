"""Fix the LM's velocity extraction by training pressure (Stages 0/1/2/4).

The velocity-critical probe showed the frozen prefix carries usable velocity
(collision decodable at 78.7%); the QA-trained LM just never learned to read it.
This script tests the fix: generate velocity-critical questions whose answers
are determined by motion (Stage 0), fine-tune the adapter's LoRA + prefix MLP on
them with the frozen encoder (Stage 1; optional aux head = Stage 2), and measure
the velocity ablation before vs after (Stage 4). Success = zeroing velocity,
which barely moves the original QA model (+0.2 pp), now causes a large accuracy
drop -- i.e. the LM has learned to use the velocity already in the prefix.

  python clevrer_benchmark/scripts/velocity_finetune.py --steps 0      # BEFORE only
  python clevrer_benchmark/scripts/velocity_finetune.py --steps 1500   # train + AFTER
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
from clevrer_benchmark.scene_converter import COLOR_MAP, SHAPE_MAP        # noqa: E402

# --- reproducibility: resolve data/checkpoint paths from env vars (see REPRODUCTION.md) ---
import os as _os_repro
from pathlib import Path as _Path_repro
_REPRO_ROOT = _Path_repro(__file__).resolve().parents[2]

DEFAULT_H5 = _os_repro.environ.get("CLEVRER_H5", str(_REPRO_ROOT / "data" / "clevrer_training_expanded.h5"))
DEFAULT_CKPT = _os_repro.environ.get("ADAPTER_CKPT", str(_REPRO_ROOT / "checkpoints" / "adapter_phase3.pt"))
DT, T0, K, THRESH = 1.0 / 25.0, 64, 25, 0.9
_COLORS = list(COLOR_MAP.items())
_SHAPES = {v: k for k, v in SHAPE_MAP.items()}


def _desc(o):
    c = min(_COLORS, key=lambda kv: float(np.sum((np.array(kv[1]) - o[15:18]) ** 2)))[0]
    return f"{c} {_SHAPES.get(int(round(float(o[18]))), 'object')}"


def _will_collide(pi, pj, vi, vj):
    d0 = float(np.linalg.norm(pi - pj))
    if d0 < THRESH:
        return None                                    # already touching
    mind = min(float(np.linalg.norm((pi + vi * k * DT) - (pj + vj * k * DT)))
               for k in range(1, K + 1))
    return mind < THRESH


def scene_mean_speed(state, mask):
    frame = state[T0]
    idx = np.where(mask > 0)[0]
    if len(idx) < 1:
        return None
    return float(np.mean([np.linalg.norm(frame[i][3:6]) for i in idx]))


def gen_example(state, mask, median):
    """Shortcut-free velocity task: is the scene's overall motion fast or slow?
    The label is a pure function of velocity magnitudes (median split), so it is
    position- and identity-independent -- the model cannot fake it without
    reading velocity. The probe confirms mean-speed is decodable from the prefix
    (R^2=0.49) and collapses to R^2=-0.16 when velocity is zeroed."""
    ms = scene_mean_speed(state, mask)
    if ms is None or abs(ms - median) < 0.04:          # drop borderline for clean labels
        return None
    q = "Is the overall motion in this scene fast or slow?"
    return q, ["fast", "slow"], (0 if ms > median else 1)


def scene_collide(frame, mask):
    """Scene-level will-collide (any pair, ballistic) -> aux label."""
    idx = np.where(mask > 0)[0]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if _will_collide(frame[i][0:3], frame[j][0:3],
                             frame[i][3:6], frame[j][3:6]):
                return 1
    return 0


def _prefix_of(adapter, state_t, mask_t, zero_vel=False):
    if zero_vel:
        state_t = state_t.clone()
        state_t[..., 3:6] = 0.0
        state_t[..., 10:13] = 0.0
    feats = adapter.extract_physics_features(state_t, mask_t)     # frozen encoder
    return adapter.create_prefix_tokens(feats)                   # prefix MLP (trainable)


def _score_given_prefix(adapter, prefix, q, choices, dev, grad=False):
    plen = adapter.num_prefix_tokens
    scores = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    for c in choices:
        tok = adapter.tokenizer([q + " Answer: " + c], return_tensors="pt",
                                truncation=True, max_length=64).to(dev)
        emb = adapter.llm.transformer.wte(tok.input_ids)
        comb = torch.cat([prefix.to(emb.dtype), emb], dim=1)
        amask = torch.cat([torch.ones(1, plen, device=dev), tok.attention_mask], 1)
        qlen = len(adapter.tokenizer.encode(q + " Answer:", add_special_tokens=False))
        labels = tok.input_ids.clone()
        labels[:, :qlen] = -100
        labels = torch.cat([torch.full((1, plen), -100, dtype=torch.long, device=dev),
                            labels], 1)
        with ctx:
            out = adapter.llm(inputs_embeds=comb, attention_mask=amask, labels=labels)
        scores.append(-out.loss)
    return torch.stack(scores)


def score_choices(adapter, state_t, mask_t, q, choices, dev, zero_vel=False, grad=False):
    prefix = _prefix_of(adapter, state_t, mask_t, zero_vel)
    return _score_given_prefix(adapter, prefix, q, choices, dev, grad)


def _state_tensor(states_np, masks_np, i, dev):
    s = torch.from_numpy(states_np[i, T0:T0 + 1].copy()).float().to(dev)  # [1,N,35]
    m = torch.from_numpy(masks_np[i]).float().unsqueeze(0).to(dev)        # [1,N]
    return s, m


def evaluate(adapter, states_np, masks_np, eval_idx, dev, tag, median):
    real_c = abl_c = tot = 0
    for i in eval_idx:
        ex = gen_example(states_np[i], masks_np[i], median)
        if ex is None:
            continue
        q, choices, gold = ex
        s, m = _state_tensor(states_np, masks_np, i, dev)
        pr = int(score_choices(adapter, s, m, q, choices, dev).argmax())
        pa = int(score_choices(adapter, s, m, q, choices, dev, zero_vel=True).argmax())
        real_c += int(pr == gold); abl_c += int(pa == gold); tot += 1
    ra, aa = 100 * real_c / max(tot, 1), 100 * abl_c / max(tot, 1)
    print(f"[{tag}] n={tot}  acc(real)={ra:.1f}%  acc(vel-zeroed)={aa:.1f}%  "
          f"velocity-contribution={ra-aa:+.1f} pp")
    return ra - aa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--adapter_checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool", type=int, default=12000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--aux_lambda", type=float, default=0.5)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    with h5py.File(args.h5, "r") as f:
        pool = min(args.pool, f["states"].shape[0])
        states_np = np.asarray(f["states"][:pool])
        masks_np = np.asarray(f["masks"][:pool, 0, :])
    split = int(0.85 * pool)
    train_idx = np.arange(0, split)
    eval_idx = np.arange(split, min(split + 800, pool))
    ms_all = [scene_mean_speed(states_np[i], masks_np[i]) for i in range(pool)]
    median = float(np.median([x for x in ms_all if x is not None]))
    print(f"pool={pool} train={len(train_idx)} eval={len(eval_idx)}  "
          f"mean-speed median={median:.3f}")

    adapter = load_adapter_model(args.adapter_checkpoint, "", device=dev)
    # Train LoRA + the prefix MLP (``adapter.*``): lets the representation
    # re-route velocity into LLM-readable prefix tokens, not just re-read a fixed
    # prefix. Encoder and base LLM stay frozen.
    for n, p in adapter.named_parameters():
        p.requires_grad = ("lora" in n.lower()) or n.startswith("adapter.")
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    # Stage 2: auxiliary will-collide head on the pooled prefix -> directly
    # pressures the prefix MLP to surface velocity-dependent dynamics, beyond
    # whatever the QA loss alone induces.
    aux_head = torch.nn.Sequential(
        torch.nn.Linear(adapter.llm_dim, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 2)).to(dev)
    trainable = trainable + list(aux_head.parameters())
    print(f"trainable params: {sum(p.numel() for p in trainable):,}  "
          f"aux_lambda={args.aux_lambda}")

    adapter.eval()
    evaluate(adapter, states_np, masks_np, eval_idx, dev, "BEFORE", median)
    if args.steps <= 0:
        return

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    rng = np.random.default_rng(0)
    step = 0
    opt.zero_grad()
    qa_l, aux_l = [], []
    while step < args.steps:
        i = int(rng.choice(train_idx))
        ex = gen_example(states_np[i], masks_np[i], median)
        if ex is None:
            continue
        q, choices, gold = ex
        s, m = _state_tensor(states_np, masks_np, i, dev)
        prefix = _prefix_of(adapter, s, m)                       # shared
        scores = _score_given_prefix(adapter, prefix, q, choices, dev, grad=True)
        gt = torch.tensor([gold], device=dev)
        qa = F.cross_entropy(scores.unsqueeze(0), gt)
        aux = F.cross_entropy(aux_head(prefix.mean(1)), gt)      # same task, direct pressure
        loss = qa + args.aux_lambda * aux
        (loss / args.accum).backward()
        qa_l.append(qa.item()); aux_l.append(aux.item())
        step += 1
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad()
        if step % 300 == 0:
            print(f"  step {step}  QA CE {np.mean(qa_l[-300:]):.3f}  "
                  f"aux CE {np.mean(aux_l[-300:]):.3f}")

    adapter.eval()
    evaluate(adapter, states_np, masks_np, eval_idx, dev, "AFTER", median)
    if args.save:
        torch.save({k: v for k, v in adapter.state_dict().items()
                    if any(t in k for t in ("lora", "adapter"))}, args.save)
        print(f"saved trainable deltas -> {args.save}")


if __name__ == "__main__":
    main()
