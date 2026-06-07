"""NeSy-WM end-to-end: IPE simulator -> discrete readout -> LM that queries it.

Closes the loop the investigation pointed at. For a collision question on a
CLEVRER scene:
  Module 1: the CLEVRER-trained IPE rolls the scene forward.
  Module 2: its rollout is reduced to a DISCRETE fact for the queried pair,
            rendered as text ("A physics simulator predicts they WILL collide").
  Module 3: the LM (LoRA fine-tuned) reads question + fact and answers.

Ground truth is the ACTUAL future collision (from the real trajectory). We then
test whether the LM's answer is causally bound to the simulator by ablating the
readout:
  full        : prompt includes the IPE's discrete prediction.
  no-readout  : prompt has no simulator fact (= the original grounded LM, which
                must predict dynamics from the continuous prefix -> ~chance).
  scrambled   : the simulator fact is FLIPPED -> if the LM is grounded in the
                simulator, accuracy collapses / follows the wrong fact.

Success = full >> no-readout, and scrambled << full. That is dynamical grounding:
the LM's dynamical answer is bound to a computation that does the dynamics --
the inverse of the +0.2 pp the raw physics prefix produced.
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
from clevrer_benchmark.scene_converter import COLOR_MAP, SHAPE_MAP  # noqa: E402
from clevrer_benchmark.scripts.velocity_finetune import score_choices  # noqa: E402
from physics_llm_adapter.intuitive_physics_engine import IPE  # noqa: E402

# --- reproducibility: resolve data/checkpoint paths from env vars (see REPRODUCTION.md) ---
import os as _os_repro
from pathlib import Path as _Path_repro
_REPRO_ROOT = _Path_repro(__file__).resolve().parents[2]

DEFAULT_H5 = _os_repro.environ.get("CLEVRER_H5", str(_REPRO_ROOT / "data" / "clevrer_training_expanded.h5"))
DEFAULT_CKPT = _os_repro.environ.get("ADAPTER_CKPT", str(_REPRO_ROOT / "checkpoints" / "adapter_phase3.pt"))
DEFAULT_IPE = "checkpoints/ipe_clevrer.pt"
T0, OBS, HOR, THRESH = 64, 8, 8, 0.9
_COLORS = list(COLOR_MAP.items())
_SHAPES = {v: k for k, v in SHAPE_MAP.items()}


def _desc(o):
    c = min(_COLORS, key=lambda kv: float(np.sum((np.array(kv[1]) - o[15:18]) ** 2)))[0]
    return f"{c} {_SHAPES.get(int(round(float(o[18]))), 'object')}"


def _collide(pos_seq, oi, oj, d0):
    if d0 < THRESH:
        return None
    d = np.linalg.norm(pos_seq[:, oi] - pos_seq[:, oj], axis=-1)
    return bool(d.min() < THRESH)


def gen(states_np, masks_np, ipe, i, rng, dev):
    """Return (di, dj, gold_collide, ipe_collide) for a random valid pair, or None."""
    frame = states_np[i, T0]
    idx = np.where(masks_np[i] > 0)[0]
    if len(idx) < 2:
        return None
    a, b = rng.choice(idx, size=2, replace=False)
    oi, oj = frame[a], frame[b]
    di, dj = _desc(oi), _desc(oj)
    if di == dj:
        return None
    d0 = float(np.linalg.norm(oi[0:3] - oj[0:3]))
    gold = _collide(states_np[i, T0 + 1:T0 + 1 + HOR, :, 0:3], a, b, d0)   # actual future
    if gold is None:
        return None
    s = torch.from_numpy(states_np[i:i + 1]).float().to(dev)
    m = torch.from_numpy(masks_np[i:i + 1]).float().to(dev)
    with torch.no_grad():
        pred, _ = ipe(s, m, T0, OBS, HOR)
    ipe_pos = pred[0, :, :, 0:3].cpu().numpy()
    ipe_c = _collide(ipe_pos, a, b, d0)
    if ipe_c is None:
        ipe_c = False
    return di, dj, int(gold), int(ipe_c)


def prompt(di, dj, fact):                 # fact in {"will","will not", None}
    pre = "" if fact is None else f"A physics simulator predicts the {di} and the {dj} {fact} collide. "
    return f"{pre}Will the {di} and the {dj} collide within the next second?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--adapter_checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--ipe", default=DEFAULT_IPE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool", type=int, default=9000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    with h5py.File(args.h5, "r") as f:
        pool = min(args.pool, f["states"].shape[0])
        states_np = np.asarray(f["states"][:pool])
        masks_np = np.asarray(f["masks"][:pool, 0, :])
    split = int(0.85 * pool)
    tr_idx, ev_idx = np.arange(split), np.arange(split, min(split + 700, pool))

    adapter = load_adapter_model(args.adapter_checkpoint, "", device=dev)
    for n, p in adapter.named_parameters():
        p.requires_grad = "lora" in n.lower()
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    ipe = IPE().to(dev)
    ipe.load_state_dict(torch.load(args.ipe, map_location=dev))
    ipe.eval()

    def st(i):
        s = torch.from_numpy(states_np[i, T0:T0 + 1].copy()).float().unsqueeze(0).to(dev)[0]
        m = torch.from_numpy(masks_np[i]).float().unsqueeze(0).to(dev)
        return s.unsqueeze(0), m

    def evaluate(tag):
        cond = {"full": 0, "no-readout": 0, "scrambled": 0}
        tot = 0
        for i in ev_idx:
            ex = gen(states_np, masks_np, ipe, i, np.random.default_rng(int(i)), dev)
            if ex is None:
                continue
            di, dj, gold, ipe_c = ex
            s, m = st(i)
            for name, fact in [("full", "will" if ipe_c else "will not"),
                               ("no-readout", None),
                               ("scrambled", "will not" if ipe_c else "will")]:
                q = prompt(di, dj, fact)
                pr = int(score_choices(adapter, s, m, q, ["yes", "no"], dev).argmax())
                cond[name] += int((pr == 0) == bool(gold))
            tot += 1
        print(f"[{tag}] n={tot}  " + "  ".join(
            f"{k}={100*v/max(tot,1):.1f}%" for k, v in cond.items()))

    print("IPE collision accuracy vs actual (the oracle ceiling):")
    # quick oracle check on eval set
    oc = ot = 0
    for i in ev_idx:
        ex = gen(states_np, masks_np, ipe, i, np.random.default_rng(int(i)), dev)
        if ex is None:
            continue
        _, _, gold, ipe_c = ex
        oc += int(ipe_c == gold); ot += 1
    print(f"  IPE vs actual: {100*oc/max(ot,1):.1f}%  (n={ot})")

    evaluate("BEFORE")
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    rng = np.random.default_rng(0)
    step = 0; opt.zero_grad(); losses = []
    while step < args.steps:
        i = int(rng.choice(tr_idx))
        ex = gen(states_np, masks_np, ipe, i, rng, dev)
        if ex is None:
            continue
        di, dj, gold, ipe_c = ex
        s, m = st(i)
        q = prompt(di, dj, "will" if ipe_c else "will not")        # train WITH readout
        sc = score_choices(adapter, s, m, q, ["yes", "no"], dev, grad=True)
        loss = F.cross_entropy(sc.unsqueeze(0), torch.tensor([0 if gold else 1], device=dev))
        (loss / args.accum).backward(); losses.append(loss.item()); step += 1
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt.zero_grad()
        if step % 400 == 0:
            print(f"  step {step}  CE {np.mean(losses[-400:]):.3f}")
    evaluate("AFTER")


if __name__ == "__main__":
    main()
