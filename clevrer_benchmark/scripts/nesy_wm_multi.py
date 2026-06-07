"""NeSy-WM, multi-question-type: prove the architecture is a general predictive-QA
engine grounded in a generalizing world model -- not a collision trick.

One IPEv2 rollout per held-out ComPhy scene feeds THREE question types, each with
ground truth from the ACTUAL future trajectory:
  collision   : will {A} and {B} collide?            (relational)
  is-moving   : will {A} be moving at the end?       (per-object dynamics)
  which-faster: which moves faster, {A} or {B}?      (comparison)

Module 2 emits the world model's predicted answer as a discrete fact; Module 3
(LoRA) answers using it. We ablate the readout (full / no-readout) per type to
show the LM's dynamical answers are grounded in the simulator across all of them.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from clevrer_benchmark.run_adapter_evaluation import load_adapter_model  # noqa: E402
from clevrer_benchmark.scripts.ipe_v2_transfer import IPEv2  # noqa: E402
from clevrer_benchmark.scripts.velocity_finetune import score_choices  # noqa: E402
from physics_llm_adapter.intuitive_physics_engine import rollout_loss  # noqa: E402
from clevrer_benchmark.scripts.nesy_wm import _desc  # noqa: E402

# --- reproducibility: resolve data/checkpoint paths from env vars (see REPRODUCTION.md) ---
import os as _os_repro
from pathlib import Path as _Path_repro
_REPRO_ROOT = _Path_repro(__file__).resolve().parents[2]

CLEVRER_H5 = _os_repro.environ.get("CLEVRER_H5", str(_REPRO_ROOT / "data" / "clevrer_training_expanded.h5"))
DEFAULT_CKPT = _os_repro.environ.get("ADAPTER_CKPT", str(_REPRO_ROOT / "checkpoints" / "adapter_phase3.pt"))
COMPHY_CACHE = "comphy_benchmark/results/ipe_comphy_cache.npz"
T0P, HOR, CTHRESH, MTHRESH = 45, 8, 0.9, 0.15


def first_pair(frame, mk):
    idx = np.where(mk > 0)[0]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if _desc(frame[i]) != _desc(frame[j]):
                return i, j
    return None


def collide(pos, i, j, d0):
    if d0 < CTHRESH:
        return None
    return int(np.linalg.norm(pos[:, i] - pos[:, j], axis=-1).min() < CTHRESH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ipe_steps", type=int, default=1600)
    ap.add_argument("--lm_steps", type=int, default=1400)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    with h5py.File(CLEVRER_H5, "r") as f:
        Sc = np.asarray(f["states"][:14000]); Mc = np.asarray(f["masks"][:14000, 0, :])
    t0c = Sc.shape[1] // 2
    Sc_t, Mc_t = torch.from_numpy(Sc).float(), torch.from_numpy(Mc).float()
    d = np.load(COMPHY_CACHE); S, M = d["states"], d["masks"]
    perm = np.random.default_rng(0).permutation(len(S)); S, M = S[perm], M[perm]
    nqa = int(0.8 * len(S))
    Sp_t, Mp_t = torch.from_numpy(S).float(), torch.from_numpy(M[:, T0P, :]).float()

    ipe = IPEv2().to(dev); opt = torch.optim.Adam(ipe.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(0)
    print("Module 1: meta-training IPEv2 (CLEVRER+ComPhy)...")
    ipe.train()
    for step in range(1, args.ipe_steps + 1):
        if step % 2 == 0:
            bi = torch.randint(0, Sc.shape[0], (96,), generator=g); s, m, t0 = Sc_t[bi].to(dev), Mc_t[bi].to(dev), t0c
        else:
            bi = torch.randint(0, nqa, (96,), generator=g); s, m, t0 = Sp_t[bi].to(dev), Mp_t[bi].to(dev), T0P
        pred, _ = ipe(s, m, t0, 8, 8); loss = rollout_loss(pred, s, m, t0, 8)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ipe.parameters(), 1.0); opt.step()
    ipe.eval()

    # Module 2: one rollout per scene -> facts (predicted) + gold (actual)
    print("Module 2: building multi-event readouts on held-out ComPhy...")
    examples = []                       # (scene, qtype, di, dj, gold, pred, fact_text)
    with torch.no_grad():
        for s0 in range(0, len(S), 256):
            sl = slice(s0, min(s0 + 256, len(S)))
            pred, _ = ipe(torch.from_numpy(S[sl]).float().to(dev),
                          torch.from_numpy(M[sl, T0P, :]).float().to(dev), T0P, 8, HOR)
            ppos = pred[:, :, :, 0:3].cpu().numpy(); pvel = pred[:, :, :, 3:6].cpu().numpy()
            for b in range(ppos.shape[0]):
                gi = s0 + b; frame = S[gi, T0P]; mk = M[gi, T0P, :]
                pr = first_pair(frame, mk)
                if pr is None:
                    continue
                i, j = pr; di, dj = _desc(frame[i]), _desc(frame[j])
                fut = S[gi, T0P + 1:T0P + 1 + HOR, :, 0:3]
                fvel = S[gi, T0P + 1:T0P + 1 + HOR, :, 3:6]
                d0 = float(np.linalg.norm(frame[i][0:3] - frame[j][0:3]))
                # collision
                gc, pc = collide(fut, i, j, d0), collide(ppos[b], i, j, d0)
                if gc is not None and pc is not None:
                    f = f"A physics simulator predicts the {di} and the {dj} {'will' if pc else 'will not'} collide. "
                    examples.append((gi, "collision", di, dj, gc, pc,
                                     f + f"Will the {di} and the {dj} collide?"))
                # is-moving (object i)
                g_mv = int(np.linalg.norm(fvel[-1, i]) > MTHRESH)
                p_mv = int(np.linalg.norm(pvel[b, -1, i]) > MTHRESH)
                f = f"A physics simulator predicts the {di} will be {'moving' if p_mv else 'stationary'}. "
                examples.append((gi, "is-moving", di, dj, g_mv, p_mv,
                                 f + f"Will the {di} be moving at the end?"))
                # which-faster (i vs j)
                g_f = int(np.linalg.norm(fvel[:, i], axis=-1).mean() > np.linalg.norm(fvel[:, j], axis=-1).mean())
                p_f = int(np.linalg.norm(pvel[b, :, i], axis=-1).mean() > np.linalg.norm(pvel[b, :, j], axis=-1).mean())
                f = f"A physics simulator predicts the {di if p_f else dj} will move faster. "
                examples.append((gi, "which-faster", di, dj, g_f, p_f,
                                 f + f"Will the {di} move faster than the {dj}?"))
    tr = [e for e in examples if e[0] < nqa]; ev = [e for e in examples if e[0] >= nqa]
    print(f"  examples: train={len(tr)} eval={len(ev)}")

    adapter = load_adapter_model(DEFAULT_CKPT, "", device=dev)
    for n, p in adapter.named_parameters():
        p.requires_grad = "lora" in n.lower()
    trainable = [p for p in adapter.parameters() if p.requires_grad]

    def st(gi):
        return (torch.from_numpy(S[gi, T0P:T0P + 1].copy()).float().unsqueeze(0).to(dev),
                torch.from_numpy(M[gi, T0P, :]).float().unsqueeze(0).to(dev))

    def q_noreadout(text):              # strip the leading "A physics simulator ... . "
        return text.split(". ", 1)[1] if ". " in text else text

    def evaluate(tag):
        full = defaultdict(lambda: [0, 0]); noro = defaultdict(lambda: [0, 0])
        for gi, qt, di, dj, gold, pred, text in ev:
            s, m = st(gi)
            pf = int(score_choices(adapter, s, m, text, ["yes", "no"], dev).argmax())
            pn = int(score_choices(adapter, s, m, q_noreadout(text), ["yes", "no"], dev).argmax())
            full[qt][0] += int((pf == 0) == bool(gold)); full[qt][1] += 1
            noro[qt][0] += int((pn == 0) == bool(gold)); noro[qt][1] += 1
        print(f"[{tag}]  " + "   ".join(
            f"{qt}: full {100*full[qt][0]/max(full[qt][1],1):.0f}% / no-readout {100*noro[qt][0]/max(noro[qt][1],1):.0f}%"
            for qt in ["collision", "is-moving", "which-faster"]))

    evaluate("BEFORE")
    opt2 = torch.optim.AdamW(trainable, lr=3e-4); rng = np.random.default_rng(0)
    step = 0; opt2.zero_grad(); losses = []
    while step < args.lm_steps and tr:
        gi, qt, di, dj, gold, pred, text = tr[int(rng.integers(len(tr)))]
        s, m = st(gi)
        sc = score_choices(adapter, s, m, text, ["yes", "no"], dev, grad=True)
        loss = F.cross_entropy(sc.unsqueeze(0), torch.tensor([0 if gold else 1], device=dev))
        (loss / 4).backward(); losses.append(loss.item()); step += 1
        if step % 4 == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt2.step(); opt2.zero_grad()
        if step % 500 == 0:
            print(f"  LM step {step}  CE {np.mean(losses[-500:]):.3f}")
    evaluate("AFTER")


if __name__ == "__main__":
    main()
