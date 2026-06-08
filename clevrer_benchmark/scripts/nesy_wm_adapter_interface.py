"""(b) Deliver the PhysicsFormer rollout to the LM THROUGH THE ORIGINAL ADAPTER,
two ways, to test whether the INTERFACE (not the computation) is the wall even
for forward prediction. Preserves the PhysicsFormer -> adapter -> LM pipeline.

Module 1: PFRollout (PhysicsFormer attention, relative-invariant) meta-trained
          on CLEVRER + ComPhy.
Module 2: per ComPhy scene, roll forward -> predicted future state + discrete fact.
Module 3 (the comparison), same adapter + LM, LoRA-tuned per condition:
  cont-future : adapter prefix = predicted FUTURE state (continuous), plain question
  discrete    : adapter prefix = snapshot, question + discrete rollout fact (tokens)

Clean cases are which-faster / is-moving: the answer lives in the future velocity
magnitude, which the discrete fact states explicitly but the continuous prefix only
carries. If discrete >> cont-future, the interface is the wall even with a rollout,
and the adapter still works as the interface once its content is discretized.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from clevrer_benchmark.run_adapter_evaluation import load_adapter_model  # noqa: E402
from clevrer_benchmark.scripts.physicsformer_rollout import PFRollout  # noqa: E402
from clevrer_benchmark.scripts.velocity_finetune import score_choices  # noqa: E402
from physics_llm_adapter.intuitive_physics_engine import rollout_loss, POS, VEL  # noqa: E402
from clevrer_benchmark.scripts.nesy_wm import _desc  # noqa: E402
from clevrer_benchmark.scripts.nesy_wm_multi import first_pair  # noqa: E402

CLEVRER_H5 = os.environ.get("CLEVRER_H5",
    str(Path(__file__).resolve().parents[2] / "data" / "clevrer_training_expanded.h5"))
DEFAULT_CKPT = os.environ.get("ADAPTER_CKPT",
    str(Path(__file__).resolve().parents[2] / "checkpoints" / "adapter_phase3.pt"))
COMPHY_CACHE = "comphy_benchmark/results/ipe_comphy_cache.npz"
T0P, HOR, CTHRESH, MTHRESH = 45, 8, 0.9, 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ipe_steps", type=int, default=1600)
    ap.add_argument("--lm_steps", type=int, default=1200)
    ap.add_argument("--max_scenes", type=int, default=0)   # 0 = all
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    with h5py.File(CLEVRER_H5, "r") as f:
        Sc = np.asarray(f["states"][:14000]); Mc = np.asarray(f["masks"][:14000, 0, :])
    t0c = Sc.shape[1] // 2
    Sc_t, Mc_t = torch.from_numpy(Sc).float(), torch.from_numpy(Mc).float()
    d = np.load(COMPHY_CACHE); S, M = d["states"], d["masks"]
    perm = np.random.default_rng(0).permutation(len(S)); S, M = S[perm], M[perm]
    if args.max_scenes:
        S, M = S[:args.max_scenes], M[:args.max_scenes]
    nqa = int(0.8 * len(S))
    Sp_t, Mp_t = torch.from_numpy(S).float(), torch.from_numpy(M[:, T0P, :]).float()

    # ---- Module 1: meta-train PFRollout (PhysicsFormer attention) ----
    ipe = PFRollout().to(dev); opt = torch.optim.Adam(ipe.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(0)
    print("Module 1: meta-training PFRollout (CLEVRER+ComPhy)...", flush=True)
    ipe.train()
    for step in range(1, args.ipe_steps + 1):
        if step % 2 == 0:
            bi = torch.randint(0, Sc.shape[0], (96,), generator=g); s, m, t0 = Sc_t[bi].to(dev), Mc_t[bi].to(dev), t0c
        else:
            bi = torch.randint(0, nqa, (96,), generator=g); s, m, t0 = Sp_t[bi].to(dev), Mp_t[bi].to(dev), T0P
        pred, _ = ipe(s, m, t0, 8, 8); loss = rollout_loss(pred, s, m, t0, 8)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ipe.parameters(), 1.0); opt.step()
        if step % 400 == 0:
            print(f"  step {step}  rollout MSE {loss.item():.4f}", flush=True)
    ipe.eval()

    # ---- Module 2: per scene -> predicted future state (final frame) + facts ----
    print("Module 2: rolling forward on ComPhy...", flush=True)
    fut_pos = {}; fut_vel = {}; examples = []   # (gi,qtype,di,dj,gold,pred,fact_text,plain_text)
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
                fut_pos[gi] = ppos[b, -1]; fut_vel[gi] = pvel[b, -1]
                fvel = S[gi, T0P + 1:T0P + 1 + HOR, :, 3:6]
                # is-moving (object i)
                g_mv = int(np.linalg.norm(fvel[-1, i]) > MTHRESH)
                p_mv = int(np.linalg.norm(pvel[b, -1, i]) > MTHRESH)
                fact = f"A physics simulator predicts the {di} will be {'moving' if p_mv else 'stationary'}. "
                q = f"Will the {di} be moving at the end?"
                examples.append((gi, "is-moving", di, dj, g_mv, p_mv, fact + q, q))
                # which-faster (i vs j)
                g_f = int(np.linalg.norm(fvel[:, i], axis=-1).mean() > np.linalg.norm(fvel[:, j], axis=-1).mean())
                p_f = int(np.linalg.norm(pvel[b, :, i], axis=-1).mean() > np.linalg.norm(pvel[b, :, j], axis=-1).mean())
                fact = f"A physics simulator predicts the {di if p_f else dj} will move faster. "
                q = f"Will the {di} move faster than the {dj}?"
                examples.append((gi, "which-faster", di, dj, g_f, p_f, fact + q, q))
    tr = [e for e in examples if e[0] < nqa]; ev = [e for e in examples if e[0] >= nqa]
    print(f"  examples: train={len(tr)} eval={len(ev)}", flush=True)

    def st_snapshot(gi):
        return (torch.from_numpy(S[gi, T0P:T0P + 1].copy()).float().unsqueeze(0).to(dev),
                torch.from_numpy(M[gi, T0P, :]).float().unsqueeze(0).to(dev))

    def st_future(gi):
        s = S[gi, T0P].copy()
        s[..., POS] = fut_pos[gi]; s[..., VEL] = fut_vel[gi]
        return (torch.from_numpy(s).float().unsqueeze(0).unsqueeze(0).to(dev),
                torch.from_numpy(M[gi, T0P, :]).float().unsqueeze(0).to(dev))

    # ---- Module 3: train + eval each interface condition ----
    def run_condition(name, state_fn, text_idx):
        adapter = load_adapter_model(DEFAULT_CKPT, "", device=dev)
        for n, p in adapter.named_parameters():
            p.requires_grad = "lora" in n.lower()
        trainable = [p for p in adapter.parameters() if p.requires_grad]
        opt2 = torch.optim.AdamW(trainable, lr=3e-4); rng = np.random.default_rng(0)
        step = 0; opt2.zero_grad()
        while step < args.lm_steps and tr:
            e = tr[int(rng.integers(len(tr)))]
            s, m = state_fn(e[0])
            sc = score_choices(adapter, s, m, e[text_idx], ["yes", "no"], dev, grad=True)
            loss = F.cross_entropy(sc.unsqueeze(0), torch.tensor([0 if e[4] else 1], device=dev))
            (loss / 4).backward(); step += 1
            if step % 4 == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt2.step(); opt2.zero_grad()
        acc = defaultdict(lambda: [0, 0])
        with torch.no_grad():
            for e in ev:
                s, m = state_fn(e[0])
                pr = int(score_choices(adapter, s, m, e[text_idx], ["yes", "no"], dev).argmax())
                acc[e[1]][0] += int((pr == 0) == bool(e[4])); acc[e[1]][1] += 1
        del adapter; torch.cuda.empty_cache()
        line = "   ".join(f"{qt}: {100*acc[qt][0]/max(acc[qt][1],1):.0f}%" for qt in ["is-moving", "which-faster"])
        print(f"[{name}]  {line}", flush=True)
        return {qt: 100 * acc[qt][0] / max(acc[qt][1], 1) for qt in ["is-moving", "which-faster"]}

    print("\nModule 3: interface comparison (same rollout, same adapter)...", flush=True)
    cont = run_condition("cont-future (continuous prefix of predicted future)", st_future, 7)
    disc = run_condition("discrete   (snapshot prefix + discrete rollout fact)", st_snapshot, 6)

    print("\n========== INTERFACE GATE (held-out ComPhy) ==========", flush=True)
    for qt in ["is-moving", "which-faster"]:
        print(f"  {qt:<13}  cont-future {cont[qt]:.0f}%   vs   discrete {disc[qt]:.0f}%   "
              f"(+{disc[qt]-cont[qt]:.0f} pp)", flush=True)
    print("\n  Same rollout, same adapter: only the interface differs. A large discrete>cont gap"
          "\n  means the interface, not the computation, is the wall even for forward prediction.", flush=True)


if __name__ == "__main__":
    main()
