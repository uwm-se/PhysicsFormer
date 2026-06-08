# NeSy-WM: an architecture for *dynamical* grounding

A design for grounding physics (not just the scene) in an LM-based reasoner.
Motivated by the findings in `FINDINGS_grounding_dynamics.md`: the grounded LM
reads static object identity but not dynamics; the LM is token-native (uses
dynamics as text, not as a continuous prefix); a from-scratch simulator (IPE)
uses dynamics and generalizes in-distribution but not across benchmarks, and
cannot yet learn hidden long-range forces (charge). NeSy-WM is engineered around
exactly those results, and grounded in current literature (sources at the end).

## Goal and the three requirements

Ground the LM's *dynamical* symbols ("collision", "velocity", "what happens
next") by binding them to a computation that actually does the dynamics, via an
interface the token-native LM can read.

1. an explicit forward simulator (world model) that evolves state;
2. a discrete/symbolic interface from simulator to LM;
3. an LM trained to *query* the interface rather than guess from priors.

## Architecture: three modules + an orchestration loop

### Module 1 — Object-centric world model (the engine)
The IPE's successor, upgraded with what the field now does and with the two
fixes our experiments demand (generalization, charge).

- **Object-slot representation** with **decoupled temporal + relational
  attention** (per-object motion vs. inter-object interaction) — the OCVP/SOLD
  design, which beats DreamerV3/TD-MPC2 on relational tasks. Generalizes the
  IPE's Interaction-Net.
- **Relative/relational-invariant features** (position deltas, pairwise relative
  position/velocity) so it learns *laws, not regimes* — the direct fix for the
  measured CLEVRER->ComPhy transfer failure.
- **Hidden-property inference as in-context system ID**: a context encoder reads
  an observation window and emits per-object latent codes (mass, charge), trained
  as in-context meta-learning. Replaces the IPE's fixed GRU.
- **Charge done right** (the open frontier): a *signed* latent + a
  **physics-informed pairwise force prior** (inverse-distance term as inductive
  bias) + a longer observation window so attraction/repulsion sign is
  identifiable.
- **Next-state rollout objective across DIVERSE simulators** (CLEVRER + ComPhy +
  ContPhy + Isaac), so it meta-learns invariant dynamics rather than one
  distribution's statistics.

### Module 2 — Discrete symbolic readout (the wall-crosser)
The rollout is parsed into discrete symbolic events the LM reads as tokens:
`collision(red_cube, blue_sphere)@t=0.32`, `enters(yellow)@t=0.1`,
`charge(A)=+1 [inferred, conf 0.8]`, binned positions. This is CRCG's
causal-event graph / the tokenizer pattern, and the empirical fix to our
encoding wall (text-injection gave +35 pp). The LM never sees continuous state.

### Module 3 — LM orchestrator (the controller)
The LM parses the question into a program/query over the world model (which
objects, what intervention, what to predict) and **decides when to simulate vs.
use perception** — CRCG's key result, which avoids rollout error accumulation by
using observed state where available and simulating only the affected /
counterfactual sub-part. Counterfactuals: the LM specifies the intervention,
the world model re-simulates from the divergence point and returns the new event
set. Trained via program-execution / tool-use traces (the LLMPhy pattern, LM
driving a physics engine).

### The loop
```
perceive state -> LM parses Q -> symbolic causal graph (what needs simulating)
      |                                  |
      +-- in-context system-ID <---------+  infer hidden props (mass/charge)
                 |
        world-model rollout (only affected objects, under intervention)
                 |
        discretize -> symbolic events / tokens
                 |
        LM reasons over events + Q -> answer
```

## How it clears each bar we hit

| Finding | Fix in NeSy-WM |
|---|---|
| LM token-native; continuous prefix unreadable | Module 2: only discrete events reach the LM |
| velocity unused by the LM | the world model does dynamics; LM consumes results |
| IPE didn't transfer cross-distribution | Module 1: meta-train over diverse sims + relative-invariant features |
| charge unlearnable | in-context system-ID + signed latent + force prior |
| rollout error accumulation | CRCG orchestration: perceive-where-possible, simulate-only-affected |

## Validation that would prove dynamical grounding

The ablations that exposed the problem become the success criteria:
1. **zeroing the world-model's dynamics output now tanks predictive accuracy**
   (dynamics load-bearing — inverse of our +0.2 pp);
2. **train on CLEVRER+others, test ComPhy predictive zero-shot** beats the
   language prior (transfer);
3. **charged-scene predictive** rises with the force-prior + system-ID.

## First concrete build (go/no-go)

Upgrade the IPE into **Module 1 only** and re-run the transfer test before
building the LM stack:
1. Replace the IPE's Interaction-Net with **slot + decoupled temporal/relational
   attention** (OCVP/SOLD-style).
2. Switch latent inference to **in-context system-ID** (context window ->
   per-object latent) and add the **signed-charge latent + inverse-distance force
   prior**.
3. Use **relative-invariant features** throughout.
4. **Meta-train on CLEVRER + ComPhy + ContPhy** jointly.
5. **Re-run `ipe_transfer.py`-style eval**: CLEVRER+others -> held-out ComPhy
   (neutral and charged) zero-shot.

Go/no-go: Module-1 v2 beats ballistic on **held-out ComPhy neutral zero-shot**
(the thing the current IPE fails, -1%) and shows a positive charged-scene gain.
If it transfers, the simulator is real; the LM stack (Modules 2-3) is then mostly
known engineering (CRCG/VRDP already work on CLEVRER).

## Honest risk

~70% engineering of de-risked parts (orchestration + discrete readout already
work on CLEVRER via CRCG/VRDP); ~30% real research, concentrated in Module 1's
**cross-distribution generalization** and **charge** — the two things our IPE
experiments showed are hard. Meta-learning + force-priors are promising but
unproven for those.

## Sources
- Think before You Simulate (CRCG, 2025): https://arxiv.org/html/2506.10753
- SOLD slot object-centric latent dynamics (2025): https://arxiv.org/html/2410.08822v2
- Object-Centric Video Prediction (OCVP): https://arxiv.org/pdf/2302.11850
- Graph Networks as learnable physics engines: https://arxiv.org/pdf/1806.01242
- Learning to Simulate Complex Physics with Graph Networks (GNS): http://proceedings.mlr.press/v80/sanchez-gonzalez18a/sanchez-gonzalez18a.pdf
- In-context learning of dynamical systems: https://arxiv.org/pdf/2410.03291
- LLMPhy (LLM + world model optimization): https://openreview.net/forum?id=qGL6fE1lqd
- ContPhy continuum benchmark: https://arxiv.org/pdf/2402.06119
- Cosmos / world models 2026: https://introl.com/blog/world-models-race-agi-2026
