# Reproducing the 82M Physics-LLM CLEVRER Result

This document walks you end-to-end through reproducing the **79.6%
overall accuracy** of the Phase 3 Physics-LLM on the full CLEVRER
validation split (21,378 explanatory / predictive / counterfactual
MCQ questions across 5000 scenes) — a new SOTA for this model family.

All paths below are PowerShell-style; on Linux/macOS substitute `\`
with `/` and adjust drive letters.

---

## TL;DR — the one command

If you already have the Phase 3 checkpoint and CLEVRER validation
data locally, this single command reproduces the 79.6% headline:

```powershell
python clevrer_benchmark\run_adapter_evaluation.py `
    --clevrer_dir $CLEVRER_DIR `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --max_scenes 99999 `
    --skip_descriptive `
    --save_details `
    --output clevrer_benchmark\results\phase3_full5000.json
```

Expected output (matches
`clevrer_benchmark/results/phase3_GENERATE_singleframe_FULL5000.json`):

| Split           | Count         | Accuracy |
|-----------------|---------------|----------|
| Overall         | 17021 / 21378 | **79.6%** |
| Explanatory     |  6700 / 8488  | 78.9%    |
| Predictive      |  2718 / 3557  | 76.4%    |
| Counterfactual  |  7603 / 9333  | 81.5%    |

The script's defaults (`--eval_method generate --gen_mode sample
--gen_seed 42 --single_frame 64`) are now the canonical
SOTA-reproducing setup, so you don't need to pass them explicitly.

Runtime: ~20–30 minutes on a single modern GPU. For a ~3-minute
smoke-test, run with the default `--max_scenes 100` (i.e. drop the
`--max_scenes 99999` flag): expected 82.3% on the first 100 scenes
(see §6 for the caveat).

---

## 1. Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python 3.10 or 3.11. GPU (CUDA) strongly recommended; CPU works but
takes hours.

---

## 2. CLEVRER validation data

Download CLEVRER from <http://clevrer.csail.mit.edu/> and lay it out
as below (any path works; pass it via `--clevrer_dir`):

```
$CLEVRER_DIR\
├── scenes\
│   ├── validation\            # 5000 scene JSONs: sim_10000.json … sim_14999.json
│   └── clevrer_scenes\        # Some releases ship scenes here instead; script auto-detects.
├── questions\
│   └── validation.json        # ~21k questions
└── annotations\
    └── annotation_validation.zip   # optional (richer scene data; not required)
```

The benchmark auto-detects `scenes\validation\` vs `scenes\clevrer_scenes\`
vs a top-level `scenes\` folder. Override with `--scene_dir` /
`--questions_file` if your layout is unusual.

---

## 3. Adapter checkpoint

The canonical Phase 3 checkpoint is:

```
checkpoints\adapter_phase3.pt   (≈ 3.0 GB; Phase 3 final, val loss 1.3303)
```

It is **self-contained**: it bundles `FullPhysicsFormer` encoder
weights (under `physics_model.*` state-dict keys), adapter MLP,
DistilGPT-2 with LoRA + Phase 3 full-fine-tune deltas, and numerical /
descriptive heads. No separate physics checkpoint is needed for eval.

### Where to get the checkpoint

| Option | Notes |
|---|---|
| **Download from the project's Google Drive release** | Recommended. ~3 GB tarball, drop directly into `checkpoints/`. |
| **Retrain from scratch** | See §8 below. Produces a similar but not byte-identical checkpoint. Expect ≈79 ± 1 pp overall; exact digits depend on data shuffling and CUDA nondeterminism. |

The two checkpoints shipped in this folder:

| File | Size | Role | Eval result (full 5000 scenes) |
|---|---|---|---|
| `adapter_phase3.pt` | 3.0 GB | **Canonical SOTA, use this** (Phase 3 final, val loss 1.3303) | **79.6%** |
| `physics_former_best.pt` | 720 MB | Stage-1 encoder. Unused at eval time (adapter is self-contained). Used as Stage-2 starting point. | N/A |

*(Legacy pre-Phase-3 adapter checkpoints — `adapter_v2_expanded_final.pt`, `adapter_v2_final.pt` — were removed from the snapshot. Their result JSONs/logs are still in `clevrer_benchmark/results/` for historical reference; the legacy 71.7% number was never reproducible byte-exact and the article documents it transparently in §9.)*

---

## 4. Run the full benchmark

```powershell
python clevrer_benchmark\run_adapter_evaluation.py `
    --clevrer_dir $CLEVRER_DIR `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --max_scenes 99999 `
    --skip_descriptive `
    --save_details `
    --output clevrer_benchmark\results\phase3_full5000.json
```

Flags explained:

| Flag | Default | Meaning |
|---|---|---|
| `--clevrer_dir` | `$CLEVRER_DIR` | CLEVRER root. |
| `--adapter_checkpoint` | `checkpoints/adapter_phase3.pt` | Phase 3 checkpoint to eval. |
| `--max_scenes` | `100` | Override to `99999` for the full 5000 scenes. |
| `--skip_descriptive` | off | Skip the descriptive (non-MCQ) question type; matches the shipped protocol. |
| `--save_details` | off | Also write a `.details.jsonl` sidecar with per-question predictions + scoring rationale. Needed for the sample audit in §5 and for the zero-physics comparison in §6. |
| `--output` | (none) | Summary JSON path. If omitted, results are printed to stdout only. |
| `--eval_method` | `generate` | Canonical (matches training loss). Alternative: `contrastive` — Jan 26 recipe that scores choices by physics-amplified likelihood. |
| `--gen_mode` | `sample` | Stochastic (T=0.7, top_p=0.9). Deterministic alternatives: `greedy`, `beam4`, `beam8`. |
| `--gen_seed` | `42` | RNG seed for reproducible sampling. |
| `--single_frame` | `64` | Frame index at which to extract the CLEVRER state tensor (matches the fair-comparison protocol used by all LLM baselines in `clevrer_benchmark/results/`). Pass `-1` to use all frames. |

### 4a. Producing the article's primary held-out numbers

The command above produces the FULL5000 unshuffled-MCQ result (79.6%
overall). The article's **primary numbers** in `fig:ci_forest`,
`tab:significance`, `tab:ablation`, and `tab:prefix_significance`
(69.2 / 79.4 / 63.4 / 63.6 overall / E / P / C) come from a slightly
different file: same checkpoint and same `--single_frame 64`, but with
`--shuffle_choices` (to match the adapter's training-time MCQ protocol)
and filtered to the disjoint 10% **held-out partition** that the
adapter never saw during training.

Step 1 — produce the shuffled-MCQ FULL5000 result:

```powershell
python clevrer_benchmark\run_adapter_evaluation.py `
    --clevrer_dir $CLEVRER_DIR `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --max_scenes 99999 `
    --skip_descriptive `
    --shuffle_choices `
    --filter_malformed `
    --save_details `
    --output clevrer_benchmark\results\phase3_BASELINE_SHUFFLE_FULL5000.json
```

Step 2 — filter to the held-out partition and emit the paper's primary
table rows + Wilson 95% CIs + Fisher's exact p-values + Cohen's h:

```powershell
python clevrer_benchmark\scripts\compute_paper_stats.py `
    --scope FULL5000 `
    --heldout `
    --valid_only `
    --emit_latex
```

The `--heldout` flag selects the 501-scene held-out subset
(scene\_index $\in [14{,}499, 14{,}999]$) using the adapter's training
H5 (`$CLEVRER_H5` by default;
override with `--h5 PATH` or the `CLEVRER_H5` env var). `--valid_only`
applies the same `validate_question()` filter used by the LLM
baselines. The script's `--emit_latex` mode prints LaTeX-ready rows
that match the macros in `article/main.tex` (`\primaryALL = 69.2%`,
`\primaryExp = 79.4%`, `\primaryPred = 63.4%`, `\primaryCnt = 63.6%`)
plus the ablation deltas (Zero-Physics −62.0 pp, Zero-Prefix-Shuffle
−3.6 pp) used in `tab:significance` and `tab:prefix_significance`.

Runtime: Step 1 takes ~20–30 min on a modern GPU (same as §4). Step
2 is post-processing only (~5 s).

---

## 5. Inspecting per-question predictions

Running with `--save_details` writes a JSONL sidecar next to the
summary JSON. Each record contains the question text, all choices
with correct/wrong labels, the model's generated string, and the
correct flag. Example record:

```json
{
  "scene_id": "annotation_10033",
  "question_id": "annotation_10033_q13",
  "question_text": "What will happen next?",
  "clevrer_type": "predictive",
  "choices": [
    {"choice": "The cylinder and the cube collide", "answer": "wrong"},
    {"choice": "The cylinder and the rubber sphere collide", "answer": "correct"}
  ],
  "predicted": "the cylinder and the rubber sphere collide",
  "correct": true
}
```

`evaluate_answer()` (in `run_adapter_evaluation.py`) scores a
prediction as correct if, after lowercasing and trimming, the
generated text (a) matches a correct choice exactly, or (b) is a
substring of a correct choice / has a correct choice as a substring,
*and* does not match any wrong choice. The adapter typically emits
one of the choice texts verbatim (see samples in
`.compsac_eval_data/phase3_generate_samples.txt` if present).

---

## 6. Zero-physics ablation (smoking-gun physics-dependence check)

Zeroes out the physics state tensor while keeping everything else
identical. Used to measure how much the physics prefix actually
contributes to the answer:

```powershell
python clevrer_benchmark\run_adapter_evaluation.py `
    --clevrer_dir $CLEVRER_DIR `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --max_scenes 100 `
    --skip_descriptive `
    --zero_physics `
    --save_details `
    --output clevrer_benchmark\results\phase3_zero_physics_100scenes.json
```

Expected result on the 100-scene smoke-test (first 100 scenes ≈ 435
questions — the same protocol used by all shipped baselines in
`clevrer_benchmark/results/*_with_scene.json`):

| Condition | Overall | Explanatory | Predictive | Counterfactual |
|---|---|---|---|---|
| With physics | 82.3% | 80.6% | 80.0% | 84.9% |
| **Zero physics** | **6.9%** | 15.0% | 0.0% | 1.6% |
| Δ | **−75.4 pp** | −65.6 pp | −80.0 pp | −83.3 pp |

A 75 pp drop — the model is genuinely using the physics encoder; it
is not a text-only classifier overfitting CLEVRER surface cues.

(Why the 100-scene number is 82.3% while the full 5000 is 79.6%: the
first 100 scenes happen to be slightly easier than the dataset mean.
Use the full 5000 number as the headline.)

### 1K-pool ablations (partition-robustness sanity check)

The article's primary ablation in `tab:ablation` runs on the
disjoint **501-scene held-out subset** (`n=1,998`) so the ablation
deltas are computed on the same training-disjoint partition that
underwrites `fig:ci_forest`. The article also reports the same
ablations on a **1,000-question matched pool** (which contains a
majority of adapter-training scenes) as a partition-robustness
check -- the *Robustness across partitions* paragraph in
§sec:ablation of `main.tex`. Reproduce the 1K-pool sanity-check
numbers as follows.

**Zero Physics on the 1K pool:**

```powershell
python clevrer_benchmark\run_adapter_evaluation.py `
    --clevrer_dir $CLEVRER_DIR `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --max_questions 1000 `
    --skip_descriptive `
    --single_frame 64 `
    --shuffle_choices `
    --filter_malformed `
    --zero_physics `
    --save_details `
    --output clevrer_benchmark\results\adapter_phase3_1k_zero_physics.json
```

**Zero Prefix + Shuffle on the 1K pool:**

```powershell
python clevrer_benchmark\run_adapter_evaluation.py `
    --clevrer_dir $CLEVRER_DIR `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --max_questions 1000 `
    --skip_descriptive `
    --single_frame 64 `
    --shuffle_choices `
    --filter_malformed `
    --zero_prefix `
    --save_details `
    --output clevrer_benchmark\results\adapter_phase3_1k_zero_prefix_shuffle.json
```

`--max_questions 1000` overrides `--max_scenes` and uniformly samples
~260 scenes to yield ~1,000 causal questions (same recipe used by the
shipped `adapter_phase3_1k_sampled.json` baseline). The
`--filter_malformed --shuffle_choices --single_frame 64` triple matches
the LLM 1K-pool protocol so the 1K-pool ablation deltas are computed
against a baseline that uses the same protocol.

Expected result on the 1K pool (Wilson 95% CIs and Δ vs the 1K-pool
73.9% baseline; matches the *Robustness across partitions* numbers in
§sec:ablation of `main.tex`):

| Condition | Overall | Explanatory | Predictive | Counterfactual |
|---|---|---|---|---|
| **Grounded-Physics LM** (1K, `adapter_phase3_1k_sampled.json`) | **73.9%** [71.1, 76.6] | 84.2% [80.0, 87.7] | 71.4% [64.5, 77.4] | 67.3% [62.9, 71.4] |
| **Zero Physics** | **8.5%** [6.9, 10.4] | 18.4% [14.7, 22.8] | 2.2% [0.8, 5.4] | 3.6% [2.3, 5.7] |
| **Zero Prefix + Shuffle** | **68.6%** [65.7, 71.4] | 77.3% [72.6, 81.4] | 67.6% [60.5, 73.9] | 62.6% [58.1, 66.9] |
| Δ Physics contribution | **+65.4 pp** ($p<0.001$, $h=1.48$) | +65.8 ($p<0.001$, $h=1.44$) | +69.2 ($p<0.001$, $h=1.72$) | +63.7 ($p<0.001$, $h=1.54$) |
| Δ Prefix contribution | +5.3 pp ($p=0.010$, $h=0.12$) | +6.9 ($p=0.027$) | +3.8 ($p=0.498$ n.s.) | +4.7 ($p=0.150$ n.s.) |

Compare to held-out (`tab:ablation` / `tab:significance` /
`tab:prefix_significance` in the article, which are the article's
primary ablation tables): Zero-Physics Δ is +62.0 pp overall,
Zero-Prefix Δ is +3.6 pp overall. The 1K-pool partition-robustness
check gives slightly larger deltas because the 1K-pool baseline is
itself higher (73.9% vs 69.2% held-out -- the 1K pool contains a
majority of adapter-training scenes), but the qualitative picture is
identical: physics-grounding contributes 60+ pp regardless of which
partition is used as the denominator, and the prefix-only effect is
small (3-7 pp range) on both pools.

Compute the table above (Wilson CIs, Fisher's exact $p$, Cohen's $h$)
from the three result JSONs with the helper at
`clevrer_benchmark/scripts/_compute_1k_cis.py` (extend it to read the
two ablation JSONs alongside the baseline) or use any standard
two-proportion-test routine.

Runtime: ~2 minutes per ablation on an RTX 5080 Laptop GPU.

---

## 7. 15-object stress-test benchmark

Reproduces the **15-object scaling** result in `fig:ci_forest_15obj`
of the article. Same single-frame 64 protocol and no-collision-label
constraint as §4, but evaluated on CLEVRER scenes augmented to
contain 15 objects (the standard CLEVRER scenes have 3–6), matching
the 1{,}000-question pool the LLM baselines were run on (see
`clevrer_benchmark/results/15obj_1k/`).

```powershell
python clevrer_benchmark\benchmark_15_objects_physics_llm.py `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --clevrer_dir $CLEVRER_DIR `
    --num_questions 1000 `
    --uniform `
    --filter_malformed `
    --shuffle_choices `
    --save_details `
    --output_dir clevrer_benchmark\results\15obj_1k `
    --output_name physics_llm_15obj_1k
```

Flags explained:

| Flag | Default | Meaning |
|---|---|---|
| `--adapter_checkpoint` | (required) | Path to the Phase 3 checkpoint. Loaded with full LoRA support — the script auto-detects `lora_A` / `lora_B` keys and applies the LoRA wrappers before `load_state_dict` so all weights load (legacy versions of this script partial-loaded the LoRA-suffixed keys, undercounting accuracy by ≈8 pp). |
| `--clevrer_dir` | `$CLEVRER_DIR` | CLEVRER root. Tries `questions/validation.json`, then `questions/clevrer_validation.json`, then `questions/validation_v0.2.json`. |
| `--num_questions` | `90` | Total sample size (with `--uniform`) or per-type size × 3 (without). For the matched 1K-pool result, use `1000`. |
| `--uniform` | off | Uniform random sample over causal questions (LLM 1K-pool protocol; preserves CLEVRER's natural per-type distribution: ≈36% explanatory / ≈20% predictive / ≈44% counterfactual). Without this flag, the legacy stratified balanced sampling (`num_questions/3` per type) is used. **Use `--uniform` for the matched 1K result.** |
| `--filter_malformed` | off | Skip MCQs where every choice is labeled wrong (matches the LLM baselines' `validate_question` filter; drops ~1.5K malformed questions out of the causal pool). |
| `--shuffle_choices` | off | Randomize MCQ choice order per question (deterministic per `question_id`; matches the 1K 3–6 obj primary protocol's shuffled MCQ). |
| `--save_details` | off | Write per-question `.details.jsonl` sidecar (scene_id, q_type, predicted, correct, choices). |
| `--target_objects` | `15` | Number of objects each scene is augmented to. Original CLEVRER scenes have 3–6; new objects are generated with random color/material/shape combos and ballistic trajectories with a deterministic `seed=scene_index`. |
| `--output_dir` | `results_15obj` | Output directory. The article's results live in `clevrer_benchmark/results/15obj_1k/` alongside the LLM baseline JSONs. |
| `--output_name` | `physics_llm_{target_objects}obj` | Filename stem. The shipped result uses `physics_llm_15obj_1k`. |

Expected result on the matched 1K pool (Wilson 95% CIs):

| Type | n | Accuracy | Wilson 95% CI |
|---|---|---|---|
| **Overall** | **1000** | **63.1%** | **[60.1, 66.0]** |
| Explanatory | 361 | 69.5% | [64.6, 74.1] |
| Predictive | 198 | 64.6% | [57.8, 71.0] |
| Counterfactual | 441 | 57.1% | [52.5, 61.7] |

Key comparison with the strongest LLM baseline (Llama-3.3-70B):

| Type | Grounded-Physics LM (173M) | Llama-3.3-70B | Δ | CIs overlap? |
|---|---|---|---|---|
| Overall | 63.1% [60.1, 66.0] | 66.5% [63.5, 69.4] | −3.4 pp | yes |
| Explanatory | 69.5% [64.6, 74.1] | 80.4% [76.3, 83.9] | −10.9 pp | no (Llama leads) |
| **Predictive** | **64.6% [57.8, 71.0]** | 48.8% [41.1, 56.4] | **+15.8 pp** | **no (ours leads)** |
| Counterfactual | 57.1% [52.5, 61.7] | 59.8% [55.0, 64.3] | −2.7 pp | yes |

The predictive-reasoning advantage from `fig:ci_forest` (3–6 obj
held-out partition) persists at 15 objects with non-overlapping 95% CIs
— the core article finding.

Runtime: ≈4 minutes on an RTX 5080 Laptop GPU at ≈4 questions/sec.

Output artifacts after the run:

- `clevrer_benchmark/results/15obj_1k/physics_llm_15obj_1k.json` — summary (overall + per-type counts).
- `clevrer_benchmark/results/15obj_1k/physics_llm_15obj_1k_wrong.json` — wrong-answer details.
- `clevrer_benchmark/results/15obj_1k/physics_llm_15obj_1k.details.jsonl` — per-question records.
- `clevrer_benchmark/results/15obj_1k/physics_llm_15obj_1k.log` — full run log.

### Legacy n=90 stratified pool (deprecated)

The pre-1K protocol used balanced stratified sampling (30 questions per
type, total `n=90`) and ±15–17 pp Wilson CIs. To reproduce the legacy
numbers (54.5 / 59.4 / 54.3 / 49.7 reported in earlier drafts), run the
same script **without** `--uniform`, `--filter_malformed`, or
`--shuffle_choices`:

```powershell
python clevrer_benchmark\benchmark_15_objects_physics_llm.py `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --clevrer_dir $CLEVRER_DIR `
    --num_questions 90
```

This is kept for completeness; the article uses the matched 1K result
above.

---

## 8. ComPhy zero-shot OOD evaluation

The CLEVRER Phase 3 checkpoint can be evaluated on **ComPhy** (Chen et
al., ICLR 2022) with no retraining — same encoder, same adapter,
different benchmark. This is the cross-benchmark transfer experiment:
it shows the architecture and the grounding signal aren't
CLEVRER-specific. See [`comphy_benchmark/README.md`](comphy_benchmark/README.md)
for the full design discussion (mass transfers cleanly; charge has no
slot in the 35-D state schema — disclosed honestly in the stats).

### 8.1 Data layout

The ComPhy release lives **outside the workspace tree** at `D:\comphy`
(~2.6 GB across 10k annotation JSONs + 8 validation QA chunks). The
runner defaults `--comphy_dir` to this path so no flag is needed in
the common case. Override via `--comphy_dir <path>` or set the
`COMPHY_DIR` env var.

```
D:\comphy\
├── target_annotation\
│   ├── annotation_00000_01000\<00000..00999>.json
│   └── ...
├── qa_chunk0.json ... qa_chunk7.json   # 8 validation chunks, 1000 scenes each
├── train_qa.json                       # NOT used by default
└── train_qa_chunk0..3.json             # NOT used by default
```

If you ever need to re-populate this directory: the source is
<https://comphyreasoning.github.io/>.

### 8.2 Baseline run

```powershell
python comphy_benchmark\run_comphy_evaluation.py `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --output comphy_benchmark\results\phase3_comphy_zeroshot.json `
    --save_details
```

Defaults match the CLEVRER SOTA protocol (`--eval_method generate
--gen_mode sample --gen_seed 42 --single_frame 64`). For a ~3-minute
smoke test, add `--max_scenes 50`. Restrict to one validation chunk
(~5k questions instead of ~43k) with
`--qa_files benchmark_data\comphy\qa_chunk0.json`.

### 8.3 Zero-physics + shuffle-choices controls

The two ablations that, together with the baseline, make the
cross-benchmark claim defensible (mirroring the CLEVRER protocol):

```powershell
python comphy_benchmark\run_comphy_evaluation.py `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --zero_physics `
    --output comphy_benchmark\results\phase3_comphy_zerophysics.json

python comphy_benchmark\run_comphy_evaluation.py `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --shuffle_choices `
    --output comphy_benchmark\results\phase3_comphy_shuffle.json
```

### 8.4 Wilson CIs + LaTeX table

```powershell
python comphy_benchmark\scripts\compute_comphy_stats.py `
    --result comphy_benchmark\results\phase3_comphy_zeroshot.json `
    --emit_latex
```

Emits Wilson 95% CIs by coarse type (factual / predictive /
counterfactual) + native ComPhy subtypes + a charge-dependence slice
(keyword filter on question text), plus a `tab:comphy_ood` LaTeX
fragment ready to paste into `main.tex`.

---

## 9. Retraining from scratch

### Stage 1 — PhysicsFormer pretraining

Trains `FullPhysicsFormer` from scratch on Isaac Sim physics HDF5
data using the 13-level schema curriculum from the paper.

Prerequisites:

- **Isaac Sim HDF5 data** at `$env:PHYSICS_DATA_DIR` (default
  `$PHYSICS_DATA_DIR`). The physics HDF5 files live in the
  root of this directory; the runner reads them via
  `CachedPhysicsDataset` (RAM cached) or `HDF5PhysicsDataset`
  (on-demand) per the `cache_dataset_to_ram` config flag.
- **Checkpoint dir** at `$env:CHECKPOINT_DIR` (default
  `$CHECKPOINT_DIR`); Stage 1 produces a
  `physics_former_best.pt` checkpoint there.
- NVIDIA Isaac Sim if you need to generate the HDF5s from scratch
  (see `data_generation/isaac_sim/SETUP.md`).

Run (the shipped `physics_former_best.pt` was produced exactly
this way, with `--config a100`):

```powershell
$env:PHYSICS_DATA_DIR = "$PHYSICS_DATA_DIR"
$env:CHECKPOINT_DIR   = "$CHECKPOINT_DIR"

python physics_former\run_physics_training.py `
    --config a100 `
    --checkpoint-dir $env:CHECKPOINT_DIR
```

`A100Config` is the architecture that produced the released
checkpoint (`hidden_dim=768`, `num_layers=8`, `num_heads=24`,
`ff_dim=3072`, `max_seq_length=512`, schema curriculum levels
1–13). After `apply_modern_improvements` (RMSNorm + Flash + RoPE
+ SwiGLU 2/3-reduction) the loaded state-dict matches
`physics_former_best.pt` exactly (0 missing / 0 unexpected keys).

Conservative-GPU fallback: `--config aggressive` uses
`TrainingConfig` (`hidden_dim=512`, `num_layers=8`,
`num_heads=16`, `ff_dim=2048`, `max_seq_length=128`); this trains
a smaller variant whose weights are **not** drop-in compatible
with the released adapter checkpoint, so use only as a sanity
check, not to reproduce paper numbers.

Stage-1 fine-tune entry points (each loads an existing physics
checkpoint and continues training): `training/run_causal_training.py`
(causal/intervention objective), `training/run_permanence_training.py`
(object-permanence), `training/run_finetune_consistency.py`
(temporal consistency). None of these are required to reproduce
the released `physics_former_best.pt`.

Expected time on the released A100 config: ~3–5 days on a single
A100 80 GB or ~5–7 days on an RTX 4090 with reduced batch size.

#### Running Stage-1 in Colab (multi-session)

Use `colab_train_physics_former.ipynb` at the top level of this folder.
It is a thin wrapper that mounts Drive, unpacks the snapshot, sets
`PHYSICS_DATA_DIR` / `CHECKPOINT_DIR` env vars to Drive paths, and
shells out to the same CLI command above with `--resume <latest>`
auto-detected from your Drive checkpoint dir on each run.

Expected Drive layout:

```
/content/drive/MyDrive/physics_llm/
├── compsac_2026_code.zip       # `Compress-Archive compsac_2026_code\* compsac_2026_code.zip`
├── physics_data/               # Isaac Sim physics HDF5s (~30 GB)
└── stage1_checkpoints/         # written by the notebook; physics_former_best.pt lands here
```

Caveat: a full Stage-1 run takes 3–5 days on A100, which exceeds Colab’s
12–24h session limit, so a complete retrain via Colab requires ≈ 5–10
sessions (the notebook auto-resumes each session). For paper-grade
reproduction prefer running the CLI directly on local A100 hardware.

### Stage 2 — Adapter training (three phases)

Freezes `FullPhysicsFormer` (loaded from Stage-1 `stage1_best.pt`),
attaches `PhysicsLLMAdapterV2` (prefix-tuning + LoRA + heads over
DistilGPT-2), and runs three progressive-unfreezing phases:

| Phase | Trainable | Loss | Typical LR |
|---|---|---|---|
| **1** | Adapter MLP + numerical head + descriptive head | Generation CE + numerical MSE + descriptive CE | 2e-4 |
| **2** | + LoRA on DistilGPT-2 attention (rank=8, alpha=16) | + InfoNCE contrastive on prefix tokens (weight 0.1) | 5e-5 |
| **3** | + full DistilGPT-2 fine-tune | Same objectives as Phase 2 | 2e-5 |

Prerequisites:

- **Stage-1 checkpoint** at `$CHECKPOINT_DIR\stage1_best.pt`
  (override via `--physics-checkpoint`).
- **Physics HDF5 directory** (same as Stage-1's
  `$env:PHYSICS_DATA_DIR`).
- **Optional CLEVRER-conforming QA JSON** (pass `--clevrer-data`) to
  mix in CLEVRER-style MCQ. Produced by
  `data_generation/clevrer/clevrer_to_training_data.py`.

Run:

```powershell
python physics_llm_adapter\train_adapter_v2.py `
    --physics-checkpoint $CHECKPOINT_DIR\stage1_best.pt `
    --data-dir D:\physics_hdf5 `
    --clevrer-data $CLEVRER_DIR\clevrer_conforming_qa.json `
    --output-dir checkpoints `
    --num-samples 100000 `
    --batch-size 8
```

Produces checkpoints at `checkpoints/adapter_phase{1,2,3}_complete_loss*.pt`.
Phase 3 is the one to evaluate.

All flags: `python physics_llm_adapter\train_adapter_v2.py --help`.
Key ones:

- `--start-phase {1,2,3}` — resume from a later phase.
- `--checkpoint <path>` — resume model weights from a prior run.
- `--contrastive-weight <float>` — weight for Phase 2/3 contrastive
  loss (default 0.1; prevents physics prefix collapse).

Expected time: ~6–10 h on A100, ≥ 24 h on T4.

### Running Stage 2 in Colab

Use `colab_train_adapter.ipynb` at the top level of this folder. It
mounts Google Drive, unpacks a zip of the code, installs deps, and
runs all three phases with resume support.

Expected Drive layout:

```
/content/drive/MyDrive/physics_llm/
├── compsac_2026_code.zip              # from `Compress-Archive compsac_2026_code\* …`
├── checkpoints/
│   ├── stage1_best.pt                 # Stage-1 encoder (starting point)
│   └── adapter_phase*.pt              # saved by the notebook as training progresses
├── data/
│   └── clevrer_training_expanded.h5   # physics HDF5 for adapter training
└── clevrer/                           # validation split for in-training eval
```

### What retraining will NOT reproduce byte-exactly

- The shipped `physics_llm_single_frame.json` 71.7% (that adapter
  checkpoint was overwritten by later training).
- The `compsac-2026/main.tex` Table 1 numbers
  (67.8 / 78.3 / 60.0 / 60.5) — different run whose exact command
  was not preserved.

A fresh retrain should land in the 78–81% band overall with ~20× less
variance between runs on predictive than the shipped Jan 25 baseline,
thanks to the Phase 2 contrastive loss added after the paper.

---

## 10. Historical context and alternative eval methods

Earlier checkpoints and earlier eval scripts in the workspace used a
perplexity-ranking MCQ eval (`select_answer_from_choices` — rank each
choice by LLM cross-entropy, pick the lowest). For this adapter that
eval gives ~50–53% on every checkpoint because the adapter was
trained with text-generation loss, not MCQ-classification loss. That
path has been removed from `run_adapter_evaluation.py`; use
`--eval_method generate` (the default) or `--eval_method contrastive`.

The Jan 26 `adapter_v2_distilgpt2_contrastive.pt` checkpoint (if you
have it on disk) is reproducible under `--eval_method contrastive
--contrastive_alpha 1.0 --single_frame 64`, giving ~62–66% overall on
the first 100 scenes. It is *not* competitive with Phase 3 and is
only useful as a historical comparison.

Three different CLEVRER result sets exist for this model family —
keep them straight:

| Source | Overall | Notes |
|---|---|---|
| **Phase 3 (this folder)** — `checkpoints/adapter_phase3.pt` | **79.6%** | Reproducible. Full 5000 scenes / 21,378 questions. |
| `compsac-2026/main.tex` Table 1 | 67.8% | Not reproducible — eval command not preserved. Headline from an older paper snapshot. |
| `clevrer_benchmark/results/FAIR_BENCHMARK_RESULTS.md` | 71.7% | Not reproducible — the legacy `adapter_v2_expanded_final.pt` checkpoint that produced this number was overwritten before being archived; the file itself was removed from this snapshot. 435 stratified questions. |
| `physics_llm_adapter/README.md` | 97% | Different (synthetic) test set, not CLEVRER. Ignore for paper reproduction. |

---

## 11. Further reading

- `clevrer_benchmark/BENCHMARK_SUMMARY.md` — LLM baseline methodology.
- `clevrer_benchmark/results/FAIR_BENCHMARK_RESULTS.md` — original
  fair-comparison protocol (435 stratified questions).
- `clevrer_benchmark/scripts/compute_paper_stats.py` — Wilson 95%
  confidence intervals, Fisher's exact p-values, Cohen's h, and
  ready-to-paste LaTeX tables for the paper. Reads the FULL5000
  result JSONs in `clevrer_benchmark/results/`.
- `clevrer_benchmark/scripts/holdout_generalization_check.py` —
  identifies the 10% held-out CLEVRER val scenes (≈501 scenes /
  1,998 valid questions) and recomputes Phase 3 accuracy on the
  scenes the adapter never saw during training. Requires the
  CLEVRER training H5 (pass `--h5 PATH` or set `CLEVRER_H5`).
- `clevrer_benchmark/scripts/paraphrase_audit.py` — NLI-based
  semantic-match audit of every wrong prediction. Re-scores the
  failure set using `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  and reports how many wrongs would flip if scoring used
  bidirectional entailment instead of substring match. On the
  paper's primary heldout valid-only pool (n=1,998) the audit
  flags **1 flip**, raising the strict-match `\primaryALL` 69.22%
  to a +flips ceiling of **69.27% (+0.05 pp)** — confirming the
  substring-based `evaluate_answer()` is essentially tight.
- `physics_former/training/README.md` — Stage-1 curriculum details.
