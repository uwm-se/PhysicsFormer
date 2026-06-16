# Physics-LLM: Code Snapshot

Clean, self-contained snapshot of the **82M Physics-LLM** — a
PhysicsFormer physics encoder + DistilGPT-2 prefix-tuned with a
`PhysicsLLMAdapterV2` + LoRA — for the CompSAC-2026 paper.

**Headline result:** **79.6% overall** on the full CLEVRER validation
split (21,378 explanatory / predictive / counterfactual MCQ across
5000 scenes), reproducible from this folder with a single command —
see [`REPRODUCTION.md`](REPRODUCTION.md).

## Results

### CLEVRER validation (full 5000 scenes, 21,378 questions)

Phase 3 checkpoint `checkpoints/adapter_phase3.pt`
under `--eval_method generate --gen_mode sample --gen_seed 42
--single_frame 64` (all are defaults):

| Split           | Count         | Accuracy  |
|-----------------|---------------|-----------|
| **Overall**     | 17021 / 21378 | **79.6%** |
| Explanatory     |  6700 / 8488  | 78.9%     |
| Predictive      |  2718 / 3557  | 76.4%     |
| Counterfactual  |  7603 / 9333  | 81.5%     |

Result JSON:
`clevrer_benchmark/results/phase3_GENERATE_singleframe_FULL5000.json`
(21,378 per-question records in the `.details.jsonl` sidecar).

### Article primary: held-out partition

The headline above is the full-5000-scene reproducibility number. The
**article reports a stricter cut**: the disjoint 10% **held-out
partition** the adapter never saw during training (501 scenes,
scene\_index $\in [14{,}499, 14{,}999]$). This is the only
training-disjoint subset for our model and is what `tab:significance`,
`tab:ablation`, and `fig:ci_forest` evaluate. LLM baselines run
zero-shot on a matched 1{,}000-question pool (any CLEVRER subset is an
unbiased zero-shot measurement for them since they never train on
CLEVRER).

**3-6 object held-out partition** (`fig:ci_forest`, all ablation
tables in §IV):

| Type           | n    | Accuracy | Wilson 95% CI |
|----------------|------|----------|---------------|
| **Overall**    | 1998 | **69.2%** | [67.2, 71.2] |
| Explanatory    |  710 | 79.4%    | [76.3, 82.2]  |
| Predictive     |  361 | 63.4%    | [58.3, 68.2]  |
| Counterfactual |  927 | 63.6%    | [60.5, 66.7]  |

Best LLM baseline on the matched 1K-pool zero-shot evaluation:
Llama-3.3-70B at 62.5% overall (non-overlapping CIs against Ours: [59.5,
65.4] vs [67.2, 71.2]). On **predictive specifically**, Grounded-Physics
LM 63.4% [58.3, 68.2] beats every LLM baseline with non-overlapping
CIs (best LLM: Qwen2.5-7B at 52.5% on the n=1,000 predictive supplement,
[49.4, 55.6]).

**15-object stress test, 1K pool** (`fig:ci_forest_15obj`): a separate
OOD evaluation on object count (the adapter trained only on 3--6
objects), independent of the held-out partition:

| Type           | n    | Accuracy | Wilson 95% CI |
|----------------|------|----------|---------------|
| **Overall**    | 1000 | **63.1%** | [60.1, 66.0] |
| Explanatory    |  361 | 69.5%    | [64.6, 74.1]  |
| **Predictive** |  198 | **64.6%** | [57.8, 71.0] |
| Counterfactual |  441 | 57.1%    | [52.5, 61.7]  |

The predictive lead persists at scene complexity beyond the training
distribution: Grounded-Physics LM 64.6% [57.8, 71.0] vs the best LLM
on 15-obj predictive (DeepSeek-V3 at 53.8% [46.0, 61.3]) and the
best-overall LLM (Llama-3.3-70B at 48.8% [41.1, 56.4]) — non-overlapping
CIs in both cases.

Result JSONs:
- `clevrer_benchmark/results/phase3_BASELINE_SHUFFLE_FULL5000.details.jsonl`
  filtered by `clevrer_benchmark/scripts/compute_paper_stats.py --heldout
  --valid_only` (3--6 obj held-out partition)
- `clevrer_benchmark/results/15obj_1k/physics_llm_15obj_1k.json` (15-obj)

Reproduction: see [`REPRODUCTION.md` §4](REPRODUCTION.md) (held-out)
and [§7](REPRODUCTION.md) (15-obj).

### ComPhy zero-shot OOD (cross-benchmark transfer)

The CLEVRER-trained Phase 3 checkpoint can also be evaluated on
**ComPhy** (Chen et al., ICLR 2022) with no retraining — tests whether
the architecture and the grounding signal transfer to a different
physics-reasoning benchmark. Mass transfers cleanly into the 35-D
state; charge has no slot in the schema and is disclosed honestly in
the stats. See [`comphy_benchmark/README.md`](comphy_benchmark/README.md)
and [`REPRODUCTION.md` §8](REPRODUCTION.md).

### Zero-physics ablation (first 100 scenes, 435 questions)

Zeroing the physics state tensor drops performance from **82.3% →
6.9%** (−75.4 pp). The model is genuinely using the physics encoder,
not memorizing text-only surface cues:

| Condition    | Overall | Explanatory | Predictive | Counterfactual |
|--------------|---------|-------------|------------|----------------|
| With physics | 82.3%   | 80.6%       | 80.0%      | 84.9%          |
| Zero physics |  6.9%   | 15.0%       |  0.0%      |  1.6%          |
| Δ            | −75.4   | −65.6       | −80.0      | −83.3          |

### vs. Jan 25 fair-comparison baseline (435 stratified questions)

| Category        | Phase 3 today (full 5000) | Jan 25 baseline (435) | Δ         |
|-----------------|---------------------------|-----------------------|-----------|
| **Overall**     | **79.6%**                 | 71.7%                 | **+7.9**  |
| Explanatory     | 78.9%                     | 74.4%                 | +4.5      |
| **Predictive**  | **76.4%**                 | 60.0%                 | **+16.4** |
| Counterfactual  | 81.5%                     | 73.5%                 | +8.0      |

Phase 3 beats the Jan 25 baseline on all three question types, with
the largest gain on the hardest category (predictive). This is on
~50× more questions (21,378 vs 435), so it is a strictly more
rigorous measurement.

## Architecture

| Component          | Value                                                  |
|--------------------|--------------------------------------------------------|
| Physics encoder    | `FullPhysicsFormer` (`physics_former.training.models.physics_former_full`) — trained Stage 1 |
| Language model     | DistilGPT-2 (~82M params)                              |
| Adapter            | `PhysicsLLMAdapterV2` (prefix-tuning, 64 prefix tokens) + LoRA (rank 8, alpha 16) on DistilGPT-2 attention |
| Auxiliary heads    | Numerical (6-D regression), Descriptive (count / exist / color / shape / material classifiers), MCQ scorer |
| Input              | Single-frame CLEVRER state tensor `[1, N, 35]`         |
| Checkpoint         | `adapter_phase3.pt` (self-contained, ≈ 3.0 GB; Phase 3 final, val loss 1.3303) |

The adapter checkpoint is **self-contained**: `FullPhysicsFormer`
weights live inside the same `.pt` file under `physics_model.*`
state-dict keys, alongside the adapter MLP, DistilGPT-2 with LoRA +
Phase 3 full-fine-tune deltas, and the auxiliary heads. No separate
physics checkpoint is needed for evaluation.

### Training recipe (Phase 3 SOTA)

- **Phase 1** — Adapter MLP + numerical/descriptive heads trainable,
  LLM fully frozen. Generation cross-entropy + numerical MSE +
  descriptive CE. LR 2e-4.
- **Phase 2** — + LoRA on DistilGPT-2 attention (rank 8, alpha 16,
  ~405k extra trainable params). + InfoNCE contrastive loss on
  prefix tokens (weight 0.1) to prevent physics collapse. LR 5e-5.
- **Phase 3** — + full DistilGPT-2 fine-tune. Same objectives as
  Phase 2. LR 2e-5.

Full command + prerequisites: [`REPRODUCTION.md`](REPRODUCTION.md#7-retraining-from-scratch).

## Reproducing the result

One command (assumes Phase 3 checkpoint is in `checkpoints/` and
CLEVRER is at `$CLEVRER_DIR`):

```powershell
pip install -r requirements.txt

python clevrer_benchmark\run_adapter_evaluation.py `
    --clevrer_dir $CLEVRER_DIR `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --max_scenes 99999 `
    --skip_descriptive `
    --save_details `
    --output clevrer_benchmark\results\phase3_full5000.json
```

The script's defaults (`--eval_method generate --gen_mode sample
--gen_seed 42 --single_frame 64`) are the canonical SOTA-reproducing
setup. Runtime ~20–30 min on a modern GPU.

For a 3-minute smoke-test, drop `--max_scenes 99999` (falls back to
the default 100 scenes; expect 82.3%).

Full walkthrough — environment, data layout, checkpoint retrieval,
zero-physics ablation, and retraining — in
[`REPRODUCTION.md`](REPRODUCTION.md).

## Folder layout

```
compsac_2026_code/
├── README.md                   # This file
├── REPRODUCTION.md             # Detailed reproduction & retraining guide
├── requirements.txt            # torch, transformers, numpy, h5py, tqdm
├── .gitignore
│
├── clevrer_benchmark/          # CLEVRER evaluation
│   ├── run_adapter_evaluation.py         # Main eval entry point (produces 79.6% SOTA)
│   ├── benchmark_15_objects_physics_llm.py
│   ├── benchmark_20_objects_physics_llm.py
│   ├── scene_converter.py      # CLEVRER scene → [T, N, 35] tensor
│   ├── question_mapper.py      # CLEVRER question → adapter format
│   ├── evaluator.py            # Shared eval utilities
│   ├── BENCHMARK_SUMMARY.md    # LLM-baseline methodology
│   ├── results/                # Shipped result JSONs (Physics-LLM + 11 LLM baselines)
│   └── scripts/                # Statistical analysis (Wilson CIs, Fisher's exact, held-out filter)
│       ├── compute_paper_stats.py
│       └── holdout_generalization_check.py
│
├── physics_llm_adapter/        # Adapter + Stage-2 training
│   ├── adapter_v2.py           # PhysicsLLMAdapterV2 class
│   ├── adapter_heads.py        # OutputType, CLEVRER vocab, numerical/descriptive heads
│   ├── mcq_head.py             # MCQ scoring head
│   ├── train_adapter_v2.py     # Stage-2 trainer (three-phase, resumable)
│   ├── language_normalizer.py  # Text normalization
│   └── clevrer_qa_generator.py # CLEVRER-style QA generation
│
├── physics_former/             # Physics encoder + Stage-1 training
│   ├── run_physics_training.py # Stage-1 main entry (produces physics_former_best.pt)
│   └── training/
│       ├── run_causal_training.py      # Stage-1 fine-tune (causal/intervention objective)
│       ├── run_permanence_training.py  # Object-permanence fine-tune
│       ├── run_finetune_consistency.py # Consistency fine-tune
│       ├── causal_training.py
│       ├── progressive_curriculum.py
│       ├── schema_curriculum.py
│       ├── configs/   models/   datasets/   losses/
│       ├── pipelines/ improvements/ common/ utils/
│       └── constants.py
│
├── data_generation/            # Training-data generation
│   ├── isaac_sim/              # Stage-1 Isaac Sim HDF5 generators (needs Isaac Sim)
│   ├── clevrer/                # CLEVRER → training data conversion
│   └── qa_generation/          # Physics QA generation (Stage 2)
│
├── colab_train_adapter.ipynb         # Stage 2 adapter training (Phase 1 + 2 + 3) — produces the SOTA Phase 3 checkpoint.
├── colab_train_physics_former.ipynb  # Stage 1 PhysicsFormer pretraining (thin CLI wrapper, multi-session resume).
│
└── checkpoints/                # Model checkpoints (see checkpoints/README.md)
    ├── README.md
    ├── adapter_phase3.pt                       # SOTA (3.0 GB; Phase 3 final, val loss 1.3303)
    └── physics_former_best.pt                  # Stage-1 encoder
```

## What was deliberately left out

- `physics_encoder_v2.py` / `CausalEncoder` / `causal_encoder_best.pt`
  — a different physics-encoder variant, not the 82M Physics-LLM.
- `demo_adapter.py`, `physics_llm_adapter/baseline_artifacts/`,
  `physics_llm_adapter/scripts/` — earlier exploratory / synthetic-QA
  experiments.
- Non-training parts of `physics_former/` (visualization, dashboards).
- `data_generation/unreal_engine/` (85 MB UE project, not used in the paper).
- `grounded_physics_adapter/`,
  `explicit_world_model/llm_adapters/grounded_physics_lm_adapter/` —
  concept-tokenizer variants, different model family.

## Notes on paper vs. snapshot numbers

Several CLEVRER result sets exist for this model family across
different protocols. **The article primary (`fig:ci_forest` and
all ablation tables) is the 10% held-out partition** the adapter
never saw during training; the full-5000-scene number is the broader
reproducibility check that includes adapter-training scenes. Both are
reproducible from this snapshot:

| Source | Overall | Pool | Reproducible? | Notes |
|---|---|---|---|---|
| **Article primary** (`fig:ci_forest`, `tab:significance`, `tab:ablation`, `tab:prefix_significance`, `tab:confusion`) | **69.2%** | 1,998 (3--6 obj held-out, training-disjoint) | **Yes** — `REPRODUCTION.md` §4 | Strict train/test separation: 501 scenes the adapter never saw. |
| **15-obj stress test** (`fig:ci_forest_15obj`) | **63.1%** | 1,000 (15-obj 1K pool, OOD on object count) | **Yes** — `REPRODUCTION.md` §7 | Separate OOD evaluation on object count; predictive lead persists. |
| Full 5000 scenes (this README §1) | 79.6% | 21,378 (full CLEVRER val) | **Yes** — single command in §Reproducing the result | Broader reproducibility check (includes adapter-training scenes); not used in the article's primary comparisons. |
| 1K-pool baseline (partition-robustness check) | 73.9% | 1,001 (matched LLM pool, 3--6 obj, includes training scenes) | **Yes** — `REPRODUCTION.md` §6 | Retained only as the denominator for the *Robustness across partitions* paragraph in `main.tex`; the article's primary numbers come from the held-out partition above. |
| `clevrer_benchmark/results/FAIR_BENCHMARK_RESULTS.md` (Jan 25) | 71.7% | 435 stratified | No | Legacy checkpoint removed from snapshot. **Superseded** — see banner in that file. |
| `physics_llm_adapter/README.md` | 97% | synthetic | N/A | Different synthetic QA set, not CLEVRER. Ignore. |

The 82M vs 173M param-count discrepancy is accounting: 82M counts
DistilGPT-2 only; 173M counts DistilGPT-2 + `FullPhysicsFormer` +
adapter layers. The underlying model is the same.
