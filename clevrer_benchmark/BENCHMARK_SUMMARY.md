# CLEVRER Benchmark Results Summary

> **⚠️ SUPERSEDED — retained as historical reference.** This document
> records the **January 25, 2026 fair-comparison snapshot** (stratified
> n=90 for the 15-object pool, single-frame-64 protocol). It has since been
> replaced. The article's current primary numbers come from the disjoint
> 10% **held-out partition** (Grounded-Physics LM, n=1,998) plus a
> matched zero-shot 1K-pool for LLM baselines. Current artifacts and
> reproduction commands:
>
> | Pool | Current artifacts | Article reference |
> |---|---|---|
> | 3--6 obj held-out (Ours) + 1K-pool (LLMs) | `results/phase3_BASELINE_SHUFFLE_FULL5000.details.jsonl` filtered by `--heldout --valid_only` (Grounded-Physics LM); `results/*_with_scene_FULL.json` + `results/*_with_scene_PREDICTIVE.json` (LLM baselines) | `fig:ci_forest`, `tab:significance`, `tab:ablation`, `tab:prefix_significance`, `tab:confusion` |
> | 15-object 1K stress test | `results/15obj_1k/physics_llm_15obj_1k.json` + `results/15obj_1k/{llama,qwen,claude,gpt4,gemini,deepseek}_15obj_*tool.json` | `fig:ci_forest_15obj` |
>
> The Grounded-Physics LM held-out primary numbers are **69.2% overall /
> 79.4% explanatory / 63.4% predictive / 63.6% counterfactual** on n=1,998
> valid items the adapter never saw during training. The 1K-pool 73.9%
> baseline is retained only as the denominator for the *Robustness across
> partitions* paragraph in `main.tex`. For the canonical reproduction
> recipe, see [`REPRODUCTION.md` §4 + §4a](../REPRODUCTION.md) (held-out
> primary, 3--6 obj) and §7 (15-object). The Jan 25 numbers below are kept
> verbatim because earlier drafts and `FAIR_BENCHMARK_RESULTS.md`
> cross-reference them; do **not** treat them as primary.

---

## Fair Comparison Protocol

All benchmarks use **identical input conditions**:
- **Single frame (frame 64)** - middle of 128-frame simulation
- **No collision event labels** - models must infer collisions from positions/velocities
- **Same scene information** - object positions, velocities, attributes

This ensures Physics-LLM and text-based LLMs receive equivalent information.

---

## Main Results: 15-Object Fair Comparison

**Location:** `results_15obj_fair/`

| Model | Params | Overall | Explanatory | Predictive | Counterfactual |
|-------|--------|---------|-------------|------------|----------------|
| **Physics-LLM** | **82M** | **62.2%** | **63.3%** | 60.0% | **63.3%** |
| Llama-3.1-8B | 8B | 48.9% | 50.0% | **66.7%** | 30.0% |
| Llama-3.1-70B | 70B | 43.3% | 46.7% | 40.0% | 43.3% |
| Qwen2.5-72B | 72B | 33.3% | 26.7% | 43.3% | 30.0% |
| DeepSeek-V3 | 671B | 32.2% | 13.3% | 63.3% | 20.0% |

### Key Findings

1. **Physics-LLM (82M) outperforms all LLMs up to 671B** by +13-30 percentage points overall
2. **Counterfactual reasoning**: Physics-LLM (63.3%) dominates - LLMs all under 45%
3. **Inverse scaling**: Larger LLMs (70B+) perform *worse* than smaller ones on complex scenes
4. **Random chance = 25%** for 4-choice MCQ

---

## Physics-LLM Object Scaling

**Locations:** `results_6obj/`, `results_10obj/`, `results_15obj_fair/`, `results_20obj/`

Tests Physics-LLM on CLEVRER questions with scenes augmented to different object counts.
All use frame 64 only.

| Objects | Overall | Explanatory | Predictive | Counterfactual |
|---------|---------|-------------|------------|----------------|
| 6 (training dist) | 53.3% | 40.0% | 80.0% | 40.0% |
| 10 | 63.3% | 60.0% | 80.0% | 50.0% |
| 15 | 62.2% | 63.3% | 60.0% | 63.3% |
| 20 | 63.3% | 60.0% | 60.0% | 70.0% |

**Finding:** Physics-LLM maintains ~60% accuracy even with 3x more objects than training (3-6 objects).

---

## Synthetic 20-Object Benchmark

**Location:** `results_20obj_synthetic/`

Synthetic collision chain scenarios with 20 objects. Tests TRUE physics scaling where all objects participate in causal chains (not just distractors).

| Model | Overall | Explanatory | Predictive | Counterfactual |
|-------|---------|-------------|------------|----------------|
| DeepSeek-V3 | 58.9% | 60.0% | 16.7% | 100.0% |
| Llama-3.1-70B | 56.7% | 63.3% | 16.7% | 90.0% |
| Qwen2.5-72B | 53.3% | 43.3% | 16.7% | 100.0% |
| Physics-LLM | 10.0% | 30.0% | 0.0% | 0.0% |
| Llama-3.1-8B | 0.0% | 0.0% | 0.0% | 0.0% |

**Key Finding:**
- All models fail predictive questions (16.7% or less) - cannot infer collisions from positions
- Large LLMs use heuristics for counterfactual (abstract object removal reasoning)
- Physics-LLM fails on novel synthetic scenes (outside CLEVRER training distribution)

---

## Important Distinctions

### Augmented CLEVRER vs Synthetic Scenes

| Aspect | Augmented CLEVRER | Synthetic Scenes |
|--------|-------------------|------------------|
| Questions | Real CLEVRER validation | Generated questions |
| Core physics | Original causal chains | New collision chains |
| Added objects | Distractors (don't affect answers) | All participate |
| Physics-LLM | 62.2% (handles distractors) | 10% (out of distribution) |
| Tests | Attention + known physics | Novel physics reasoning |

---

## File Organization

```
clevrer_benchmark/
├── results_15obj_fair/         # MAIN RESULTS - fair comparison
│   ├── physics_llm_15obj.json
│   ├── llama-70b_15obj.json
│   ├── llama-8b_15obj.json
│   ├── qwen-72b_15obj.json
│   └── deepseek-v3_15obj.json
├── results_6obj/               # Physics-LLM scaling
├── results_10obj/              # Physics-LLM scaling
├── results_20obj/              # Physics-LLM scaling
├── results_20obj_synthetic/    # Synthetic benchmark
├── benchmark_15_objects.py           # LLM benchmark (fair)
├── benchmark_15_objects_physics_llm.py  # Physics-LLM (fair, frame 64)
└── benchmark_15_objects_synthetic.py    # Synthetic benchmark
```

---

## Reproduction Commands

> **Note:** the commands below produce the legacy n=90 stratified results.
> For the **current 1K-pool** numbers reported in the article, see
> [`REPRODUCTION.md` §7](../REPRODUCTION.md) which documents the
> `--num_questions 1000 --uniform --filter_malformed --shuffle_choices`
> recipe.

### Fair 15-Object Benchmark (LEGACY, n=90 stratified)
```bash
# Physics-LLM (uses frame 64 only)
python benchmark_15_objects_physics_llm.py \
    --adapter_checkpoint ../checkpoints/adapter_phase3.pt \
    --target_objects 15 \
    --output_dir results_15obj_fair

# LLMs (no collision hints)
python benchmark_15_objects.py \
    --model llama-70b \
    --output_dir results_15obj_fair \
    --api_key YOUR_KEY
```

### Fair 15-Object Benchmark (CURRENT, 1K pool)
```bash
# Physics-LLM, matched 1K pool
python benchmark_15_objects_physics_llm.py \
    --adapter_checkpoint ../checkpoints/adapter_phase3.pt \
    --clevrer_dir $CLEVRER_DIR \
    --num_questions 1000 --uniform --filter_malformed --shuffle_choices \
    --save_details \
    --output_dir results/15obj_1k --output_name physics_llm_15obj_1k
```

### Synthetic Benchmark
```bash
python benchmark_15_objects_synthetic.py \
    --model qwen-72b \
    --num_questions 90 \
    --api_key YOUR_KEY
```

---

## Changelog

- **2026-04-27**: Superseded by 1K-pool protocol.
  - Re-evaluated Grounded-Physics LM and all LLM baselines on the matched
    1{,}000-question uniformly-sampled pool for both 3–6 obj and 15-obj.
  - Added Wilson 95% CIs and forest plots (`fig:ci_forest`,
    `fig:ci_forest_15obj`) for primary comparison tables.
  - Article and `REPRODUCTION.md` updated to use 1K pool as primary.
  - This file kept verbatim for historical traceability; current numbers
    live in the `15obj_1k/` subdirectory and the article tables.
- **2025-01-25**: Established fair comparison protocol
  - Removed collision hints from all LLM benchmarks
  - Modified Physics-LLM to use frame 64 only (matching LLM input)
  - Reran all benchmarks with fair conditions
