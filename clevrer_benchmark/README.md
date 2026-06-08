# CLEVRER Benchmark Evaluation for Physics-LLM

This module evaluates Physics-LLM on the [CLEVRER benchmark](http://clevrer.csail.mit.edu/), focusing on **Explanatory**, **Predictive**, and **Counterfactual** question types where physics grounding should provide the most benefit.

## Why CLEVRER?

CLEVRER was generated using the **Bullet physics engine** (same as PyBullet) and provides ground-truth motion traces and object state histories. This allows Physics-LLM to be evaluated using state tensors rather than video frames, maintaining the architectural claim while evaluating on an established benchmark.

## Question Types

| CLEVRER Type | Physics-LLM Mapping | Description |
|--------------|---------------------|-------------|
| Descriptive | N/A (skipped) | Tests perception, not physics |
| **Explanatory** | causal_attribution | "What caused the collision?" |
| **Predictive** | will_collide, trajectory | "Will X collide with Y?" |
| **Counterfactual** | counterfactual | "What if X were removed?" |

## Installation

```bash
# From project root
pip install -r requirements.txt
```

## Dataset Setup

1. Download CLEVRER from http://clevrer.csail.mit.edu/

2. Required files:
   - Validation set scenes (JSON files with object trajectories)
   - Validation set questions (questions and answers)

3. Expected directory structure:
```
clevrer/
├── scenes/
│   ├── validation/
│   │   ├── sim_00000.json
│   │   ├── sim_00001.json
│   │   └── ...
└── questions/
    └── validation.json
```

## Usage

### Full Evaluation

```bash
python clevrer_benchmark/run_evaluation.py \
    --clevrer_dir /path/to/clevrer \
    --checkpoint /path/to/physics_llm_checkpoint.pt \
    --split validation \
    --output results/clevrer_results.json
```

### Quick Evaluation (100 scenes)

```bash
python clevrer_benchmark/run_evaluation.py \
    --clevrer_dir /path/to/clevrer \
    --checkpoint /path/to/checkpoint.pt \
    --quick
```

### Compare with Baseline

```bash
python clevrer_benchmark/run_evaluation.py \
    --clevrer_dir /path/to/clevrer \
    --checkpoint /path/to/checkpoint.pt \
    --output physics_llm_results.json \
    --compare baseline_results.json
```

## Module Structure

```
clevrer_benchmark/
├── __init__.py              # Package init
├── scene_converter.py       # CLEVRER scene → Physics-LLM state tensor
├── question_mapper.py       # CLEVRER question → Physics-LLM format
├── evaluator.py             # Main evaluation logic
├── run_evaluation.py        # CLI entry point
└── README.md                # This file
```

## State Tensor Format

CLEVRER scenes are converted to Physics-LLM's 35-dimensional state vector:

| Index | Field | Description |
|-------|-------|-------------|
| 0-2 | Position | x, y, z coordinates |
| 3-5 | Velocity | vx, vy, vz (derived from position deltas) |
| 6-9 | Orientation | Quaternion (identity default) |
| 10-12 | Angular velocity | wx, wy, wz (zeros - not in CLEVRER) |
| 13-15 | Bounding box | width, height, depth |
| 16 | Mass | Based on material (rubber=1.0, metal=2.0) |
| 17 | Friction | Based on material |
| 18 | Restitution | Based on material |
| 19-21 | Color | RGB values |
| 22 | Object type | Shape encoding (sphere=0, cube=1, cylinder=2) |
| 23-34 | Reserved | Padding |

## Expected Results

Based on CLEVRER leaderboard patterns:

| Model | Explanatory | Predictive | Counterfactual |
|-------|-------------|------------|----------------|
| Neuro-symbolic (SOTA) | ~80% | ~80% | ~70% |
| GPT-4 (text only) | ~40%* | ~35%* | ~25%* |
| **Physics-LLM (target)** | **85-95%** | **90-97%** | **75-90%** |

*Estimated based on literature patterns

## Output Format

Results are saved as JSON:

```json
{
  "overall": {
    "total_questions": 1500,
    "correct": 1350,
    "accuracy": 0.90,
    "mean_score": 0.92
  },
  "by_clevrer_type": {
    "explanatory": {"total": 500, "correct": 450, "accuracy": 0.90},
    "predictive": {"total": 500, "correct": 475, "accuracy": 0.95},
    "counterfactual": {"total": 500, "correct": 400, "accuracy": 0.80}
  },
  "individual_results": [...]
}
```

## Counterfactual Handling

For "What if X were removed?" questions:

1. Remove object X from state tensor (zero out state and mask)
2. Run Physics-LLM forward prediction
3. Compare collision events between original and counterfactual
4. Generate answer based on differences

## Citation

If using this benchmark evaluation:

```bibtex
@inproceedings{yi2020clevrer,
  title={CLEVRER: Collision Events for Video Representation and Reasoning},
  author={Yi, Kexin and Gan, Chuang and Li, Yunzhu and Kohli, Pushmeet and Wu, Jiajun and Torralba, Antonio and Tenenbaum, Joshua B},
  booktitle={ICLR},
  year={2020}
}
```
