# Grounded-Physics LM — code bundle

This subfolder is a self-contained code drop accompanying the article
**"Grounded Physics Representations Enable Causal Reasoning in Language Models"**
(see [`article/main.pdf`](article/main.pdf)). It mirrors the layout of the
upstream `compsac_2026_code/` source tree, restricted to the components
the article describes and depends on. Trained weights, raw CLEVRER data,
and stored eval outputs are intentionally **not** included — only source.

## Layout

```
code/
  physics_former/        # PhysicsFormer encoder + Isaac-Sim pretraining pipeline
  physics_llm_adapter/   # 64-token prefix adapter (V2/V3) + held-out split helpers
  clevrer_benchmark/     # Evaluation harness + LLM baselines + ablations + probes
  data_generation/       # Isaac Sim episode generation + QA generators
  notebooks/             # Colab trainers used in the article
    colab_train_physics_former.ipynb   # Encoder pretraining on Isaac Sim
    colab_train_adapter.ipynb          # 3-phase adapter training on CLEVRER (DistilGPT-2)
  article/               # LaTeX source + figures
  README.md              # Upstream project README
  REPRODUCTION.md        # Step-by-step reproduction protocol
  requirements.txt       # Pinned dependencies
```

## What's referenced where

The article names these scripts and modules verbatim:

| Article reference | File |
|---|---|
| `run_llm_with_scene.py` | [`clevrer_benchmark/run_llm_with_scene.py`](clevrer_benchmark/run_llm_with_scene.py) |
| `run_predictive_supplement.py` | [`clevrer_benchmark/run_predictive_supplement.py`](clevrer_benchmark/run_predictive_supplement.py) |
| `encoder_ood_probing.py` (§V-G) | [`clevrer_benchmark/scripts/encoder_ood_probing.py`](clevrer_benchmark/scripts/encoder_ood_probing.py) |
| `wrong_answer_patterns.py` | [`clevrer_benchmark/scripts/wrong_answer_patterns.py`](clevrer_benchmark/scripts/wrong_answer_patterns.py) |
| `free_form_transfer_test.py` | [`clevrer_benchmark/scripts/free_form_transfer_test.py`](clevrer_benchmark/scripts/free_form_transfer_test.py) |
| `paraphrased_mcq_test.py` | [`clevrer_benchmark/scripts/paraphrased_mcq_test.py`](clevrer_benchmark/scripts/paraphrased_mcq_test.py) |
| `compute_paper_stats.py` (Table I/II/III) | [`clevrer_benchmark/scripts/compute_paper_stats.py`](clevrer_benchmark/scripts/compute_paper_stats.py) |
| 10% held-out partition (scene_index ∈ [14499, 14999]) | [`physics_llm_adapter/phase9_splits.py`](physics_llm_adapter/phase9_splits.py) |
| Architecture (Fig. 1) | [`clevrer_benchmark/scripts/plot_architecture_diagram.py`](clevrer_benchmark/scripts/plot_architecture_diagram.py) |
| CI forest plots (Fig. 3, Fig. 4) | [`clevrer_benchmark/scripts/plot_ci_forest.py`](clevrer_benchmark/scripts/plot_ci_forest.py) |

The `validate_question()` filter the article cites is the inline
`valid_only=True` branch in `run_llm_with_scene.py:run_benchmark()` and
the mirrored filter in `run_adapter_evaluation.py` — both reject MCQ items
with no `correct`-labeled choice.

## What's deliberately excluded

- `data/` — the 1.7 GB H5 (`clevrer_training_expanded.h5`) used at training time.
  Regenerate from CLEVRER scenes via [`data_generation/qa_generation/`](data_generation/qa_generation/).
- `checkpoints/` — `physics_former_best.pt` (50 MB encoder) and adapter weights.
- `clevrer_benchmark/results/` — saved eval JSONs and logs.
- Post-submission notebooks: `colab_train_adapter_v3*.ipynb`,
  `colab_train_adapter_v4_qwen_lora.ipynb`, and the Phase 8/9/10 notebooks
  describe later iterations on a larger backbone (Qwen2.5-1.5B) and are
  **not** the article's Grounded-Physics LM. They are omitted here.

## Reproducing the article

See [`REPRODUCTION.md`](REPRODUCTION.md) for the end-to-end protocol.
The short version:

1. Generate Isaac Sim training data via `data_generation/isaac_sim/generate_isaac_hdf5.py`.
2. Pretrain PhysicsFormer on the H5 via `physics_former/run_physics_training.py`
   (or run `notebooks/colab_train_physics_former.ipynb`).
3. Train the 3-phase adapter on CLEVRER causal MCQ via
   `notebooks/colab_train_adapter.ipynb`. Adapter sees only the 90% train
   partition; the 10% held-out partition is the article's primary eval pool.
4. Run LLM baselines on the matched 1 000-question pool:
   `python clevrer_benchmark/run_llm_with_scene.py --valid_only`.
5. Evaluate the adapter on the 1 998-item held-out pool:
   `python clevrer_benchmark/run_adapter_evaluation.py --heldout --valid_only`.
6. Tables I-III and Fig. 3/4 are produced by
   `clevrer_benchmark/scripts/compute_paper_stats.py` and
   `clevrer_benchmark/scripts/plot_ci_forest.py`.

## Environment variables

Scripts default to portable, repo-relative paths but accept these
overrides so you can keep data and checkpoints outside the source tree:

| Variable | What it points to | Default |
|---|---|---|
| `CLEVRER_DIR` | CLEVRER root containing `scenes/`, `questions/`, `annotations/` | `clevrer` |
| `CLEVRER_H5` | Pre-converted CLEVRER training H5 (see `data_generation/clevrer/`) | `data/clevrer_training_expanded.h5` |
| `CHECKPOINT_DIR` | Directory holding `physics_former_best.pt` and adapter weights | `checkpoints` |
| `PHYSICS_DATA_DIR` | Parent of `CHECKPOINT_DIR` + Isaac Sim HDF5 shards | (no default) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | Provider keys for LLM baselines in `run_llm_with_scene.py` | (no default; missing → error) |

In docstrings and CLI examples the placeholders `$CLEVRER_DIR`,
`$CLEVRER_H5`, `$CHECKPOINT_DIR`, and `$PHYSICS_DATA_DIR` refer to these
variables.

## License

Released under the [MIT License](LICENSE). Attribution lives in the
LICENSE header and in the article (see [`article/main.pdf`](article/main.pdf)).
The source tree contains no API keys, no trained weights, no raw CLEVRER
data, and no cached notebook outputs.

## Citation

```bibtex
@inproceedings{pokora2026grounded,
  title={Grounded Physics Representations Enable Causal Reasoning in Language Models},
  author={Pokora, Jesse and Zhao, Tian},
  booktitle={Proc. IEEE Conf. on Computers, Software, and Applications (COMPSAC)},
  year={2026}
}
```
