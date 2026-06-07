# ComPhy zero-shot OOD evaluation

Zero-shot transfer of the CLEVRER-trained Phase 3 adapter to **ComPhy**
(Chen et al., ICLR 2022 — Compositional Physical Reasoning of Objects and
Events from Videos). No retraining, no fine-tuning: the same
`checkpoints/adapter_phase3.pt` that produced the 79.6% CLEVRER SOTA is
evaluated on ComPhy validation questions.

The purpose of this experiment is to show three things in one run:

1. **Architecture transfers** — the 35-D state schema, the encoder, and
   the prefix-tuned LLM all consume ComPhy scenes without modification.
2. **Grounding signal is not CLEVRER-specific** — the encoder produces
   useful prefix tokens on a different physics-reasoning dataset.
3. **Predictive advantage persists outside CLEVRER** — the model's lead
   on predictive questions (the hardest CLEVRER category) carries over.

## What transfers cleanly, what does not

| Channel                  | CLEVRER source       | ComPhy source                  | 35-D slot    |
|--------------------------|----------------------|--------------------------------|--------------|
| Position                 | trajectory[t].location | trajectory[t].objects[i].location | `state[0:3]` |
| Velocity                 | derived              | trajectory[t].objects[i].velocity | `state[3:6]` |
| Orientation (quaternion) | identity quat        | Euler -> quat from `orientation`  | `state[6:10]` |
| Angular velocity         | zeroed               | trajectory[t].objects[i].angular_velocity | `state[10:13]` |
| **Mass (hidden)**        | material default     | object_property[i].mass (1 / 5)   | `state[13]` |
| Radius                   | constant 0.3         | object_property[i].scale          | `state[14]` |
| Color RGB / shape        | static               | object_property[i].color / shape  | `state[15:18]`, `state[18]` |
| Friction                 | material default     | object_property[i].lateral_friction | `state[20]` |
| Inside-scene flag        | (always true)        | trajectory[t].objects[i].inside_scene | `state[21]` + masks |
| Restitution              | material default     | object_property[i].restitution    | `state[34]` |
| **Charge (hidden)**      | n/a                  | object_property[i].charge         | **NOT REPRESENTED** |

**Charge is an honest architectural limitation.** The 35-D state schema
has no charge slot, so questions about magnetic attraction/repulsion are
answered with charge invisible to the model. The stats script reports a
charge-dependence slice via a keyword filter on the question text so the
limitation is disclosed in any reported number.

## Data layout

The ComPhy release lives **outside the workspace tree** by default
(`D:\comphy` on this machine — 10k annotation JSONs + 8 QA chunks total
~2.6 GB). Keeping it out of the repo avoids polluting VS Code, search,
and file enumerations with 10k+ files. The runner default
(`DEFAULT_COMPHY_DIR`) is `D:/comphy`; pass `--comphy_dir` or set the
`COMPHY_DIR` env var to override.

```
D:\comphy\
├── target_annotation\                  # one annotation per scene (10k total)
│   ├── annotation_00000_01000\<00000..00999>.json
│   ├── annotation_01000_02000\<01000..01999>.json
│   └── ...
├── qa_chunk0.json ... qa_chunk7.json   # validation QA, 1000 scenes each
├── train_qa.json                       # training QA (NOT used by default)
└── train_qa_chunk0..3.json             # training QA chunks (NOT used by default)
```

If you ever need to re-populate this directory: the source is
<https://comphyreasoning.github.io/>. The `.gitignore` keeps any future
in-repo copy under `benchmark_data/comphy/` untracked so you can
temporarily land data in the workspace without polluting git.

Validation distribution (across all 8 chunks ≈ 8000 scenes, ≈ 43k questions):

| Native type                       | Coarse type   | Approx share |
|-----------------------------------|---------------|--------------|
| factual (no `question_type`)      | factual       | ~53% |
| counterfactual_multiple_choice    | counterfactual| ~37% |
| predictive_multiple_choice        | predictive    | ~10% |

Factual is open-ended (single-word/phrase answer); the two MC families
each have 2 choices labelled `correct`/`wrong`.

## Running the eval

Zero-config (data path is the in-repo default):

```powershell
python comphy_benchmark\run_comphy_evaluation.py `
    --adapter_checkpoint checkpoints\adapter_phase3.pt `
    --output comphy_benchmark\results\phase3_comphy_zeroshot.json `
    --save_details
```

Defaults match the CLEVRER SOTA protocol:
`--eval_method generate --gen_mode sample --gen_seed 42 --single_frame 64`.

For a ~3-minute smoke test on a single GPU, add `--max_scenes 50`.

Useful flags:

- `--skip_factual` — focus on predictive + counterfactual MC (mirrors
  CLEVRER's `--skip_descriptive`).
- `--only predictive,counterfactual` — same effect with explicit list.
- `--zero_physics` — text-only ablation on ComPhy.
- `--shuffle_choices` — positional-bias control.
- `--qa_files benchmark_data/comphy/qa_chunk0.json` — restrict to one chunk
  (~5k questions, ~25% the time of the full val).

## Stats / paper table

```powershell
python comphy_benchmark\scripts\compute_comphy_stats.py `
    --result comphy_benchmark\results\phase3_comphy_zeroshot.json `
    --emit_latex
```

Emits:

- Wilson 95% CIs for overall + coarse types (factual / predictive /
  counterfactual) + ComPhy native subtypes
- Charge-dependence slice computed from a keyword filter on the question
  text (`charge|attract|repel|magnet`) — honest disclosure of how much
  accuracy comes from questions the partial-observability model can
  fully see
- A LaTeX table fragment (`tab:comphy_ood`) ready to paste alongside
  `tab:significance` and `fig:ci_forest` in `main.tex`

## Suggested ablation set

Once the baseline number is in, run two ablations on the same ComPhy
question pool to make the cross-benchmark claim defensible:

1. `--zero_physics` — text-only baseline on ComPhy. Mirrors the CLEVRER
   zero-physics ablation; expects a large drop, confirms the adapter is
   using the physics prefix on OOD data too.
2. `--shuffle_choices` — positional-bias control. Mirrors the
   `BASELINE_SHUFFLE` CLEVRER protocol; any persisting advantage is not
   from MCQ ordering.

Together with the baseline, these three runs are the minimal set that
make the OOD transfer claim defensible.

