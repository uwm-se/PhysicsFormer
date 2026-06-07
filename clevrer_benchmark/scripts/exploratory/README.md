# Exploratory scripts (not paper-reproduction entry points)

These scripts run but back **no claim** in the paper. They are kept for
the record, not for reproduction. The canonical reproduction scripts and
their claim mapping are in the repo-root `REPRODUCTION.md` (§12).

| Script | What it is | Why it is not canonical |
|---|---|---|
| `physicsformer_oracle.py` | Rolls PhysicsFormer forward and compares to a constant-velocity (ballistic) baseline. | The paper's stand-in justification rests on the IPEv2 `-1%` absolute-feature result in `../ipe_v2_transfer.py`, not on an oracle number. |
| `ipe_collision_oracle.py` | Same comparison for the IPE collision predictor. | Supporting diagnostic; no cited number. |
| `ipe_v2_charge.py` | Meta-trains IPEv2 with the charge force-prior. | Charge is out of scope for the paper. |
| `dynamics_probe.py` | Probes the adapter prefix for dynamics channels. | Exploratory; superseded by `../heldout_channel_ablation.py`. |

Each resolves `CLEVRER_H5` / `ADAPTER_CKPT` from environment variables
(repo-relative defaults), same as the canonical scripts, and accepts
`--device cpu`. They import their dependencies (`ipe_v2_transfer`,
`intuitive_physics_engine`, `run_adapter_evaluation`) from the parent
`scripts/` package, so run them from the repo root.
