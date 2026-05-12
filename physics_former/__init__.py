"""
physics_former

Subset of the original physics_former package, retaining everything
under `training/` needed to:
  (a) pretrain FullPhysicsFormer from Isaac Sim HDF5 data (Stage 1),
  (b) load FullPhysicsFormer weights embedded in the adapter
      checkpoint for evaluation.

Not included in this snapshot: visualization, evaluation dashboards,
and experimental branches outside `training/`.
"""
