"""
Dataset loaders for Stage-1 physics training.

Only the HDF5 and RAM-cached Isaac-Sim physics datasets are included
in this snapshot. Counting / arithmetic / symbolic multi-task datasets
are not part of the code path that produced `physics_former_best.pt`
(validation_history shows those tasks were never trained) and have
been removed for the CompSAC-2026 reproduction package.
"""

from .hdf5_physics_dataset import HDF5PhysicsDataset
from .cached_physics_dataset import CachedPhysicsDataset

__all__ = [
    'HDF5PhysicsDataset',
    'CachedPhysicsDataset',
]
