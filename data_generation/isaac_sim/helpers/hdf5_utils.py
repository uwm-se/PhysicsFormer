"""
Copyright (c) 2026 Style Machine LLC. All rights reserved.

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Style Machine LLC.
"""

"""
HDF5 Utilities for Isaac Sim Data Generation

Provides standardized HDF5 file creation and writing for physics training data.
"""

import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from .state_extraction import MAX_OBJECTS, STATE_DIM


@dataclass
class HDF5Config:
    max_seq_length: int = 512
    max_objects: int = MAX_OBJECTS
    state_dim: int = STATE_DIM
    chunk_size: int = 50
    compression: str = 'gzip'
    compression_opts: int = 4


def create_physics_hdf5(
    output_path: Path,
    num_episodes: int,
    config: Optional[HDF5Config] = None
) -> h5py.File:
    if config is None:
        config = HDF5Config()
    
    f = h5py.File(output_path, 'w')
    
    chunk_episodes = min(config.chunk_size, num_episodes)
    
    f.create_dataset(
        'states',
        shape=(num_episodes, config.max_seq_length, config.max_objects, config.state_dim),
        dtype='float32',
        chunks=(chunk_episodes, config.max_seq_length, config.max_objects, config.state_dim),
        compression=config.compression,
        compression_opts=config.compression_opts
    )
    
    f.create_dataset(
        'masks',
        shape=(num_episodes, config.max_seq_length, config.max_objects),
        dtype='float32',
        chunks=(chunk_episodes, config.max_seq_length, config.max_objects),
        compression=config.compression,
        compression_opts=config.compression_opts
    )
    
    f.create_dataset(
        'next_states',
        shape=(num_episodes, config.max_seq_length - 1, config.max_objects, config.state_dim),
        dtype='float32',
        chunks=(chunk_episodes, config.max_seq_length - 1, config.max_objects, config.state_dim),
        compression=config.compression,
        compression_opts=config.compression_opts
    )
    
    dt = h5py.special_dtype(vlen=str)
    f.create_dataset('schemas', shape=(num_episodes,), dtype=dt)
    
    f.create_dataset(
        'sequence_lengths',
        shape=(num_episodes,),
        dtype='int32'
    )
    
    f.attrs['max_seq_length'] = config.max_seq_length
    f.attrs['max_objects'] = config.max_objects
    f.attrs['state_dim'] = config.state_dim
    
    return f


def write_episode_batch(
    hdf5_file: h5py.File,
    start_idx: int,
    states_batch: list,
    masks_batch: list,
    schema_name: str,
    sequence_lengths: Optional[list] = None
):
    num_episodes = len(states_batch)
    
    for i, (states, masks) in enumerate(zip(states_batch, masks_batch)):
        ep_idx = start_idx + i
        hdf5_file['states'][ep_idx] = states
        hdf5_file['masks'][ep_idx] = masks
        hdf5_file['next_states'][ep_idx] = states[1:, :, :]
        hdf5_file['schemas'][ep_idx] = schema_name
        
        if sequence_lengths is not None:
            hdf5_file['sequence_lengths'][ep_idx] = sequence_lengths[i]
        else:
            valid_steps = np.sum(masks.sum(axis=1) > 0)
            hdf5_file['sequence_lengths'][ep_idx] = int(valid_steps)


def append_to_hdf5(
    existing_path: Path,
    new_states: np.ndarray,
    new_masks: np.ndarray,
    schema_name: str
):
    with h5py.File(existing_path, 'a') as f:
        current_size = f['states'].shape[0]
        new_size = current_size + len(new_states)
        
        for key in ['states', 'masks', 'next_states', 'schemas', 'sequence_lengths']:
            f[key].resize(new_size, axis=0)
        
        for i, (states, masks) in enumerate(zip(new_states, new_masks)):
            ep_idx = current_size + i
            f['states'][ep_idx] = states
            f['masks'][ep_idx] = masks
            f['next_states'][ep_idx] = states[1:, :, :]
            f['schemas'][ep_idx] = schema_name
            f['sequence_lengths'][ep_idx] = int(np.sum(masks.sum(axis=1) > 0))
