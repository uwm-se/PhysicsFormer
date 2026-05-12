"""
Copyright (c) 2026 Style Machine LLC. All rights reserved.

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Style Machine LLC.
"""

"""Convert large CLEVRER JSON to compact HDF5 format for efficient training."""

import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc


def convert_json_to_hdf5(json_path: Path, hdf5_path: Path, chunk_size: int = 1000):
    """Stream-convert large JSON to HDF5 format."""
    print(f"Converting {json_path} to {hdf5_path}...")
    
    # First pass: count samples and scan ALL for max object count
    print("Pass 1: Counting samples and scanning object counts...")
    with open(json_path, 'r') as f:
        f.readline()  # Skip opening bracket
        sample_count = 0
        first_sample = None
        max_objects_seen = 0
        object_count_hist = {}

        for line in f:
            line = line.strip()
            if line.startswith('{'):
                if line.endswith(','):
                    line = line[:-1]
                sample = json.loads(line)
                if first_sample is None:
                    first_sample = sample
                n_obj = np.array(sample['states']).shape[1]
                max_objects_seen = max(max_objects_seen, n_obj)
                object_count_hist[n_obj] = object_count_hist.get(n_obj, 0) + 1
                sample_count += 1
                if sample_count % 10000 == 0:
                    print(f"  Scanned {sample_count} samples (max objects so far: {max_objects_seen})...")

    print(f"Total samples: {sample_count}")
    print(f"Object count distribution across all samples:")
    for n_obj in sorted(object_count_hist.keys()):
        count = object_count_hist[n_obj]
        pct = 100 * count / sample_count
        print(f"  {n_obj} objects: {count:,} samples ({pct:.1f}%)")

    # Use the TRUE max across all samples (not just the first)
    states = np.array(first_sample['states'])
    seq_len, _, state_dim = states.shape
    num_objects = max_objects_seen
    print(f"State shape: ({seq_len}, {num_objects}, {state_dim})")
    print(f"  num_objects set to {num_objects} (max across all {sample_count:,} samples — no truncation)")
    
    # Create HDF5 file with chunked datasets
    with h5py.File(hdf5_path, 'w') as hf:
        # Create datasets with chunking for efficient access
        states_ds = hf.create_dataset(
            'states', 
            shape=(sample_count, seq_len, num_objects, state_dim),
            dtype=np.float32,
            chunks=(min(100, sample_count), seq_len, num_objects, state_dim),
            compression='gzip',
            compression_opts=4
        )
        
        masks_ds = hf.create_dataset(
            'masks',
            shape=(sample_count, seq_len, num_objects),
            dtype=np.float32,
            chunks=(min(100, sample_count), seq_len, num_objects),
            compression='gzip'
        )
        
        # String data stored as variable-length
        dt = h5py.special_dtype(vlen=str)
        questions_ds = hf.create_dataset('questions', shape=(sample_count,), dtype=dt)
        answers_ds = hf.create_dataset('answers', shape=(sample_count,), dtype=dt)
        question_types_ds = hf.create_dataset('question_types', shape=(sample_count,), dtype=dt)
        metadata_ds = hf.create_dataset('metadata', shape=(sample_count,), dtype=dt)
        
        # Numerical targets
        num_targets_ds = hf.create_dataset(
            'numerical_targets',
            shape=(sample_count, 2),
            dtype=np.float32,
            chunks=(min(1000, sample_count), 2),
            compression='gzip'
        )
        
        # Second pass: write data (padding only — no truncation possible)
        print("Pass 2: Writing data...")
        padded_count = 0
        with open(json_path, 'r') as f:
            f.readline()  # Skip opening bracket
            idx = 0

            for line in tqdm(f, total=sample_count, desc="Converting"):
                line = line.strip()
                if not line.startswith('{'):
                    continue
                if line.endswith(','):
                    line = line[:-1]

                try:
                    sample = json.loads(line)

                    states = np.array(sample['states'], dtype=np.float32)
                    mask = np.array(sample['mask'], dtype=np.float32)

                    # Pad smaller scenes up to max (zero-padded, mask=0)
                    if states.shape[1] < num_objects:
                        pad_objs = num_objects - states.shape[1]
                        states = np.pad(states, ((0,0), (0,pad_objs), (0,0)))
                        mask = np.pad(mask, ((0,0), (0,pad_objs)))
                        padded_count += 1

                    states_ds[idx] = states
                    masks_ds[idx] = mask
                    questions_ds[idx] = sample['question']
                    answers_ds[idx] = sample['answer']
                    question_types_ds[idx] = sample['question_type']
                    metadata_ds[idx] = json.dumps(sample['metadata'])

                    num_targets = sample.get('numerical_targets', {})
                    num_targets_ds[idx] = [
                        num_targets.get('count', 0.0),
                        num_targets.get('value', 0.0)
                    ]

                    idx += 1

                    if idx % 10000 == 0:
                        gc.collect()

                except json.JSONDecodeError as e:
                    print(f"  Skipping malformed line at {idx}: {e}")
                    continue

        print(f"  Padded {padded_count:,} samples (had fewer than {num_objects} objects)")
        
        # Store metadata
        hf.attrs['num_samples'] = sample_count
        hf.attrs['seq_len'] = seq_len
        hf.attrs['num_objects'] = num_objects
        hf.attrs['state_dim'] = state_dim
    
    # Check file size
    size_mb = hdf5_path.stat().st_size / (1024 * 1024)
    print(f"\nDone! HDF5 file size: {size_mb:.1f} MB")
    print(f"Compression ratio: {json_path.stat().st_size / hdf5_path.stat().st_size:.1f}x")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file')
    
    args = parser.parse_args()
    
    convert_json_to_hdf5(Path(args.input), Path(args.output))
