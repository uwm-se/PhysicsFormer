"""
Copyright (c) 2026 Style Machine LLC. All rights reserved.

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Style Machine LLC.
"""

"""
Standalone QA Dataset Generator

Generates QA pairs from physics HDF5 data for adapter training.
Output format: HDF5 for reproducibility and cross-platform compatibility.

Usage:
    python generate_qa_dataset.py --data_dir $PHYSICS_DATA_DIR --output adapter_qa_cache.h5 --num_samples 10000
"""

import argparse
import json
import sys
from pathlib import Path
import random
import numpy as np
import torch
import h5py
from tqdm import tqdm

from qa_generator import PhysicsQAGenerator, QuestionType


# Question types for training (no metaphors)
TRAINING_QUESTION_TYPES = [
    # Category 1: Basic Physical Properties
    QuestionType.OBJECT_COUNT,
    QuestionType.OBJECT_POSITION,
    QuestionType.OBJECT_VELOCITY,
    QuestionType.OBJECT_MASS,
    QuestionType.MOTION_DIRECTION,
    # Category 2: Physical Quantities
    QuestionType.KINETIC_ENERGY,
    QuestionType.TOTAL_MOMENTUM,
    QuestionType.RELATIVE_VELOCITY,
    QuestionType.SPATIAL_DISTANCE,
    QuestionType.SPEED_COMPARISON,
    QuestionType.MASS_COMPARISON,
    # Category 3: Predictive Reasoning
    QuestionType.COLLISION_PREDICTION,
    QuestionType.TRAJECTORY_EXTRAPOLATION,
    QuestionType.TIME_TO_EVENT,
    QuestionType.REACHABILITY,
    QuestionType.PATH_OBSTRUCTION,
    # Category 4: Relational Reasoning
    QuestionType.PROXIMITY,
    QuestionType.SPATIAL_CONTAINMENT,
    QuestionType.RELATIVE_POSITION,
    QuestionType.CONTACT_STATE,
    # Category 5: CLEVRER-Style Reasoning
    QuestionType.CAUSAL_CHAIN,
    QuestionType.FUTURE_PREDICTION,
    QuestionType.COUNTERFACTUAL_REASONING,
]


def load_physics_states_from_hdf5(data_dir: str, max_samples: int, max_objects: int = 20, state_dim: int = 28):
    """Load physics states from HDF5 files."""
    print("\n" + "="*60)
    print("STEP 1: Loading Physics States from HDF5")
    print("="*60)
    
    data_path = Path(data_dir)
    h5_files = list(data_path.glob("*.h5"))
    
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    
    print(f"  Found {len(h5_files)} HDF5 files")
    print(f"  Target samples: {max_samples}")
    print(f"  Max objects: {max_objects}, State dim: {state_dim}")
    samples = []
    samples_per_file = max(100, max_samples // len(h5_files))
    print(f"  Samples per file: ~{samples_per_file}")
    
    files_processed = 0
    files_with_errors = 0
    
    for h5_file in tqdm(h5_files, desc="Loading HDF5 files"):
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'states' in f:
                    states = f['states'][:]
                elif 'state' in f:
                    states = f['state'][:]
                else:
                    continue
                
                n_samples = min(len(states), samples_per_file)
                indices = random.sample(range(len(states)), n_samples)
                
                for idx in indices:
                    state = states[idx]
                    
                    # Handle different state formats
                    if state.ndim == 1:
                        state = state.reshape(max_objects, -1)
                    elif state.ndim == 3:
                        state = state[0]  # Take first timestep
                    
                    # Pad/truncate objects
                    if state.shape[0] < max_objects:
                        pad = np.zeros((max_objects - state.shape[0], state.shape[1]))
                        state = np.concatenate([state, pad], axis=0)
                    elif state.shape[0] > max_objects:
                        state = state[:max_objects]
                    
                    # Handle 35D -> 28D conversion
                    if state.shape[1] >= 35:
                        state = np.concatenate([state[:, 0:6], state[:, 13:]], axis=1)
                    
                    # Pad/truncate state_dim
                    if state.shape[1] < state_dim:
                        pad = np.zeros((state.shape[0], state_dim - state.shape[1]))
                        state = np.concatenate([state, pad], axis=1)
                    elif state.shape[1] > state_dim:
                        state = state[:, :state_dim]
                    
                    mask = np.any(state != 0, axis=1).astype(np.float32)
                    
                    samples.append({
                        'state': torch.tensor(state, dtype=torch.float32),
                        'mask': torch.tensor(mask, dtype=torch.float32)
                    })
                    
                    if len(samples) >= max_samples:
                        break
                        
            files_processed += 1
                        
        except Exception as e:
            files_with_errors += 1
            print(f"\n  ⚠️  Error loading {h5_file.name}: {e}")
        
        if len(samples) >= max_samples:
            break
    
    print(f"\n  ✓ Loaded {len(samples):,} physics states")
    print(f"  ✓ Files processed: {files_processed}/{len(h5_files)}")
    if files_with_errors > 0:
        print(f"  ⚠️  Files with errors: {files_with_errors}")
    return samples


def generate_qa_dataset(physics_samples, num_qa_per_sample: int = 3, seed: int = 42):
    """Generate QA pairs from physics samples."""
    print("\n" + "="*60)
    print("STEP 2: Generating QA Pairs")
    print("="*60)
    print(f"  Physics samples: {len(physics_samples):,}")
    print(f"  QA pairs per sample: {num_qa_per_sample}")
    print(f"  Expected total: ~{len(physics_samples) * num_qa_per_sample:,} QA pairs")
    print(f"  Question types: {len(TRAINING_QUESTION_TYPES)}")
    
    qa_generator = PhysicsQAGenerator(
        question_types=TRAINING_QUESTION_TYPES,
        seed=seed
    )
    
    qa_dataset = []
    failed_generations = 0
    question_type_counts = {}
    
    for sample in tqdm(physics_samples, desc="Generating QA pairs"):
        state = sample['state']
        mask = sample['mask']
        
        for _ in range(num_qa_per_sample):
            try:
                qa_pair = qa_generator.generate_qa_pair(state, mask)
                
                # Extract numerical targets if available
                numerical_targets = {}
                if 'kinetic_energy' in qa_pair.metadata:
                    numerical_targets['kinetic_energy'] = qa_pair.metadata['kinetic_energy']
                if 'momentum_magnitude' in qa_pair.metadata:
                    numerical_targets['momentum'] = qa_pair.metadata['momentum_magnitude']
                if 'distance' in qa_pair.metadata:
                    numerical_targets['distance'] = qa_pair.metadata['distance']
                if 'count' in qa_pair.metadata:
                    numerical_targets['object_count'] = float(qa_pair.metadata['count'])
                
                qa_dataset.append({
                    'states': state,
                    'mask': mask,
                    'question': qa_pair.question,
                    'answer': qa_pair.answer,
                    'question_type': qa_pair.question_type.value,
                    'metadata': qa_pair.metadata,
                    'numerical_targets': numerical_targets if numerical_targets else None,
                })
                
                # Track question type distribution
                qt = qa_pair.question_type.value
                question_type_counts[qt] = question_type_counts.get(qt, 0) + 1
                
            except Exception as e:
                failed_generations += 1
                continue
    
    print(f"\n  ✓ Generated {len(qa_dataset):,} QA pairs")
    if failed_generations > 0:
        print(f"  ⚠️  Failed generations: {failed_generations}")
    
    # Show question type distribution
    print("\n  Question Type Distribution:")
    for qt, count in sorted(question_type_counts.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / len(qa_dataset) if qa_dataset else 0
        print(f"    {qt}: {count:,} ({pct:.1f}%)")
    if len(question_type_counts) > 10:
        print(f"    ... and {len(question_type_counts) - 10} more types")
    
    return qa_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate QA dataset for adapter training")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing HDF5 physics data")
    parser.add_argument("--output", type=str, default="adapter_qa_cache.h5", help="Output file path (HDF5 format)")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of physics samples to load")
    parser.add_argument("--qa_per_sample", type=int, default=3, help="QA pairs per physics sample")
    parser.add_argument("--max_objects", type=int, default=20, help="Maximum objects per scene")
    parser.add_argument("--state_dim", type=int, default=28, help="State dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("QA Dataset Generator")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output file: {args.output}")
    print(f"Num samples: {args.num_samples}")
    print(f"QA per sample: {args.qa_per_sample}")
    print(f"Question types: {len(TRAINING_QUESTION_TYPES)}")
    print("=" * 60)
    
    # Load physics states
    physics_samples = load_physics_states_from_hdf5(
        args.data_dir,
        max_samples=args.num_samples,
        max_objects=args.max_objects,
        state_dim=args.state_dim
    )
    
    # Generate QA pairs
    qa_dataset = generate_qa_dataset(
        physics_samples,
        num_qa_per_sample=args.qa_per_sample,
        seed=args.seed
    )
    
    # Save dataset to HDF5
    print("\n" + "="*60)
    print("STEP 3: Saving Dataset to HDF5")
    print("="*60)

    output_path = Path(args.output)
    if output_path.suffix == '.pt':
        output_path = output_path.with_suffix('.h5')
    print(f"  Saving to: {output_path}")

    n_samples = len(qa_dataset)
    if n_samples == 0:
        print("ERROR: No QA pairs were generated. Check your data directory.")
        return

    state_shape = qa_dataset[0]['states'].shape
    mask_shape = qa_dataset[0]['mask'].shape

    with h5py.File(output_path, 'w') as f:
        # Create tensor datasets
        states_ds = f.create_dataset(
            'states', shape=(n_samples,) + state_shape,
            dtype='float32', compression='gzip'
        )
        masks_ds = f.create_dataset(
            'masks', shape=(n_samples,) + mask_shape,
            dtype='float32', compression='gzip'
        )

        # String datasets
        dt = h5py.special_dtype(vlen=str)
        questions_ds = f.create_dataset('questions', shape=(n_samples,), dtype=dt)
        answers_ds = f.create_dataset('answers', shape=(n_samples,), dtype=dt)
        question_types_ds = f.create_dataset('question_types', shape=(n_samples,), dtype=dt)
        metadata_ds = f.create_dataset('metadata', shape=(n_samples,), dtype=dt)
        numerical_targets_ds = f.create_dataset('numerical_targets', shape=(n_samples,), dtype=dt)

        # Fill datasets
        for i, sample in enumerate(tqdm(qa_dataset, desc="Writing HDF5")):
            states_ds[i] = sample['states'].numpy()
            masks_ds[i] = sample['mask'].numpy()
            questions_ds[i] = sample['question']
            answers_ds[i] = sample['answer']
            question_types_ds[i] = sample['question_type']
            metadata_ds[i] = json.dumps(sample['metadata'])
            numerical_targets_ds[i] = json.dumps(sample['numerical_targets']) if sample['numerical_targets'] else '{}'

        # Metadata
        f.attrs['num_samples'] = n_samples
        f.attrs['state_shape'] = state_shape
        f.attrs['num_question_types'] = len(TRAINING_QUESTION_TYPES)
        f.attrs['format_version'] = '1.0'

    file_size = output_path.stat().st_size
    print(f"\n  Saved {len(qa_dataset):,} QA pairs")
    print(f"  File size: {file_size / 1e6:.1f} MB")
    
    # Final summary
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"  Total QA pairs: {len(qa_dataset):,}")
    print(f"  Output file: {output_path}")
    print(f"  Ready for adapter training!")
    print("="*60)


if __name__ == "__main__":
    main()
