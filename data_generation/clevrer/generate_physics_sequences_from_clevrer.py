"""
Generate physics_sequences.h5 from CLEVRER validation data.

Converts CLEVRER scene annotations to the format expected by the ablation study:
- sequences: [n_samples, seq_len, num_objects, state_dim]
- schemas: [n_samples] - physics scenario labels

Output format: HDF5 for reproducibility and cross-platform compatibility.

Copyright (c) 2026 Style Machine LLC. All rights reserved.
"""

import json
import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from clevrer_benchmark.scene_converter import (
    SHAPE_MAP, COLOR_MAP, MATERIAL_MAP, DEFAULT_SIZE, CLEVRER_FPS
)


def derive_velocity(positions: list, frame_idx: int, fps: float = CLEVRER_FPS) -> np.ndarray:
    """Derive velocity from position deltas with error handling."""
    if frame_idx == 0 or frame_idx >= len(positions):
        return np.zeros(3, dtype=np.float32)
    try:
        current_pos = np.array(positions[frame_idx], dtype=np.float32)
        prev_pos = np.array(positions[frame_idx - 1], dtype=np.float32)
        if current_pos.shape != (3,) or prev_pos.shape != (3,):
            return np.zeros(3, dtype=np.float32)
        return (current_pos - prev_pos) * fps
    except (TypeError, ValueError, IndexError):
        return np.zeros(3, dtype=np.float32)


def construct_state_vector_28d(
    position: list,
    velocity: np.ndarray,
    shape: str,
    color: str,
    material: str
) -> np.ndarray:
    """
    Construct 28D state vector for ablation study (without quaternion/angular velocity).

    State vector layout (28 dimensions):
    [0:3]   - Position (x, y, z)
    [3:6]   - Linear velocity (vx, vy, vz)
    [6]     - Mass
    [7]     - Radius
    [8:11]  - Color (r, g, b)
    [11]    - Shape type
    [12]    - Is static
    [13]    - Friction
    [14]    - Is active
    [15]    - Inertia
    [16:19] - Dimensions (width, height, depth)
    [19:22] - Force
    [22:25] - Torque
    [25]    - Restitution
    [26:28] - Reserved
    """
    state = np.zeros(28, dtype=np.float32)

    state[0:3] = position
    state[3:6] = velocity

    mat_props = MATERIAL_MAP.get(material, MATERIAL_MAP['rubber'])
    state[6] = mat_props['mass']
    state[7] = DEFAULT_SIZE
    state[8:11] = COLOR_MAP.get(color, [0.5, 0.5, 0.5])
    state[11] = SHAPE_MAP.get(shape, 0)
    state[12] = 0.0  # Is static
    state[13] = mat_props['friction']
    state[14] = 1.0  # Is active
    state[15] = 0.0  # Inertia
    state[16:19] = [DEFAULT_SIZE, DEFAULT_SIZE, DEFAULT_SIZE]
    state[25] = mat_props['restitution']

    return state


def classify_schema(scene: dict) -> int:
    """
    Classify scene into physics schema category based on collisions and object properties.

    Schema categories (0-36, matching ablation study):
    0-5: Free fall / gravity
    6-11: Single collision types
    12-17: Multi-collision scenarios
    18-23: Object property interactions (mass, friction)
    24-29: Complex dynamics (rolling, sliding)
    30-36: Edge cases and special scenarios
    """
    collisions = scene.get('collision', [])
    objects = scene.get('object_property', [])
    n_objects = len(objects)
    n_collisions = len(collisions)

    # Count materials
    metals = sum(1 for o in objects if o.get('material') == 'metal')
    rubbers = n_objects - metals

    # Count shapes
    spheres = sum(1 for o in objects if o.get('shape') == 'sphere')
    cubes = sum(1 for o in objects if o.get('shape') == 'cube')
    cylinders = sum(1 for o in objects if o.get('shape') == 'cylinder')

    # Classify based on characteristics
    if n_collisions == 0:
        # No collisions - free fall or gravity
        return min(n_objects, 5)
    elif n_collisions == 1:
        # Single collision
        if spheres >= 2:
            return 6  # Sphere-sphere collision
        elif cubes >= 2:
            return 7  # Cube-cube collision
        elif spheres >= 1 and cubes >= 1:
            return 8  # Sphere-cube collision
        elif metals >= 1 and rubbers >= 1:
            return 9  # Material interaction
        else:
            return 10
    elif n_collisions == 2:
        # Two collisions
        return 12 + min(spheres, 5)
    elif n_collisions >= 3:
        # Multiple collisions - complex dynamics
        if metals > rubbers:
            return 24 + min(n_collisions - 3, 5)
        else:
            return 30 + min(n_collisions - 3, 6)

    return 0


def process_scene(scene_path: Path, seq_len: int = 32, num_objects: int = 10) -> tuple:
    """
    Process a single CLEVRER scene file.

    Returns:
        sequence: [seq_len, num_objects, 28] tensor
        schema: int category label
    """
    with open(scene_path, 'r') as f:
        scene = json.load(f)

    objects = scene.get('object_property', [])
    trajectories = scene.get('motion_trajectory', [])

    if len(objects) == 0 or len(trajectories) == 0:
        return None, None

    # Initialize sequence tensor
    sequence = np.zeros((seq_len, num_objects, 28), dtype=np.float32)

    # Sample frames evenly across the trajectory
    n_frames = len(trajectories)
    frame_indices = np.linspace(0, n_frames - 1, seq_len, dtype=int)

    for t_idx, frame_idx in enumerate(frame_indices):
        frame = trajectories[frame_idx]
        frame_objects = frame.get('objects', [])

        for obj_idx, obj in enumerate(frame_objects):
            if obj_idx >= num_objects:
                break

            # Get object properties
            obj_props = objects[obj_idx] if obj_idx < len(objects) else {}
            shape = obj_props.get('shape', 'sphere')
            color = obj_props.get('color', 'gray')
            material = obj_props.get('material', 'rubber')

            # Get position
            position = obj.get('location', [0, 0, 0])

            # Derive velocity from position history
            all_positions = [
                trajectories[i]['objects'][obj_idx].get('location', [0, 0, 0])
                if i < len(trajectories) and obj_idx < len(trajectories[i].get('objects', []))
                else [0, 0, 0]
                for i in range(n_frames)
            ]
            velocity = derive_velocity(all_positions, frame_idx)

            # Construct state vector
            state = construct_state_vector_28d(position, velocity, shape, color, material)
            sequence[t_idx, obj_idx] = state

    # Classify schema
    schema = classify_schema(scene)

    return sequence, schema


def main():
    parser = argparse.ArgumentParser(description='Generate physics_sequences.h5 from CLEVRER')
    parser.add_argument('--clevrer-path', type=str, default='$CLEVRER_DIR/scenes/validation',
                        help='Path to CLEVRER validation scenes')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: Google Drive or local)')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of samples to generate')
    parser.add_argument('--seq-len', type=int, default=32,
                        help='Sequence length')
    parser.add_argument('--num-objects', type=int, default=10,
                        help='Max objects per scene (padded)')
    parser.add_argument('--state-dim', type=int, default=28,
                        help='State vector dimension')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size for processing')
    args = parser.parse_args()

    clevrer_path = Path(args.clevrer_path)
    if not clevrer_path.exists():
        raise FileNotFoundError(f"CLEVRER path not found: {clevrer_path}")

    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Try Google Drive first (Colab), then local
        gdrive_path = Path("/content/drive/MyDrive/physics_action_predictor/data")
        local_path = Path(__file__).parent.parent / "data"
        output_dir = gdrive_path if gdrive_path.exists() else local_path

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all scene files
    scene_files = sorted(clevrer_path.glob('*.json'))[:args.num_samples]
    print(f"Found {len(scene_files)} scene files")

    # Process scenes in batches to avoid memory issues
    all_sequences = []
    all_schemas = []
    batch_size = args.batch_size

    for batch_start in range(0, len(scene_files), batch_size):
        batch_files = scene_files[batch_start:batch_start + batch_size]
        batch_sequences = []
        batch_schemas = []

        print(f"\nProcessing batch {batch_start // batch_size + 1}/{(len(scene_files) + batch_size - 1) // batch_size}")

        for scene_path in tqdm(batch_files, desc=f"Batch {batch_start // batch_size + 1}"):
            try:
                seq, schema = process_scene(scene_path, args.seq_len, args.num_objects)
                if seq is not None:
                    batch_sequences.append(seq)
                    batch_schemas.append(schema)
            except Exception as e:
                print(f"Warning: Failed to process {scene_path.name}: {e}")
                continue

        if batch_sequences:
            # Stack batch and convert to tensor
            batch_tensor = torch.from_numpy(np.stack(batch_sequences))
            all_sequences.append(batch_tensor)
            all_schemas.extend(batch_schemas)
            print(f"  Processed {len(batch_sequences)} scenes in this batch")

            # Force garbage collection
            del batch_sequences
            import gc
            gc.collect()

    # Concatenate all batches
    print(f"\nConcatenating {len(all_sequences)} batches...")
    sequences_tensor = torch.cat(all_sequences, dim=0)
    schemas_tensor = torch.tensor(all_schemas, dtype=torch.long)

    print(f"Successfully processed {len(all_schemas)} scenes")
    print(f"Sequences shape: {sequences_tensor.shape}")
    print(f"Schemas shape: {schemas_tensor.shape}")

    # Schema distribution
    schema_counts = torch.bincount(schemas_tensor, minlength=37)
    print(f"Schema distribution (non-zero): ", end="")
    for i, c in enumerate(schema_counts):
        if c > 0:
            print(f"{i}:{c.item()} ", end="")
    print()

    output_file = output_dir / "physics_sequences.h5"

    # Save to HDF5 format
    with h5py.File(output_file, 'w') as f:
        # Store main data
        f.create_dataset('sequences', data=sequences_tensor.numpy(), compression='gzip')
        f.create_dataset('schemas', data=schemas_tensor.numpy(), compression='gzip')

        # Store metadata as attributes
        f.attrs['source'] = 'CLEVRER'
        f.attrs['num_samples'] = len(all_schemas)
        f.attrs['seq_len'] = args.seq_len
        f.attrs['num_objects'] = args.num_objects
        f.attrs['state_dim'] = args.state_dim

    print(f"\nSaved to: {output_file}")
    print(f"  Sequences: {sequences_tensor.shape}")
    print(f"  Schemas: {schemas_tensor.shape}")


if __name__ == '__main__':
    main()
