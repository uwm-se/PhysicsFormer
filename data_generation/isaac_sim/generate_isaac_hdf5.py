"""
Copyright (c) 2026 Style Machine LLC. All rights reserved.

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Style Machine LLC.
"""

"""
Generate Physics Training Data from Isaac Sim to HDF5 Format

This script runs Isaac Sim tasks and exports state trajectories to HDF5
format compatible with PhysicsFormer training.

Usage:
    python generate_isaac_hdf5.py --task FrankaCabinet --episodes 1200
    python generate_isaac_hdf5.py --task all --episodes 1200
    python generate_isaac_hdf5.py --list-schemas
"""

import sys
import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import gc

from helpers.state_extraction import (
    MAX_OBJECTS, STATE_DIM, StateExtractor,
    create_state_array, create_mask_array
)
from helpers.hdf5_utils import (
    create_physics_hdf5, write_episode_batch, HDF5Config
)
from helpers.schema_registry import (
    ISAAC_SCHEMAS, get_schema_config, get_task_to_schema_mapping,
    list_available_schemas, list_available_tasks
)

MAX_SEQ_LENGTH = 512


def extract_state_from_task(task, env_idx: int, state_extractor: StateExtractor) -> tuple:
    """Extract state using the configured StateExtractor."""
    return state_extractor.extract_all(task, env_idx)


def generate_task_data(
    task_name: str,
    num_episodes: int,
    output_dir: Path,
    num_envs: int = 4,
    max_steps: int = None,
    headless: bool = True
):
    """Generate HDF5 data for a single Isaac Sim task."""
    
    task_to_schema = get_task_to_schema_mapping()
    schema_name = task_to_schema.get(task_name, f"isaac_{task_name.lower()}")
    schema_config = get_schema_config(schema_name)
    
    if max_steps is None:
        max_steps = schema_config.max_steps if schema_config else 200
    
    state_extractor = schema_config.create_state_extractor() if schema_config else StateExtractor()
    
    output_file = output_dir / f"{schema_name}.h5"
    
    print(f"\n{'='*60}")
    print(f"Generating: {task_name} -> {schema_name}")
    print(f"Episodes: {num_episodes}, Envs: {num_envs}, Steps: {max_steps}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
    try:
        from omni.isaac.gym.vec_env import VecEnvBase
        sys.path.insert(0, str(Path(__file__).parent / "OmniIsaacGymEnvs"))
        from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
        from hydra import compose, initialize_config_dir
        
        config_dir = Path(__file__).parent / "OmniIsaacGymEnvs" / "omniisaacgymenvs" / "cfg"
        
        with initialize_config_dir(config_path=str(config_dir)):
            cfg = compose(config_name="config", overrides=[
                f"task={task_name}",
                f"num_envs={num_envs}",
                f"headless={headless}",
                "test=True"
            ])
        
        env = VecEnvRLGames(
            headless=headless,
            sim_device="cuda:0",
            enable_livestream=False,
            enable_viewport=False
        )
        
        from omniisaacgymenvs.utils.task_util import initialize_task
        task = initialize_task(cfg, env)
        
    except ImportError as e:
        print(f"ERROR: Could not import Isaac Sim modules: {e}")
        print("Make sure you're running from the Isaac Sim Python environment")
        print("Generating synthetic placeholder data instead...")
        generate_synthetic_data(schema_name, num_episodes, output_dir, max_steps)
        return
    
    import torch
    hdf5_config = HDF5Config(max_seq_length=MAX_SEQ_LENGTH)
    
    with create_physics_hdf5(output_file, num_episodes, hdf5_config) as f:
        episodes_per_batch = num_envs
        num_batches = (num_episodes + episodes_per_batch - 1) // episodes_per_batch
        
        episode_idx = 0
        for batch in tqdm(range(num_batches), desc=f"  {task_name}"):
            env.reset()
            
            batch_states = []
            batch_masks = []
            
            for env_idx in range(min(num_envs, num_episodes - episode_idx)):
                episode_states = np.zeros((MAX_SEQ_LENGTH, MAX_OBJECTS, STATE_DIM), dtype=np.float32)
                episode_masks = np.zeros((MAX_SEQ_LENGTH, MAX_OBJECTS), dtype=np.float32)
                
                for step in range(min(max_steps, MAX_SEQ_LENGTH)):
                    states, masks = extract_state_from_task(task, env_idx, state_extractor)
                    episode_states[step] = states
                    episode_masks[step] = masks
                    
                    actions = torch.zeros((num_envs, task.num_actions), device="cuda:0")
                    actions[:, :] = torch.randn_like(actions) * 0.1
                    _, _, dones, _ = env.step(actions)
                    
                    if dones[env_idx]:
                        break
                
                batch_states.append(episode_states)
                batch_masks.append(episode_masks)
            
            write_episode_batch(f, episode_idx, batch_states, batch_masks, schema_name)
            episode_idx += len(batch_states)
            
            if batch % 10 == 0:
                gc.collect()
    
    env.close()
    print(f"  ✓ Saved {num_episodes} episodes to {output_file.name}")


def generate_synthetic_data(
    schema_name: str,
    num_episodes: int,
    output_dir: Path,
    max_steps: int = 200
):
    """Generate synthetic placeholder data when Isaac Sim is not available."""
    
    output_file = output_dir / f"{schema_name}.h5"
    print(f"\n  Generating synthetic data for {schema_name}...")
    
    with h5py.File(output_file, 'w') as f:
        states_ds = f.create_dataset(
            'states',
            shape=(num_episodes, MAX_SEQ_LENGTH, MAX_OBJECTS, STATE_DIM),
            dtype='float32',
            chunks=(min(50, num_episodes), MAX_SEQ_LENGTH, MAX_OBJECTS, STATE_DIM),
            compression='gzip',
            compression_opts=4
        )
        
        masks_ds = f.create_dataset(
            'masks',
            shape=(num_episodes, MAX_SEQ_LENGTH, MAX_OBJECTS),
            dtype='float32',
            chunks=(min(50, num_episodes), MAX_SEQ_LENGTH, MAX_OBJECTS),
            compression='gzip',
            compression_opts=4
        )
        
        next_states_ds = f.create_dataset(
            'next_states',
            shape=(num_episodes, MAX_SEQ_LENGTH - 1, MAX_OBJECTS, STATE_DIM),
            dtype='float32',
            chunks=(min(50, num_episodes), MAX_SEQ_LENGTH - 1, MAX_OBJECTS, STATE_DIM),
            compression='gzip',
            compression_opts=4
        )
        
        dt = h5py.special_dtype(vlen=str)
        schemas_ds = f.create_dataset('schemas', shape=(num_episodes,), dtype=dt)
        
        for ep in tqdm(range(num_episodes), desc=f"  {schema_name}"):
            num_objects = np.random.randint(3, 8)
            
            states = np.zeros((MAX_SEQ_LENGTH, MAX_OBJECTS, STATE_DIM), dtype=np.float32)
            masks = np.zeros((MAX_SEQ_LENGTH, MAX_OBJECTS), dtype=np.float32)
            
            initial_pos = np.random.randn(num_objects, 3) * 0.3
            initial_vel = np.random.randn(num_objects, 3) * 0.1
            
            for t in range(min(max_steps, MAX_SEQ_LENGTH)):
                dt = 1.0 / 60.0
                
                for obj in range(num_objects):
                    pos = initial_pos[obj] + initial_vel[obj] * t * dt
                    pos[2] = max(0, pos[2] - 0.5 * 9.81 * (t * dt) ** 2)
                    
                    states[t, obj, 0:3] = pos
                    states[t, obj, 3:6] = initial_vel[obj]
                    states[t, obj, 6:10] = [0, 0, 0, 1]  # Identity quaternion
                    states[t, obj, 13] = 1.0  # Mass
                    states[t, obj, 14] = 0.05  # Radius
                    states[t, obj, 15:18] = np.random.rand(3)  # Color
                    states[t, obj, 18] = np.random.randint(0, 3)  # Shape
                    states[t, obj, 19] = 0  # Not static
                    states[t, obj, 20] = 0.5  # Friction
                    masks[t, obj] = 1.0
            
            states_ds[ep] = states
            masks_ds[ep] = masks
            next_states_ds[ep] = states[1:, :, :]
            schemas_ds[ep] = schema_name
    
    print(f"  ✓ Saved {num_episodes} synthetic episodes to {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate Isaac Sim data to HDF5")
    parser.add_argument("--task", type=str, default="FrankaCabinet",
                        help=f"Task name or 'all'. Options: {list_available_tasks()}")
    parser.add_argument("--episodes", type=int, default=1200,
                        help="Episodes per task")
    parser.add_argument("--output-dir", type=str, default="D:/physics_hdf5",
                        help="Output directory for HDF5 files")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run without GUI")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data (no Isaac Sim required)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.task.lower() == "all":
        tasks = list_available_tasks()
    else:
        tasks = [args.task]
    
    print("=" * 60)
    print("ISAAC SIM HDF5 DATA GENERATION")
    print("=" * 60)
    print(f"\nTasks: {tasks}")
    print(f"Episodes per task: {args.episodes}")
    print(f"Output: {output_dir}")
    
    for task in tasks:
        if args.synthetic:
            task_to_schema = get_task_to_schema_mapping()
            schema_name = task_to_schema.get(task, f"isaac_{task.lower()}")
            generate_synthetic_data(schema_name, args.episodes, output_dir, args.max_steps)
        else:
            generate_task_data(
                task,
                args.episodes,
                output_dir,
                num_envs=args.num_envs,
                max_steps=args.max_steps,
                headless=args.headless
            )
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    import torch
    main()
