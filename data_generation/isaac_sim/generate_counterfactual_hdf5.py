"""
Generate Counterfactual Physics Data from Isaac Sim to HDF5 Format.

Uses existing Isaac Sim HDF5 episodes as templates: reads the initial frame,
injects it into the Isaac Sim environment, applies a perturbation (object
removal, mass change, velocity change, position shift), and re-simulates
with PhysX to produce a counterfactual trajectory.

Only processes collision-heavy schemas used in training (skips locomotion,
manipulation, and control tasks).

Usage:
    python generate_counterfactual_hdf5.py --source-dir D:/physics_hdf5 --episodes-per-schema 200
    python generate_counterfactual_hdf5.py --source-dir D:/physics_hdf5 --perturbations 3 --list-schemas
"""

import sys
import argparse
import json
import numpy as np
import h5py
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from helpers.state_extraction import (
    MAX_OBJECTS, STATE_DIM, STATE_INDICES,
)
from helpers.hdf5_utils import create_physics_hdf5, HDF5Config

MAX_SEQ_LENGTH = 512

# ═══════════════════════════════════════════════════════════════
# Schemas used in training (collision-heavy, multi-object physics)
# Skip: locomotion (ant, humanoid, anymal), manipulation (franka,
# hand), control (cartpole, quadcopter, ball_balance)
# ═══════════════════════════════════════════════════════════════

TRAINING_SCHEMAS = {
    # Direct matches: disk file + registry + custom task
    'isaac_block_stacking',
    'isaac_domino_line',
    'isaac_multi_body_collision',      # alias for isaac_multi_ball_collision
    'isaac_ramp_roll',                  # alias for isaac_ramp_rolling
    # In registry, will attempt Isaac Sim task load
    'isaac_friction_variety',
    'isaac_projectile_trajectory',
    # Collision/scatter schemas on disk (no custom tasks yet — will fail and report)
    'isaac_angled_collision',
    'isaac_angled_throw',
    'isaac_angular_momentum',
    'isaac_asymmetric_collision',
    'isaac_billiard_break',
    'isaac_bowling_strike',
    'isaac_chain_collision',
    'isaac_cluster_collision',
    'isaac_cube_slide',
    'isaac_directed_scatter',
    'isaac_explosion_scatter',
    'isaac_friction_compare',
    'isaac_funnel_scatter',
    'isaac_head_on_collision',
    'isaac_horizontal_throw',
    'isaac_impact_scatter',
    'isaac_lob_throw',
    'isaac_multi_drop',
    'isaac_multi_projectile',
    'isaac_obstacle_drop',
    'isaac_offset_stack',
    'isaac_pyramid_stack',
    'isaac_ramp_slide',
    'isaac_simple_stack',
    'isaac_simultaneous_drop',
    'isaac_tall_stack',
    'isaac_unstable_stack',
    'isaac_varied_height_drop',
    'isaac_varied_mass_drop',
    'isaac_wedge_deflect',
}


# ═══════════════════════════════════════════════════════════════
# Perturbations
# ═══════════════════════════════════════════════════════════════

@dataclass
class Perturbation:
    kind: str           # 'remove_object', 'change_mass', 'change_velocity', 'shift_position'
    object_idx: int     # Index in the state array
    params: Dict

    def to_json(self) -> str:
        return json.dumps({
            'kind': self.kind,
            'object_idx': self.object_idx,
            'params': self.params,
        })


def random_perturbation(masks: np.ndarray) -> Optional[Perturbation]:
    """Generate a random perturbation targeting an active, non-static object."""
    active = np.where(masks > 0.5)[0]
    if len(active) == 0:
        return None

    obj_idx = int(np.random.choice(active))
    kind = np.random.choice(
        ['remove_object', 'change_mass', 'change_velocity', 'shift_position'],
        p=[0.30, 0.25, 0.25, 0.20],
    )

    if kind == 'remove_object':
        params = {}
    elif kind == 'change_mass':
        params = {'factor': float(np.random.uniform(0.5, 3.0))}
    elif kind == 'change_velocity':
        params = {'delta': np.random.randn(3).tolist()}
    elif kind == 'shift_position':
        params = {'delta': (np.random.randn(3) * 0.2).tolist()}
    else:
        params = {}

    return Perturbation(kind=kind, object_idx=obj_idx, params=params)


# ═══════════════════════════════════════════════════════════════
# Scene creation and state extraction (Isaac Sim core API)
# ═══════════════════════════════════════════════════════════════

def _apply_perturbation_to_state(state_vec, perturbation, obj_idx):
    """Apply perturbation to a state vector copy. Returns (modified_vec, should_skip)."""
    if perturbation is None or perturbation.object_idx != obj_idx:
        return state_vec, False

    vec = state_vec.copy()
    if perturbation.kind == 'remove_object':
        return vec, True
    elif perturbation.kind == 'change_mass':
        vec[STATE_INDICES['mass']] *= perturbation.params.get('factor', 2.0)
    elif perturbation.kind == 'change_velocity':
        if 'delta' in perturbation.params:
            delta = np.array(perturbation.params['delta'], dtype=np.float32)
            vec[STATE_INDICES['linear_velocity']] += delta
    elif perturbation.kind == 'shift_position':
        delta = np.array(perturbation.params.get('delta', [0.2, 0, 0]), dtype=np.float32)
        vec[STATE_INDICES['position']] += delta
    return vec, False


# Y-offset for "disabled" objects: place them far below ground
_DISABLED_POS = [0.0, 0.0, -100.0]
_IDENTITY_ORIENT = [0.0, 0.0, 0.0, 1.0]

# Module-level cache for World singleton and object pool
_world = None
_pool_view = None


def _to_numpy(x):
    """Convert torch tensor or numpy array to numpy."""
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    return np.asarray(x)


def _ensure_pool():
    """Create the World and object pool once (singleton). Returns (world, view)."""
    global _world, _pool_view
    if _world is not None and _pool_view is not None:
        return _world, _pool_view

    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicSphere
    from omni.isaac.core.prims import RigidPrimView

    _world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0/60.0,
        rendering_dt=1.0/60.0,
    )
    _world.scene.add_default_ground_plane()

    # Create MAX_OBJECTS spheres, all initially disabled (below ground)
    for i in range(MAX_OBJECTS):
        DynamicSphere(
            prim_path=f"/World/pool/obj_{i}",
            name=f"pool_obj_{i}",
            position=_DISABLED_POS,
            radius=0.05,
            mass=1.0,
        )

    _pool_view = RigidPrimView(
        prim_paths_expr="/World/pool/obj_*",
        name="object_pool",
    )
    _world.scene.add(_pool_view)
    _world.reset()
    print(f"  Object pool created: {MAX_OBJECTS} spheres")

    return _world, _pool_view


def generate_counterfactual_data(
    schema_name: str,
    source_h5_path: str,
    output_dir: Path,
    num_episodes: int = 200,
    perturbations_per_episode: int = 3,
    headless: bool = True,
    branch_timestep: int = 0,
):
    """Generate counterfactual HDF5 data for a single schema using Isaac Sim core API.

    Object pool approach: creates MAX_OBJECTS spheres once (singleton), then
    repositions them for each episode. Unused objects are placed far below ground.
    This avoids creating/destroying prims and invalidating simulation views.
    """
    output_file = output_dir / f"cf_{schema_name}.h5"
    total_episodes = num_episodes * perturbations_per_episode

    print(f"\n{'='*60}")
    print(f"Counterfactual: {schema_name}")
    print(f"Source episodes: {num_episodes}, perturbations each: {perturbations_per_episode}")
    print(f"Total counterfactual episodes: {total_episodes}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")

    # Load source data
    with h5py.File(source_h5_path, 'r') as src:
        n_source = src['states'].shape[0]
        source_seq_len = src['states'].shape[1]
        source_episodes = min(num_episodes, n_source)
        print(f"  Source has {n_source} episodes ({source_seq_len} steps each), using {source_episodes}")

        source_initials = []
        source_masks = []
        for i in range(source_episodes):
            t0 = branch_timestep
            source_initials.append(src['states'][i, t0])
            source_masks.append(src['masks'][i, t0])

    max_steps = min(source_seq_len, MAX_SEQ_LENGTH)

    # Get or create the singleton World + pool
    world, view = _ensure_pool()

    # Delete existing output file to avoid Windows file lock issues
    if output_file.exists():
        output_file.unlink()

    # --- Generate counterfactuals ---
    # Use chunk_size=1 since we write one episode at a time (avoids gzip read errors)
    hdf5_config = HDF5Config(max_seq_length=MAX_SEQ_LENGTH, chunk_size=1)

    with create_physics_hdf5(output_file, total_episodes, hdf5_config) as f:
        vlen_str = h5py.special_dtype(vlen=str)
        f.create_dataset('perturbations', shape=(total_episodes,), dtype=vlen_str)
        f.create_dataset('source_episodes', shape=(total_episodes,), dtype='int32')

        cf_idx = 0

        for ep in tqdm(range(source_episodes), desc=f"  {schema_name}"):
            initial_state = source_initials[ep]
            initial_mask = source_masks[ep]

            for p_idx in range(perturbations_per_episode):
                pert = random_perturbation(initial_mask)
                if pert is None:
                    continue

                # --- Set up episode: position all pool objects ---
                positions = np.zeros((MAX_OBJECTS, 3), dtype=np.float32)
                orientations = np.zeros((MAX_OBJECTS, 4), dtype=np.float32)
                velocities = np.zeros((MAX_OBJECTS, 6), dtype=np.float32)
                active_mask = np.zeros(MAX_OBJECTS, dtype=np.float32)

                for obj_idx in range(MAX_OBJECTS):
                    if initial_mask[obj_idx] < 0.5:
                        positions[obj_idx] = _DISABLED_POS
                        orientations[obj_idx] = _IDENTITY_ORIENT
                        continue

                    state_vec, should_skip = _apply_perturbation_to_state(
                        initial_state[obj_idx], pert, obj_idx,
                    )
                    if should_skip:
                        positions[obj_idx] = _DISABLED_POS
                        orientations[obj_idx] = _IDENTITY_ORIENT
                        continue

                    positions[obj_idx] = state_vec[STATE_INDICES['position']]
                    orientations[obj_idx] = state_vec[STATE_INDICES['orientation']]
                    velocities[obj_idx, :3] = state_vec[STATE_INDICES['linear_velocity']]
                    velocities[obj_idx, 3:6] = state_vec[STATE_INDICES['angular_velocity']]
                    active_mask[obj_idx] = 1.0

                # Apply to simulation (view accepts numpy or torch)
                view.set_world_poses(positions=positions, orientations=orientations)
                view.set_velocities(velocities=velocities)

                # Step simulation and record trajectory
                ep_states = np.zeros(
                    (MAX_SEQ_LENGTH, MAX_OBJECTS, STATE_DIM), dtype=np.float32
                )
                ep_masks = np.zeros(
                    (MAX_SEQ_LENGTH, MAX_OBJECTS), dtype=np.float32
                )

                for step in range(max_steps):
                    world.step(render=False)

                    poses = view.get_world_poses()
                    pos_np = _to_numpy(poses[0])
                    orient_np = _to_numpy(poses[1])
                    vel_np = _to_numpy(view.get_velocities())

                    for obj_idx in range(MAX_OBJECTS):
                        if active_mask[obj_idx] < 0.5:
                            continue
                        ep_states[step, obj_idx, STATE_INDICES['position']] = pos_np[obj_idx, :3]
                        ep_states[step, obj_idx, STATE_INDICES['orientation']] = orient_np[obj_idx]
                        ep_states[step, obj_idx, STATE_INDICES['linear_velocity']] = vel_np[obj_idx, :3]
                        ep_states[step, obj_idx, STATE_INDICES['angular_velocity']] = vel_np[obj_idx, 3:6]
                        ep_masks[step, obj_idx] = 1.0

                # Write to HDF5 (next_states has shape [max_seq_length-1, ...])
                f['states'][cf_idx] = ep_states
                f['masks'][cf_idx] = ep_masks
                f['next_states'][cf_idx] = ep_states[1:MAX_SEQ_LENGTH]
                f['schemas'][cf_idx] = f"cf_{schema_name}"
                f['sequence_lengths'][cf_idx] = max_steps
                f['perturbations'][cf_idx] = pert.to_json()
                f['source_episodes'][cf_idx] = ep

                cf_idx += 1

            if ep % 10 == 0:
                gc.collect()

        # Trim if needed
        if cf_idx < total_episodes:
            for key in f.keys():
                f[key].resize(cf_idx, axis=0)

    print(f"  Saved {cf_idx} counterfactual episodes to {output_file.name}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate counterfactual physics data using Isaac Sim"
    )
    parser.add_argument(
        "--source-dir", type=str, default="D:/physics_hdf5",
        help="Directory containing source Isaac Sim .h5 files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="D:/physics_counterfactual_hdf5",
        help="Output directory for counterfactual HDF5 files",
    )
    parser.add_argument(
        "--episodes-per-schema", type=int, default=200,
        help="Max source episodes to use per schema",
    )
    parser.add_argument(
        "--perturbations", type=int, default=3,
        help="Counterfactual perturbations per source episode",
    )
    parser.add_argument(
        "--branch-timestep", type=int, default=0,
        help="Frame to branch from (0 = initial state)",
    )
    parser.add_argument(
        "--headless", action="store_true", default=True,
        help="Run Isaac Sim without GUI",
    )
    parser.add_argument(
        "--list-schemas", action="store_true",
        help="List training schemas and exit",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=False,
        help="Skip schemas that already have output files (for resuming)",
    )

    args = parser.parse_args()

    if args.list_schemas:
        print("Training schemas (counterfactual generation targets):")
        for name in sorted(TRAINING_SCHEMAS):
            print(f"  {name}")
        print(f"\nTotal: {len(TRAINING_SCHEMAS)} schemas")
        return

    # Initialize Isaac Sim before any omni.* imports
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})
    print("Isaac Sim initialized (headless={})".format(args.headless))

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find source HDF5 files that match training schemas
    h5_files = sorted(source_dir.glob('*.h5'))
    schema_files = {}
    for h5_path in h5_files:
        # Match filename to schema (e.g., isaac_multi_ball_collision.h5)
        name = h5_path.stem
        if name in TRAINING_SCHEMAS:
            schema_files[name] = h5_path

    if not schema_files:
        print(f"No training schema HDF5 files found in {source_dir}")
        print(f"Expected files like: isaac_block_stacking.h5, isaac_domino_line.h5, ...")
        print(f"Available files: {[f.name for f in h5_files]}")
        return

    print("=" * 60)
    print("COUNTERFACTUAL DATA GENERATION (Isaac Sim)")
    print("=" * 60)
    print(f"\nSource: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Schemas found: {len(schema_files)}")
    for name, path in schema_files.items():
        with h5py.File(path, 'r') as f:
            n = f['states'].shape[0]
        print(f"  {name}: {n} source episodes -> {path.name}")
    print(f"Perturbations per episode: {args.perturbations}")
    print(f"Branch timestep: {args.branch_timestep}")

    succeeded = []
    failed = []

    for schema_name, source_path in schema_files.items():
        if args.skip_existing:
            cf_path = output_dir / f"cf_{schema_name}.h5"
            if cf_path.exists() and cf_path.stat().st_size > 5_000_000:
                print(f"\n  Skipping {schema_name} (output exists: {cf_path.stat().st_size / 1e6:.1f} MB)")
                succeeded.append(schema_name)
                continue
        try:
            generate_counterfactual_data(
                schema_name=schema_name,
                source_h5_path=str(source_path),
                output_dir=output_dir,
                num_episodes=args.episodes_per_schema,
                perturbations_per_episode=args.perturbations,
                headless=args.headless,
                branch_timestep=args.branch_timestep,
            )
            succeeded.append(schema_name)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            failed.append((schema_name, error_msg))
            print(f"\nERROR generating {schema_name}: {error_msg}")
            print(f"  Skipping to next schema...\n")

    # ── Final Report ──
    print(f"\n{'='*60}")
    print("GENERATION REPORT")
    print(f"{'='*60}")

    if succeeded:
        cf_files = sorted(output_dir.glob('cf_*.h5'))
        total_eps = 0
        total_mb = 0
        print(f"\nSUCCEEDED ({len(succeeded)}/{len(schema_files)}):")
        for cf_path in cf_files:
            with h5py.File(cf_path, 'r') as f:
                n = f['states'].shape[0]
            mb = cf_path.stat().st_size / 1e6
            total_eps += n
            total_mb += mb
            print(f"  {cf_path.name}: {n} episodes ({mb:.1f} MB)")
        print(f"  Total: {total_eps} counterfactual episodes, {total_mb:.1f} MB")

    if failed:
        print(f"\nFAILED ({len(failed)}/{len(schema_files)}):")
        for schema_name, error_msg in failed:
            print(f"  {schema_name}: {error_msg}")

    if not failed:
        print("\nAll schemas completed successfully.")

    # Shut down Isaac Sim
    simulation_app.close()
    print("\nIsaac Sim shut down.")


if __name__ == "__main__":
    main()
