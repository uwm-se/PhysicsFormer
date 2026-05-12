"""
Render Counterfactual HDF5 Trajectories as Videos via Isaac Sim.

Replays counterfactual physics trajectories in Isaac Sim's renderer,
captures RGB frames, and assembles them into MP4 videos with ffmpeg.

Object properties (radius, color, shape, dimensions) are read from the
original Isaac Sim HDF5 files since counterfactual files only store
position/velocity/orientation.

Usage:
    C:\\isaac\\Scripts\\python.exe render_counterfactual_videos.py
    C:\\isaac\\Scripts\\python.exe render_counterfactual_videos.py --schema cf_isaac_billiard_break
    C:\\isaac\\Scripts\\python.exe render_counterfactual_videos.py --max-frames 100 --resolution 1920 1080
"""

import sys
import argparse
import json
import subprocess
import shutil
import tempfile
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple

from helpers.state_extraction import (
    MAX_OBJECTS, STATE_DIM, STATE_INDICES, SHAPE_TYPES,
)

# Defaults
DEFAULT_CF_DIR = "D:/physics_counterfactual_hdf5"
DEFAULT_ORIG_DIR = "D:/physics_hdf5"
DEFAULT_OUTPUT_DIR = "D:/physics_counterfactual_videos"
DEFAULT_MAX_FRAMES = 150
DEFAULT_RESOLUTION = (1280, 720)


def load_object_properties(orig_h5_path: str, num_active: int):
    """Load object properties (radius, color, shape, dims) from original HDF5."""
    props = []
    with h5py.File(orig_h5_path, 'r') as f:
        state_vec = f['states'][0, 0]  # First episode, first frame: [MAX_OBJECTS, 35]
        for i in range(num_active):
            sv = state_vec[i]
            props.append({
                'radius': float(sv[STATE_INDICES['radius']]),
                'color': sv[STATE_INDICES['color']].tolist(),
                'shape_type': int(sv[STATE_INDICES['shape_type']]),
                'mass': float(sv[STATE_INDICES['mass']]),
                'dimensions': sv[STATE_INDICES['dimensions']].tolist(),
                'is_static': bool(sv[STATE_INDICES['is_static']] > 0.5),
                'friction': float(sv[STATE_INDICES['friction']]),
            })
    return props


def load_cf_trajectory(cf_h5_path: str, max_frames: int, episode: int = 0):
    """Load counterfactual trajectory and perturbation info from HDF5."""
    with h5py.File(cf_h5_path, 'r') as f:
        states = f['states'][episode]  # [T, MAX_OBJECTS, 35]
        masks = f['masks'][episode]    # [T, MAX_OBJECTS]

        # Number of active objects from first frame mask
        num_active = int(masks[0].sum())
        num_frames = min(states.shape[0], max_frames)

        # Extract positions and orientations
        positions = states[:num_frames, :num_active, STATE_INDICES['position']].copy()
        orientations = states[:num_frames, :num_active, STATE_INDICES['orientation']].copy()

        # Read perturbation metadata
        perturbation_info = None
        if 'perturbations' in f:
            try:
                raw = f['perturbations'][episode]
                pert_str = raw.decode() if isinstance(raw, bytes) else str(raw)
                perturbation_info = json.loads(pert_str)
            except Exception:
                pass

    return positions, orientations, num_active, num_frames, perturbation_info


def compute_camera_pose(positions: np.ndarray):
    """Compute camera position and look-at target from trajectory positions.

    Args:
        positions: [num_frames, num_objects, 3]

    Returns:
        cam_pos: [3] camera position
        look_at: [3] look-at target (scene center)
    """
    # Flatten to [N, 3] ignoring disabled objects at origin
    all_pos = positions.reshape(-1, 3)
    # Filter out disabled positions (at z=-100 or origin with zero movement)
    valid = np.abs(all_pos[:, 2] + 100) > 1.0  # not at -100
    if valid.sum() < 3:
        valid = np.ones(len(all_pos), dtype=bool)
    all_pos = all_pos[valid]

    center = all_pos.mean(axis=0)
    extent = all_pos.max(axis=0) - all_pos.min(axis=0)
    scene_size = max(extent.max(), 0.5)

    # Isometric-ish view: above and behind
    cam_distance = scene_size * 3.0
    cam_pos = center + np.array([0.0, -cam_distance * 0.7, cam_distance * 0.5])

    return cam_pos, center


def create_scene(world, object_props, num_active, positions_frame0, orientations_frame0):
    """Create Isaac Sim scene objects from HDF5 object properties.

    Returns a RigidPrimView covering all created objects.
    """
    from omni.isaac.core.objects import DynamicSphere, DynamicCuboid, VisualSphere, VisualCuboid
    from omni.isaac.core.prims import RigidPrimView

    prim_paths = []
    for i in range(num_active):
        props = object_props[i] if i < len(object_props) else {}
        radius = props.get('radius', 0.05)
        color_rgb = props.get('color', [0.5, 0.5, 0.5])
        shape_type = props.get('shape_type', 0)
        mass = props.get('mass', 1.0)
        dims = props.get('dimensions', [0.1, 0.1, 0.1])
        is_static = props.get('is_static', False)

        # Ensure minimum radius for visibility
        if radius < 0.005:
            radius = 0.05

        prim_path = f"/World/objects/obj_{i}"
        prim_paths.append(prim_path)

        pos = positions_frame0[i].tolist()
        # Convert orientation from [x,y,z,w] HDF5 format to [w,x,y,z] Isaac Sim format
        ori = orientations_frame0[i]
        orient = [float(ori[3]), float(ori[0]), float(ori[1]), float(ori[2])]

        color = np.array(color_rgb, dtype=np.float32)

        if shape_type == 1:  # Box
            dim_x = max(dims[0], 0.01)
            dim_y = max(dims[1], 0.01)
            dim_z = max(dims[2], 0.01)
            DynamicCuboid(
                prim_path=prim_path,
                name=f"obj_{i}",
                position=pos,
                orientation=orient,
                size=1.0,
                scale=[dim_x, dim_y, dim_z],
                color=color,
                mass=mass,
            )
        else:  # Sphere (default)
            DynamicSphere(
                prim_path=prim_path,
                name=f"obj_{i}",
                position=pos,
                orientation=orient,
                radius=radius,
                color=color,
                mass=mass,
            )

    # Create a view for batch pose setting
    view = RigidPrimView(
        prim_paths_expr="/World/objects/obj_*",
        name="replay_objects",
    )
    world.scene.add(view)

    return view


def setup_camera(cam_pos, look_at, resolution):
    """Create a camera prim and render product for RGB capture."""
    import omni.kit.commands
    from pxr import Gf, UsdGeom, Sdf

    stage = omni.usd.get_context().get_stage()

    # Create camera prim
    cam_prim_path = "/World/render_camera"
    camera = UsdGeom.Camera.Define(stage, cam_prim_path)

    # Set camera transform
    direction = np.array(look_at) - np.array(cam_pos)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Compute camera rotation matrix (look-at)
    forward = direction
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build 4x4 transform matrix (USD uses row-major)
    xform = Gf.Matrix4d()
    xform.SetRow(0, Gf.Vec4d(float(right[0]), float(right[1]), float(right[2]), 0))
    xform.SetRow(1, Gf.Vec4d(float(up[0]), float(up[1]), float(up[2]), 0))
    xform.SetRow(2, Gf.Vec4d(float(-forward[0]), float(-forward[1]), float(-forward[2]), 0))
    xform.SetRow(3, Gf.Vec4d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]), 1))

    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xform_op = xformable.AddTransformOp()
    xform_op.Set(xform)

    # Set focal length for reasonable FOV
    camera.GetFocalLengthAttr().Set(18.0)

    # Create render product
    import omni.replicator.core as rep
    rp = rep.create.render_product(cam_prim_path, resolution)

    return cam_prim_path, rp


def create_rgb_annotator(rp):
    """Create and attach an RGB annotator to a render product (once per session)."""
    import omni.replicator.core as rep
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([rp])
    return rgb_annot


def capture_frame(rgb_annot):
    """Capture an RGB frame from an attached annotator."""
    data = rgb_annot.get_data()
    if data is not None and len(data) > 0:
        # data is [H, W, 4] (RGBA uint8)
        if data.ndim == 3 and data.shape[2] >= 3:
            return data[:, :, :3]  # RGB only
    return None


def render_schema(
    schema_name: str,
    cf_dir: Path,
    orig_dir: Path,
    output_dir: Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
    resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
    episode: int = 0,
):
    """Render a single counterfactual schema to MP4."""
    from omni.isaac.core import World

    cf_path = cf_dir / f"{schema_name}.h5"
    if not cf_path.exists():
        print(f"  Counterfactual HDF5 not found: {cf_path}")
        return False

    # Derive original schema name (strip 'cf_' prefix)
    orig_name = schema_name[3:] if schema_name.startswith('cf_') else schema_name
    orig_path = orig_dir / f"{orig_name}.h5"

    # Load trajectory
    print(f"  Loading trajectory from {cf_path.name}...")
    positions, orientations, num_active, num_frames, pert_info = load_cf_trajectory(
        str(cf_path), max_frames, episode
    )
    print(f"  {num_active} objects, {num_frames} frames")

    if pert_info:
        kind = pert_info.get('kind', '?')
        obj_idx = pert_info.get('object_idx', '?')
        print(f"  Perturbation: {kind} on object #{obj_idx}")

    # Load object properties from original file
    if orig_path.exists():
        print(f"  Loading object properties from {orig_path.name}...")
        object_props = load_object_properties(str(orig_path), num_active)
    else:
        print(f"  Original HDF5 not found ({orig_path.name}), using defaults")
        object_props = [{'radius': 0.05, 'color': [0.5, 0.5, 0.5], 'shape_type': 0, 'mass': 1.0}] * num_active

    # Compute camera placement
    cam_pos, look_at = compute_camera_pose(positions)
    print(f"  Camera: pos={cam_pos.round(2)}, look_at={look_at.round(2)}")

    # Create World
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    world.scene.add_default_ground_plane()

    # Create objects
    view = create_scene(world, object_props, num_active, positions[0], orientations[0])

    # Setup camera and render product
    cam_path, rp = setup_camera(cam_pos.tolist(), look_at.tolist(), resolution)
    print(f"  Camera created at {cam_path}")

    # Initialize
    world.reset()

    # Warm up renderer (let it settle for a few frames)
    for _ in range(5):
        world.step(render=True)

    # Create RGB annotator (once, reuse for all frames)
    rgb_annot = create_rgb_annotator(rp)

    # Create temp directory for frames
    frame_dir = Path(tempfile.mkdtemp(prefix=f"{schema_name}_frames_"))
    print(f"  Rendering {num_frames} frames to {frame_dir}...")

    # Replay trajectory and capture frames
    from PIL import Image
    import omni.replicator.core as rep

    for frame_idx in range(num_frames):
        # Build pose arrays for all objects (pad to view size)
        n_prims = view.count
        frame_positions = np.zeros((n_prims, 3), dtype=np.float32)
        frame_orientations = np.zeros((n_prims, 4), dtype=np.float32)
        frame_orientations[:, 0] = 1.0  # w=1 default (identity)

        for obj_idx in range(min(num_active, n_prims)):
            frame_positions[obj_idx] = positions[frame_idx, obj_idx]
            # Convert [x,y,z,w] → [w,x,y,z]
            ori = orientations[frame_idx, obj_idx]
            frame_orientations[obj_idx] = [ori[3], ori[0], ori[1], ori[2]]

        # Set poses (no physics, pure replay)
        view.set_world_poses(
            positions=frame_positions,
            orientations=frame_orientations,
        )

        # Render and capture
        world.step(render=True)
        rep.orchestrator.step()

        rgb = capture_frame(rgb_annot)
        if rgb is not None:
            img = Image.fromarray(rgb.astype(np.uint8))
            img.save(str(frame_dir / f"frame_{frame_idx:04d}.png"))

        if frame_idx % 30 == 0:
            print(f"    Frame {frame_idx}/{num_frames}", flush=True)

    # No world.clear() — in subprocess mode the process exits cleanly;
    # clearing invalidates the simulation view and breaks any subsequent schema.

    # Assemble video with ffmpeg
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{schema_name}.mp4"

    print(f"  Assembling video: {output_path}")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", str(frame_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_path),
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr[:200]}")
        return False

    # Cleanup temp frames
    shutil.rmtree(frame_dir, ignore_errors=True)

    file_size = output_path.stat().st_size / 1e6
    print(f"  Saved: {output_path} ({file_size:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Render counterfactual HDF5 trajectories as videos via Isaac Sim"
    )
    parser.add_argument(
        "--cf-dir", type=str, default=DEFAULT_CF_DIR,
        help="Directory containing counterfactual HDF5 files",
    )
    parser.add_argument(
        "--orig-dir", type=str, default=DEFAULT_ORIG_DIR,
        help="Directory containing original Isaac Sim HDF5 files (for object properties)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for MP4 videos",
    )
    parser.add_argument(
        "--schema", type=str, default=None,
        help="Render a single schema (e.g. cf_isaac_billiard_break)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help="Maximum frames per video",
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=list(DEFAULT_RESOLUTION),
        help="Output resolution (width height)",
    )
    parser.add_argument(
        "--episode", type=int, default=0,
        help="Episode index to render from each HDF5",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=False,
        help="Skip schemas that already have output MP4 files (for resuming)",
    )
    args = parser.parse_args()

    cf_dir = Path(args.cf_dir)
    orig_dir = Path(args.orig_dir)
    output_dir = Path(args.output_dir)

    if not cf_dir.exists():
        print(f"Counterfactual directory not found: {cf_dir}")
        return

    # ── Single-schema mode: initialize Isaac Sim and render directly ──
    if args.schema:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({
            "headless": True,
            "width": args.resolution[0],
            "height": args.resolution[1],
        })
        print(f"Isaac Sim initialized (resolution={args.resolution[0]}x{args.resolution[1]})")

        try:
            render_schema(
                args.schema, cf_dir, orig_dir, output_dir,
                max_frames=args.max_frames,
                resolution=tuple(args.resolution),
                episode=args.episode,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            simulation_app.close()
        return

    # ── Batch mode: spawn a subprocess per schema ──
    # Each subprocess gets its own Isaac Sim instance, avoiding the
    # world.clear() simulation-view-invalidation bug.
    schemas = sorted([f.stem for f in cf_dir.glob('cf_isaac_*.h5')])

    print("=" * 70)
    print(f"RENDERING {len(schemas)} COUNTERFACTUAL SCHEMAS (subprocess-per-schema)")
    print("=" * 70)
    print(f"\nCounterfactual dir: {cf_dir}")
    print(f"Original dir:       {orig_dir}")
    print(f"Output dir:         {output_dir}")
    print(f"Resolution:         {args.resolution[0]}x{args.resolution[1]}")
    print(f"Max frames:         {args.max_frames}")

    success_count = 0
    skipped_count = 0
    failed_schemas = []

    for i, schema in enumerate(schemas):
        print(f"\n[{i + 1}/{len(schemas)}] {schema}")

        # Skip existing videos if requested
        if args.skip_existing:
            mp4_path = output_dir / f"{schema}.mp4"
            if mp4_path.exists() and mp4_path.stat().st_size > 10_000:
                size_kb = mp4_path.stat().st_size / 1e3
                print(f"  Skipping (exists: {size_kb:.0f} KB)")
                skipped_count += 1
                success_count += 1
                continue

        # Build subprocess command
        script_path = str(Path(__file__).resolve())
        cmd = [
            sys.executable, script_path,
            "--cf-dir", str(cf_dir),
            "--orig-dir", str(orig_dir),
            "--output-dir", str(output_dir),
            "--schema", schema,
            "--max-frames", str(args.max_frames),
            "--resolution", str(args.resolution[0]), str(args.resolution[1]),
            "--episode", str(args.episode),
        ]

        try:
            result = subprocess.run(
                cmd, timeout=600, capture_output=False,
            )
            if result.returncode == 0:
                success_count += 1
            else:
                print(f"  Subprocess exited with code {result.returncode}")
                failed_schemas.append(schema)
        except subprocess.TimeoutExpired:
            print(f"  Subprocess timed out (600s)")
            failed_schemas.append(schema)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break

    print("\n" + "=" * 70)
    print(f"RESULTS: {success_count}/{len(schemas)} schemas rendered")
    if skipped_count:
        print(f"  ({skipped_count} skipped — already existed)")
    print("=" * 70)

    if failed_schemas:
        print(f"\nFailed ({len(failed_schemas)}):")
        for s in failed_schemas:
            print(f"  {s}")


if __name__ == "__main__":
    main()
