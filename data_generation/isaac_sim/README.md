# Isaac Sim Data Generation

Standalone tools for generating physics simulation training data from Isaac Sim and PyBullet.

## Files

- `generate_isaac_hdf5.py` - Generate HDF5 data from Isaac Sim tasks
- `generate_robotics_hdf5.py` - Generate HDF5 data from PyBullet robotics simulations
- `generate_robotics_schemas.py` - Robotics schema definitions and episode generation
- `helpers/` - Utility modules for state extraction, HDF5 writing, and schema registry
- `SETUP.md` - Isaac Sim environment setup instructions

## Usage

### Generate PyBullet Robotics Data (No Isaac Sim Required)

```bash
python generate_robotics_hdf5.py \
    --episodes 1200 \
    --output-dir D:/physics_hdf5 \
    --batch-size 100
```

### Generate Isaac Sim Data (Requires Isaac Sim)

```bash
# Single task
python generate_isaac_hdf5.py --task FrankaCabinet --episodes 1200

# All tasks
python generate_isaac_hdf5.py --task all --episodes 1200

# Synthetic data (no Isaac Sim required)
python generate_isaac_hdf5.py --task FrankaCabinet --episodes 1200 --synthetic
```

## Robotics Schemas

The PyBullet generator supports these schemas:

| Schema | Description | Timesteps |
|--------|-------------|-----------|
| `object_pickup` | Grasping and lifting objects | 150 |
| `fragile_objects` | Handling glass/ceramic objects | 100 |
| `occlusion_static` | Objects behind occluders | 50 |
| `object_placement` | Placing objects on surfaces | 150 |
| `rotation_manipulation` | Rotating objects | 200 |

## Output Format

HDF5 files contain:
- `states`: `[episodes, seq_len, max_objects, state_dim]` - Physics states
- `masks`: `[episodes, seq_len, max_objects]` - Object masks
- `next_states`: `[episodes, seq_len-1, max_objects, state_dim]` - Next timestep states
- `schemas`: `[episodes]` - Schema name per episode

### State Vector (21 dimensions)

| Index | Field | Description |
|-------|-------|-------------|
| 0-2 | position | x, y, z |
| 3-5 | velocity | vx, vy, vz |
| 6-9 | orientation | quaternion (x, y, z, w) |
| 10-12 | angular_velocity | wx, wy, wz |
| 13 | mass | object mass |
| 14 | radius | bounding radius |
| 15-17 | color | RGB |
| 18 | shape_type | 0=sphere, 1=box, 2=cylinder |
| 19 | is_static | 1 if static, 0 if dynamic |
| 20 | graspable | 1 if graspable, 0 otherwise |

## For Adapter Training

After generating data, copy the HDF5 files to:
```
/content/drive/MyDrive/physics_action_predictor/data/physics_former_adapter/physics_hdf5/
```

Or use the QA data generation tools to create the `adapter_qa_cache.pt` file directly.
