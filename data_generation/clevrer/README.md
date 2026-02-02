# CLEVRER Data Conversion

Scripts for converting CLEVRER benchmark data into physics state tensors for training.

## Prerequisites

### Download CLEVRER Dataset

1. Download from: http://clevrer.csail.mit.edu/
2. Required files:
   - `train/` - Training videos and annotations
   - `val/` - Validation videos and annotations
   - `test/` - Test videos (optional)

### Directory Structure

```
CLEVRER/
├── train/
│   ├── video_00000.mp4
│   ├── scene_00000.json
│   └── ...
├── val/
│   ├── video_10000.mp4
│   ├── scene_10000.json
│   └── ...
└── questions/
    ├── train.json
    ├── validation.json
    └── test.json
```

## Scripts

### 1. scene_converter.py

Converts CLEVRER scene JSON annotations to physics state tensors.

**Key functions:**
- `load_clevrer_scene()` - Parse scene annotation
- `clevrer_scene_to_state_tensor()` - Convert to [T, N, D] tensor

**State vector (28D):**
```
[0:3]   - Position (x, y, z)
[3:6]   - Velocity (vx, vy, vz)
[6]     - Mass
[7]     - Radius
[8:11]  - Shape one-hot (sphere, cube, cylinder)
[11:19] - Color one-hot (8 colors)
[19:21] - Material one-hot (metal, rubber)
[21:28] - Reserved
```

### 2. generate_physics_sequences_from_clevrer.py

Generates `physics_sequences.pt` for ablation studies.

```bash
python generate_physics_sequences_from_clevrer.py \
    --clevrer_dir /path/to/CLEVRER \
    --output physics_sequences.pt \
    --num_samples 1000 \
    --seq_length 128
```

### 3. clevrer_to_training_data.py

Converts CLEVRER questions to PhysicsReasoningDataset format.

**Question types generated:**
- `CAUSAL_CHAIN` - "What caused X?" (explanatory)
- `FUTURE_PREDICTION` - "What will happen next?" (predictive)
- `COUNTERFACTUAL_REASONING` - "What if X were removed?" (counterfactual)

```bash
python clevrer_to_training_data.py \
    --clevrer_dir /path/to/CLEVRER \
    --output training_data.json \
    --question_types causal,predictive,counterfactual
```

### 4. convert_to_compact.py

Converts processed data to compact HDF5 format.

```bash
python convert_to_compact.py \
    --input_dir ./processed \
    --output clevrer_training.h5
```

### 5. hdf5_dataset.py

PyTorch Dataset class for loading HDF5 training data.

```python
from hdf5_dataset import CLEVRERHDF5Dataset

dataset = CLEVRERHDF5Dataset(
    hdf5_path="clevrer_training.h5",
    question_types=["explanatory", "predictive", "counterfactual"]
)
```

## Output Format

### HDF5 Structure

```
clevrer_training.h5
├── physics_states/     # [N, T, O, D] float32
├── object_masks/       # [N, O] bool
├── questions/          # [N] string
├── answers/            # [N] string
├── question_types/     # [N] string
└── metadata/
    ├── num_samples
    ├── seq_length
    └── state_dim
```

## Notes

- CLEVRER uses 25 FPS video
- Velocities are derived from position deltas
- Maximum 10 objects per scene
- Exclude descriptive questions for causal reasoning training
