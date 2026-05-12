# QA Data Generation

Standalone tools for generating QA datasets for the Physics-LLM Adapter.

Output format: HDF5 for reproducibility and cross-platform compatibility.

## Files

- `qa_generator.py` - Core QA generation logic with question templates
- `generate_qa_dataset.py` - Standalone script to generate QA cache files

## Usage

### Generate QA Dataset

```bash
# From this directory
python generate_qa_dataset.py \
    --data_dir $PHYSICS_DATA_DIR \
    --output adapter_qa_cache.h5 \
    --num_samples 10000 \
    --qa_per_sample 3
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | (required) | Directory containing HDF5 physics data |
| `--output` | `adapter_qa_cache.h5` | Output file path (HDF5 format) |
| `--num_samples` | 10000 | Number of physics samples to load |
| `--qa_per_sample` | 3 | QA pairs per physics sample |
| `--max_objects` | 20 | Maximum objects per scene |
| `--state_dim` | 28 | State dimension |
| `--seed` | 42 | Random seed |

### Output Format (HDF5)

```
adapter_qa_cache.h5
├── states/          # [N, num_objects, state_dim] float32
├── masks/           # [N, num_objects] float32
├── questions/       # [N] variable-length string
├── answers/         # [N] variable-length string
├── question_types/  # [N] variable-length string
├── metadata/        # [N] JSON string
├── numerical_targets/ # [N] JSON string
└── attrs:
    ├── num_samples
    ├── state_shape
    └── format_version
```

### Loading HDF5 Data

```python
import h5py
import torch
import json

with h5py.File('adapter_qa_cache.h5', 'r') as f:
    states = torch.tensor(f['states'][:])
    masks = torch.tensor(f['masks'][:])
    questions = [q for q in f['questions'][:]]
    answers = [a for a in f['answers'][:]]
    metadata = [json.loads(m) for m in f['metadata'][:]]
```

## Question Types

The generator supports 24 question types across 5 categories:

1. **Basic Physical Properties** - Object count, position, velocity, mass, motion direction
2. **Physical Quantities** - Kinetic energy, momentum, relative velocity, distance, comparisons
3. **Predictive Reasoning** - Collision prediction, trajectory extrapolation, time to event
4. **Relational Reasoning** - Proximity, containment, relative position, contact state
5. **CLEVRER-Style** - Causal chain, future prediction, counterfactual reasoning

## For Colab

After generating the dataset, upload `adapter_qa_cache.pt` to:
```
/content/drive/MyDrive/physics_action_predictor/data/physics_former_adapter/adapter_qa_cache.pt
```
