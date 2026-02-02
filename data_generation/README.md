# Data Generation Pipeline

This folder contains all scripts needed to reproduce the training and evaluation data for the Physics-LLM project.

**Output Format**: All scripts output HDF5 format for reproducibility and cross-platform compatibility.

## Directory Structure

```
data_generation/
├── isaac_sim/           # Isaac Sim physics schema generation
├── clevrer/             # CLEVRER benchmark data conversion
├── qa_generation/       # Question-Answer dataset generation
└── README.md           # This file
```

## Data Generation Order

For full reproducibility, run the scripts in this order:

### Step 1: Generate Physics Schemas (Isaac Sim)

```bash
cd isaac_sim/
python generate_robotics_schemas.py --output_dir ./schemas
python generate_isaac_hdf5.py --schema_dir ./schemas --output physics_data.h5
```

See [isaac_sim/README.md](isaac_sim/README.md) for detailed setup instructions.

### Step 2: Convert CLEVRER Benchmark Data

```bash
cd clevrer/
# Convert CLEVRER scenes to physics state tensors (outputs .h5)
python generate_physics_sequences_from_clevrer.py \
    --clevrer-path /path/to/CLEVRER/scenes/validation \
    --output-dir ./clevrer_physics

# Convert questions to training data (outputs .h5)
python clevrer_to_training_data.py \
    --clevrer_dir /path/to/CLEVRER \
    --output clevrer_training_data.h5 \
    --batch
```

See [clevrer/README.md](clevrer/README.md) for CLEVRER dataset requirements.

### Step 3: Generate QA Training Data

```bash
cd qa_generation/
python generate_qa_dataset.py \
    --data_dir /path/to/physics_hdf5 \
    --output adapter_qa_cache.h5 \
    --num_samples 10000
```

See [qa_generation/README.md](qa_generation/README.md) for question type options.

## Output Files (All HDF5)

| File | Description | Size (approx) |
|------|-------------|---------------|
| `physics_data.h5` | Isaac Sim physics states | ~2GB |
| `physics_sequences.h5` | CLEVRER physics sequences | ~500MB |
| `clevrer_training_data.h5` | CLEVRER questions + physics | ~300MB |
| `adapter_qa_cache.h5` | Question-Answer pairs | ~100MB |

## Why HDF5?

- **Cross-platform**: Works on Windows, Linux, macOS, Colab
- **Compression**: gzip compression reduces file sizes
- **Partial loading**: Load specific samples without loading entire file
- **Metadata**: Store attributes alongside data
- **Industry standard**: Widely supported in scientific computing

## Hardware Requirements

- **Isaac Sim generation**: NVIDIA GPU (RTX 3090+ recommended)
- **CLEVRER conversion**: CPU-only, 16GB RAM
- **QA generation**: CPU-only, 8GB RAM

## Citation

If you use this data generation pipeline, please cite:

```bibtex
@inproceedings{pokora2026physics,
  title={Physics-Grounded Language Models for Embodied Reasoning},
  author={Pokora, Jesse},
  booktitle={COMPSAC 2026},
  year={2026}
}
```
