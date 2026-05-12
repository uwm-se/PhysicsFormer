# Isaac Sim Integration Setup Guide

## Prerequisites
- Windows 10/11
- RTX 4080+ GPU (you have RTX 4080 Laptop ✓)
- Python 3.11
- ~50GB disk space

## Step 1: Create Python 3.11 Virtual Environment

```powershell
# Create a new conda environment with Python 3.11
conda create -n isaac_sim python=3.11 -y
conda activate isaac_sim
```

## Step 2: Install Isaac Sim via pip

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install Isaac Sim (this will download ~15-20GB of packages)
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Install CUDA-enabled PyTorch
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

## Step 3: Enable Long Path Support (Windows)

Run as Administrator in PowerShell:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## Step 4: Verify Installation

```powershell
# First run will download extensions (~10 min)
isaacsim --help
```

## Step 5: Clone OmniIsaacGymEnvs

```powershell
cd <repo-root>/physics_former/data_generation/isaac_sim
git clone https://github.com/isaac-sim/OmniIsaacGymEnvs.git
cd OmniIsaacGymEnvs
pip install -e .
```

## Step 6: Test a Task

```powershell
# Run Franka Cabinet task (manipulation)
python scripts/rlgames_train.py task=FrankaCabinet num_envs=4 headless=True
```

## Available Tasks for Our Use Case

| Task | Description | Relevance |
|------|-------------|-----------|
| FrankaCabinet | Robot opens drawer | Manipulation |
| FrankaDeformable | Robot manipulates soft tube | Deformable physics |
| ShadowHand | Dexterous hand manipulation | Object rotation |
| AllegroHand | In-hand manipulation | Grasping |
| BallBalance | Balance ball on platform | Physics prediction |

## Next Steps

After installation, run:
```powershell
python generate_isaac_data.py --task FrankaCabinet --episodes 1200
```

This will generate HDF5 data compatible with our PhysicsFormer training pipeline.
