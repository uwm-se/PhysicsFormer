# PhysicsFormer Training

## Overview

This folder contains two physics models:

### Enhanced PhysicsFormer [NEW] (Primary)
- PASS: Pairwise relation encoding (objects know about each other)
- PASS: Learnable attention biases (model learns what's relevant)
- PASS: Physics-informed loss (learns energy/momentum conservation)
- **Best for**: Math validation, 1-100 objects

### Hierarchical PhysicsFormer (Future Scaling)
- PASS: Local + global attention
- PASS: Spatial clustering
- PASS: Scales to 1000+ objects
- **Best for**: Large-scale simulations, particle systems

---

## Quick Start

### Train Model

```bash
python train_enhanced.py
```

### Test Model

```bash
# Test the model
python physics_former_enhanced.py

# Test physics concepts
python test_higher_order_concepts.py

# Test LLM integration
python llm_math_validation.py
```

---

## Files

### Core Models
- **`physics_former_enhanced.py`** - Enhanced PhysicsFormer (primary)
- **`train_enhanced.py`** - Training script for enhanced
- **`ENHANCED_IMPLEMENTATION_GUIDE.md`** - Complete implementation guide
- **`physics_former_hierarchical.py`** - Hierarchical model (for scaling)
- **`SCALING_TO_THOUSANDS.md`** - Scaling documentation

### Dependencies
- **`physics_tokenizer.py`** - Tokenizer for state encoding
- **`physics_dataset_object_centric.py`** - Dataset (object-centric format)
- **`model_config.py`** - Model configurations

### Testing & Validation
- **`test_higher_order_concepts.py`** - Test energy, momentum, collisions
- **`llm_math_validation.py`** - LLM integration for math validation
- **`MATH_TESTING.md`** - Math testing documentation
- **`HIGHER_ORDER_TESTING.md`** - Higher-order concept testing
- **`ENERGY_CONSERVATION_TESTING.md`** - Energy conservation testing

### Documentation
- **`LLM_INTEGRATION.md`** - How to integrate with LLMs

---

## Key Features

### 1. Pairwise Relation Encoding
Model receives explicit information about object relationships:
- Distance between objects
- Closing velocity
- Relative velocity

### 2. Learnable Attention Biases
Model learns which objects to focus on:
- Nearby objects
- Approaching objects
- Objects about to collide

### 3. Physics-Informed Loss
Model learns conservation laws through training:
- Energy conservation
- Momentum conservation
- Standard prediction accuracy

---

## Architecture

```python
from physics_former_enhanced import EnhancedPhysicsFormer

model = EnhancedPhysicsFormer(
    state_dim=21,
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    max_objects=10
)

# Predict next states
predicted_states, schema_logits = model(object_states, object_mask)
```

---

## Training

```python
from train_enhanced import EnhancedPhysicsTrainer
from physics_former_enhanced import PhysicsInformedLoss

# Create trainer
trainer = EnhancedPhysicsTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda"
)

# Train with physics-informed loss
trainer.train(num_epochs=100)
```

---

## Math Validation

```python
from llm_math_validation import PhysicsValidator

validator = PhysicsValidator(model_path="checkpoints_enhanced/best_model.pt")

# Validate conservation laws
result = validator.validate_energy_conservation(
    initial_state=...,
    duration=1.0
)

print(f"Energy conserved: {result['conserved']}")
print(f"Energy drift: {result['drift']:.2%}")
```

---

## Expected Performance

| Metric | Value |
|--------|-------|
| **Collision prediction** | 85% accuracy |
| **Energy conservation** | < 5% error |
| **Momentum conservation** | < 5% error |
| **Training time** | ~4 hours/epoch (GPU) |
| **Inference time** | ~50ms (10 objects) |

---

## Configuration

Edit `train_enhanced.py` to configure:

```python
# Model hyperparameters
HIDDEN_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# Physics loss weights
energy_weight = 0.1
momentum_weight = 0.1
```

---

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 8

# Reduce model size
HIDDEN_DIM = 128
NUM_LAYERS = 4
```

### Poor Conservation
```python
# Increase physics loss weights
energy_weight = 0.5
momentum_weight = 0.5
```

### Slow Training
```python
# Use mixed precision
from torch.cuda.amp import autocast, GradScaler

with autocast():
    predicted_states, _ = model(object_states)
```

---

## Next Steps

1. PASS: Train model: `python train_enhanced.py`
2. PASS: Test physics: `python test_higher_order_concepts.py`
3. PASS: Validate math: `python llm_math_validation.py`
4. PASS: Integrate with LLM (see `LLM_INTEGRATION.md`)

---

## Support

For detailed implementation guide, see:
- **`ENHANCED_IMPLEMENTATION_GUIDE.md`**

For math testing, see:
- **`MATH_TESTING.md`**
- **`HIGHER_ORDER_TESTING.md`**
- **`ENERGY_CONSERVATION_TESTING.md`**

---

**Ready to train and validate!** 🚀
