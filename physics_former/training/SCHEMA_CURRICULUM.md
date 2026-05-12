# Schema Curriculum Learning

## Overview

The PhysicsFormer training uses **progressive schema curriculum** to gradually introduce physics concepts from simple to complex. This improves learning stability and final accuracy.

## Two-Level Curriculum System

### 1. Sequence Length Curriculum (Existing)
- **What**: Progressively increases sequence length from 32 → 64 → 128 → 256 steps
- **Why**: Longer sequences are harder to learn (more temporal dependencies)
- **When**: Progresses after 5+ epochs with no improvement (convergence)
- **File**: `progressive_curriculum.py`

### 2. Schema Curriculum (NEW)
- **What**: Progressively introduces physics schemas from 8 difficulty groups
- **Why**: Complex schemas (chaos, multi-scale) are harder to predict
- **When**: Should progress as model masters each group
- **File**: `datasets/hdf5_physics_dataset.py`, `datasets/cached_physics_dataset.py`

## Schema Groups (Progressive Difficulty)

### Group 1: Basic Single-Object Physics (6 schemas)
**Easiest** - Single objects, simple dynamics
- `projectile_motion` - Parabolic trajectories
- `pendulum_simple` - Single oscillator
- `friction_sliding` - Basic friction
- `friction_rolling` - Rolling motion
- `drag_force` - Air resistance
- `spring_damped` - Damped oscillation

### Group 2: Two-Object Interactions (4 schemas)
**Easy** - Pairwise interactions
- `collision_elastic` - Energy conservation
- `collision_inelastic` - Energy loss
- `attraction_repulsion` - Forces between objects
- `gravitational_slingshot` - Gravity assist

### Group 3: Constrained Systems (6 schemas)
**Medium** - Mechanical constraints
- `circular_motion` - Circular paths
- `orbit_circular` - Stable orbits
- `orbit_elliptical` - Elliptical orbits
- `pulley_system` - Mechanical advantage
- `stack_balance` - Stacking stability
- `bridge_stability` - Structural support

### Group 4: Multi-Object Dynamics (6 schemas)
**Medium-Hard** - Multiple interacting objects
- `pendulum_double` - Coupled oscillators
- `cause_effect_chain` - Causal sequences
- `partition_merge` - Splitting/combining
- `multi_scale_interaction` - Different scales *(excluded by default)*

### Group 5: Boundaries and Constraints (7 schemas)
**Hard** - Spatial boundaries and containment
- `boundary_deformation` - Flexible boundaries
- `barrier_breakthrough` - Breaking through walls
- `bottleneck_constraint` - Flow restriction
- `enclosure_protection` - Containment
- `escape_containment` - Breaking free
- `permeable_boundary` - Selective passage
- `container_overflow` - Capacity limits *(excluded - 19.8GB)*

### Group 6: Equilibrium and Transitions (6 schemas)
**Hard** - State changes and thresholds
- `equilibrium_dynamic` - Dynamic balance
- `critical_point` - Phase transitions
- `threshold_trigger` - Activation thresholds
- `saturation_limit` - Capacity limits *(excluded - 9.2GB)*
- `hysteresis` - History dependence *(excluded)*
- `irreversibility` - One-way processes *(excluded)*

### Group 7: Patterns and Synchronization (5 schemas)
**Very Hard** - Collective behavior
- `synchronization` - Phase locking
- `interference_constructive` - Wave interference
- `symmetry_breaking` - Symmetry loss
- `resilience_recovery` - Return to equilibrium
- `emergent_pattern` - Self-organization *(excluded - 10.6GB)*

### Group 8: Chaos and Complexity (3 schemas)
**Hardest** - Chaotic and unpredictable
- `exponential_growth` - Runaway growth
- `chaos_driven_oscillator` - Chaotic forcing *(excluded)*
- `chaos_double_pendulum` - Deterministic chaos *(excluded)*
- `hierarchy_cascade` - Cascading effects *(excluded)*
- `chain_reaction` - Sequential dependencies *(excluded)*

## Excluded Schemas (10 total)

These schemas are excluded from training due to:
- **Chaotic dynamics**: Exponential sensitivity to initial conditions
- **Large datasets**: 10-20GB files (memory/time intensive)
- **Path-dependent**: History-dependent, non-reversible behavior
- **Multi-scale**: Complex interactions across scales

```python
excluded_schemas = [
    'chaos_double_pendulum.h5',      # Chaotic
    'chaos_driven_oscillator.h5',    # Chaotic
    'container_overflow.h5',         # 19.8 GB
    'emergent_pattern.h5',           # 10.6 GB
    'saturation_limit.h5',           # 9.2 GB
    'irreversibility.h5',            # Path-dependent
    'hysteresis.h5',                 # History-dependent
    'multi_scale_interaction.h5',    # Multiple scales
    'hierarchy_cascade.h5',          # Cascading effects
    'chain_reaction.h5'              # Sequential dependencies
]
```

**Training schemas: 43 - 10 = 33 schemas**

## Configuration

### Current Settings

```python
# config.py
schema_curriculum_level = 1  # Start with Group 1 only (6 schemas)
```

### Usage

```python
# Start with easiest schemas (Group 1)
dataset = CachedPhysicsDataset(
    data_dir=data_dir,
    schema_curriculum_level=1  # Only 6 basic schemas
)

# Progress to more complex (Groups 1-4)
dataset = CachedPhysicsDataset(
    data_dir=data_dir,
    schema_curriculum_level=4  # 6+4+6+6 = 22 schemas
)

# All allowed schemas (Groups 1-8, minus excluded)
dataset = CachedPhysicsDataset(
    data_dir=data_dir,
    schema_curriculum_level=8  # All 33 allowed schemas
)
```

## Progression Strategy

### Recommended Schedule

| Epochs | Schema Level | Schemas | Description |
|--------|--------------|---------|-------------|
| 1-5    | Level 1      | 6       | Basic single-object physics |
| 6-10   | Level 2      | 10      | Add two-object interactions |
| 11-15  | Level 3      | 16      | Add constrained systems |
| 16-20  | Level 4      | 22      | Add multi-object dynamics |
| 21-30  | Level 5      | 28      | Add boundaries (minus excluded) |
| 31-35  | Level 6      | 31      | Add equilibrium (minus excluded) |
| 36-40  | Level 7      | 32      | Add patterns (minus excluded) |
| 41-50  | Level 8      | 33      | Add complexity (minus excluded) |

### Automatic Progression (TODO)

Add to `progressive_curriculum.py`:

```python
class ProgressiveCurriculum:
    def __init__(self, ...):
        self.schema_level = 1
        self.schema_progression = [1, 2, 3, 4, 5, 6, 7, 8]
        self.min_epochs_per_schema_level = 5
    
    def should_progress_schema(self):
        """Check if should add more schema groups."""
        if self.schema_level >= 8:
            return False
        if self.epochs_in_phase < self.min_epochs_per_schema_level:
            return False
        if self.epochs_without_improvement < self.convergence_patience:
            return False
        return True
    
    def progress_schema_level(self):
        """Add next schema group."""
        self.schema_level += 1
        # Reload dataset with new schema level
        # Reset phase counters
```

## Benefits

1. **Faster Initial Learning**: Start with simple concepts
2. **Better Stability**: Avoid overwhelming model with chaos early
3. **Improved Accuracy**: Build strong foundation before complexity
4. **Reduced Memory**: Smaller datasets in early epochs
5. **Curriculum Transfer**: Skills from simple schemas transfer to complex

## Monitoring

Check training logs for:
```
[CURRICULUM] Loading 6 schemas from 1 groups:
  Group 1: 6 schemas
```

As training progresses, this should increase:
```
[CURRICULUM] Loading 22 schemas from 4 groups:
  Group 1: 6 schemas
  Group 2: 4 schemas
  Group 3: 6 schemas
  Group 4: 6 schemas
```

## Next Steps

1. ✅ Implement schema filtering in datasets
2. ✅ Add `schema_curriculum_level` to config
3. ✅ Pass level from config to datasets
4. ⏳ **TODO**: Auto-progression in training loop
5. ⏳ **TODO**: Save/load schema level in checkpoints
6. ⏳ **TODO**: Metrics per schema group
