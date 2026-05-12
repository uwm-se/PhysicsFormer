# Training Pipeline Comparison & Refactoring Plan

## Current State Analysis

### CLS Training Pipeline (`cls_pipeline.py`)
**Purpose:** Prevent catastrophic forgetting using Complementary Learning Systems

**Key Features:**
1. PASS: **CLS Memory System** - Hippocampal buffer + consolidation
2. PASS: **Metrics Logging** - Comprehensive experiment tracking
3. PASS: **Curriculum Learning** - Progressive difficulty
4. PASS: **Auxiliary Tasks** - Richer representations
5. PASS: **Contrastive Learning** - Better embeddings
6. PASS: **Enhanced Number Encoder** - Better generalization
7. PASS: **Advanced Optimizer** - AdamW + warmup + cosine
8. PASS: **Multi-task Validation** - Forgetting detection
9. PASS: **Schema Protection** - Freeze encoder for later stages
10. PASS: **Adaptive Consolidation** - Prioritize weak tasks

### Improved Ablation Pipeline (`improved_ablation_pipeline.py`)
**Purpose:** Ablation study comparing with/without improvements

**Key Features:**
1. FAIL: **No CLS Memory** - Sequential training only
2. FAIL: **No Metrics Logging** - Missing comprehensive tracking
3. PASS: **Curriculum Learning** - Same as CLS
4. PASS: **Auxiliary Tasks** - Same as CLS
5. PASS: **Contrastive Learning** - Same as CLS
6. PASS: **Enhanced Number Encoder** - Same as CLS
7. PASS: **Advanced Optimizer** - Same as CLS
8. FAIL: **No Multi-task Validation** - No forgetting detection
9. FAIL: **No Schema Protection** - All parameters trainable
10. FAIL: **No Adaptive Consolidation** - No replay mechanism

## Missing Features in Ablation Pipeline

### Critical Missing Features:

1. **Metrics Logging** FAIL:
   - No comprehensive experiment tracking
   - No dataset statistics
   - No system resource monitoring
   - No automatic report generation

2. **Multi-task Validation** FAIL:
   - No forgetting detection
   - No cross-task performance tracking
   - No validation history

3. **Schema Protection** FAIL:
   - All parameters trainable in all stages
   - Risk of overwriting physics grounding

4. **Checkpoint Management** FAIL:
   - Basic checkpointing only
   - No stage-specific checkpoints
   - No CLS memory saving (N/A)

5. **Progress Tracking** FAIL:
   - No detailed statistics
   - No consolidation metrics
   - No forgetting events tracking

## Shared Components to Extract

### 1. Data Loading Module
**File:** `training/common/data_loading.py`

```python
def create_dataloaders(config, batch_size=64):
    """
    Create dataloaders for all training stages.
    
    Shared by both pipelines.
    """
    # Physics, Counting, Arithmetic, Symbolic loaders
    pass
```

### 2. Metrics Logging Module
**File:** `training/common/metrics_integration.py`

```python
def setup_metrics_logger(experiment_name, config, track_gpu=True):
    """
    Initialize metrics logger with standard configuration.
    
    Shared by both pipelines.
    """
    pass

def log_training_metrics(logger, epoch, step, loss, lr, grad_norm, **kwargs):
    """
    Log training metrics in standard format.
    """
    pass
```

### 3. Validation Module
**File:** `training/common/validation.py`

```python
def validate_all_tasks(model, validation_loaders, epoch):
    """
    Multi-task validation to detect catastrophic forgetting.
    
    Should be used by both pipelines.
    """
    pass

def detect_forgetting(validation_history, task, current_accuracy, threshold=0.1):
    """
    Detect if forgetting occurred for a task.
    """
    pass
```

### 4. Checkpoint Management Module
**File:** `training/common/checkpointing.py`

```python
def save_checkpoint(model, optimizer, stage, epoch, path, **metadata):
    """
    Save checkpoint with metadata.
    
    Shared format for both pipelines.
    """
    pass

def load_checkpoint(path, model=None, optimizer=None):
    """
    Load checkpoint with validation.
    """
    pass
```

### 5. Training Utilities Module
**File:** `training/common/training_utils.py`

```python
def compute_gradient_norm(model):
    """
    Compute total gradient norm.
    """
    pass

def freeze_encoder(model):
    """
    Freeze encoder for schema protection.
    """
    pass

def unfreeze_all(model):
    """
    Unfreeze all parameters.
    """
    pass

def print_training_progress(batch_idx, total_batches, loss, **metrics):
    """
    Print standardized training progress.
    """
    pass
```

### 6. Configuration Module
**File:** `training/common/config_utils.py`

```python
def get_config(config_name):
    """
    Get configuration by name.
    
    Shared config loading.
    """
    pass

def override_config(config, **overrides):
    """
    Override config parameters.
    """
    pass
```

## Refactoring Plan

### Phase 1: Extract Shared Modules
1. Create `training/common/` directory
2. Extract data loading logic
3. Extract metrics integration
4. Extract validation logic
5. Extract checkpoint management
6. Extract training utilities

### Phase 2: Update CLS Pipeline
1. Import from common modules
2. Remove duplicated code
3. Keep CLS-specific logic (memory, consolidation)
4. Test thoroughly

### Phase 3: Update Ablation Pipeline
1. Import from common modules
2. Add missing features:
   - Metrics logging
   - Multi-task validation
   - Schema protection (optional flag)
3. Keep ablation-specific logic (no CLS)
4. Test thoroughly

### Phase 4: Ensure Comparability
1. Both use same metrics logger
2. Both use same validation
3. Both use same checkpointing
4. Both use same data loading
5. **Only difference:** CLS memory vs no CLS

## Implementation Priority

### High Priority (Do First):
1. PASS: **Metrics Logging** - Already integrated in CLS, add to ablation
2. PASS: **Data Loading** - Extract to shared module
3. PASS: **Validation** - Extract and add to ablation
4. PASS: **Checkpoint Management** - Standardize across both

### Medium Priority:
5. **Training Utilities** - Extract common functions
6. **Configuration** - Standardize config loading
7. **Progress Tracking** - Unified statistics

### Low Priority:
8. **Visualization** - Shared plotting utilities
9. **Reporting** - Unified report generation

## Expected Outcome

### After Refactoring:

**CLS Pipeline:**
```python
from training.common import (
    create_dataloaders,
    setup_metrics_logger,
    validate_all_tasks,
    save_checkpoint,
    compute_gradient_norm
)

class CLSTrainingPipeline:
    def __init__(self, ...):
        # CLS-specific: Memory system
        self.cls_memory = CLSMemorySystem(...)
        
        # Shared: Metrics, validation, etc.
        self.logger = setup_metrics_logger(...)
    
    def train_stage_with_cls(self, ...):
        # CLS-specific: Consolidation
        if self.use_cls:
            self.cls_memory.consolidate(...)
        
        # Shared: Training loop, metrics, validation
        log_training_metrics(...)
        validate_all_tasks(...)
```

**Ablation Pipeline:**
```python
from training.common import (
    create_dataloaders,
    setup_metrics_logger,
    validate_all_tasks,
    save_checkpoint,
    compute_gradient_norm
)

class AblationPipeline:
    def __init__(self, ...):
        # No CLS memory
        
        # Shared: Metrics, validation, etc.
        self.logger = setup_metrics_logger(...)
    
    def train_stage(self, ...):
        # No consolidation
        
        # Shared: Training loop, metrics, validation
        log_training_metrics(...)
        validate_all_tasks(...)
```

## Benefits

### Code Quality:
- PASS: DRY principle (Don't Repeat Yourself)
- PASS: Single source of truth
- PASS: Easier maintenance
- PASS: Consistent behavior

### Comparability:
- PASS: Same metrics format
- PASS: Same validation logic
- PASS: Same checkpointing
- PASS: **Only difference:** CLS vs no CLS

### Testing:
- PASS: Test shared modules once
- PASS: Test pipeline-specific logic separately
- PASS: Easier to verify correctness

### Features:
- PASS: Ablation pipeline gets all missing features
- PASS: Both pipelines stay in sync
- PASS: New features added to both automatically

## Next Steps

1. **Create common modules directory**
2. **Extract data loading** (highest reuse)
3. **Add metrics to ablation** (critical missing feature)
4. **Extract validation** (needed by both)
5. **Standardize checkpointing**
6. **Test both pipelines**
7. **Update documentation**

## Summary

**Current State:**
- CLS pipeline: Feature-complete PASS:
- Ablation pipeline: Missing 5 critical features FAIL:
- Code duplication: ~60% FAIL:

**After Refactoring:**
- CLS pipeline: Feature-complete PASS:
- Ablation pipeline: Feature-complete PASS:
- Code duplication: ~10% PASS:
- Comparability: Perfect PASS:

**The pipelines will be truly comparable with only the CLS memory system as the differentiator!**
