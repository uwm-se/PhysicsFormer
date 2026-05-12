# CLS Pipeline Refactoring Plan

## Current State
- `cls_pipeline.py`: 2088 lines (too large!)
- Target: Break into modules, each <900 lines

## Completed Modules (in `modules/`)
✅ `training_loop.py` - Core epoch training (130 lines)
✅ `gradient_handler.py` - Gradient management (70 lines)
✅ `checkpoint_manager.py` - Checkpoint save/load (80 lines)
✅ `progress_tracker.py` - Progress bars & logging (70 lines)
✅ `consolidation_manager.py` - CLS memory & replay (95 lines)
✅ `optimizer_factory.py` - Optimizer & scheduler creation (120 lines)
✅ `validation_runner.py` - Validation logic (100 lines)
✅ `batch_processor.py` - Batch preparation (110 lines)
✅ `metrics_logger.py` - Metrics logging (150 lines)

**Total: ~925 lines in 9 reusable modules**

## Refactoring Strategy

### Phase 1: Extract to Modules (DONE)
Created standalone modules that can be imported and used.

### Phase 2: Refactor cls_pipeline.py
Replace inline code with module imports:

1. **Import modules at top**
   ```python
   from .modules import (
       TrainingLoop, GradientHandler, CheckpointManager,
       ProgressTracker, ConsolidationManager, OptimizerFactory,
       ValidationRunner, BatchProcessor, MetricsLogger
   )
   ```

2. **Replace optimizer creation** (~50 lines → 5 lines)
   - Old: Manual AdamW + scheduler setup
   - New: `OptimizerFactory.create_optimizer()` + `create_scheduler()`

3. **Replace gradient handling** (~30 lines → 3 lines)
   - Old: Manual clip_grad_norm + clip_grad_value + NaN checks
   - New: `GradientHandler.clip_and_check()`

4. **Replace checkpoint logic** (~80 lines → 10 lines)
   - Old: Manual torch.save/load with path management
   - New: `CheckpointManager.save()` / `load()`

5. **Replace progress logging** (~100 lines → 15 lines)
   - Old: Manual progress bar + ETA calculation
   - New: `ProgressTracker.update()` + `get_summary()`

6. **Replace CLS memory** (~150 lines → 20 lines)
   - Old: Manual memory buffers + sampling
   - New: `ConsolidationManager.store_experience()` + `sample_for_consolidation()`

7. **Replace batch preparation** (~60 lines → 5 lines)
   - Old: Manual device movement + task-specific prep
   - New: `BatchProcessor.prepare_batch()`

8. **Replace validation** (~120 lines → 10 lines)
   - Old: Manual validation loop
   - New: `ValidationRunner.validate()`

9. **Replace metrics logging** (~80 lines → 10 lines)
   - Old: Manual file writing + tensorboard
   - New: `MetricsLogger.log_training_step()` etc.

### Expected Result
- `cls_pipeline.py`: 2088 lines → **~800 lines** (61% reduction)
- Still fully functional
- Much more maintainable
- Modules can be reused in other projects

## Next Steps
1. Delete `cls_pipeline_v2.py` (not needed)
2. Gradually refactor `cls_pipeline.py` section by section
3. Test after each refactoring step
4. Keep original as backup until fully tested

## Benefits
- ✅ Smaller, more focused files
- ✅ Easier to test individual components
- ✅ Reusable across projects
- ✅ Easier to understand and maintain
- ✅ Single responsibility principle
