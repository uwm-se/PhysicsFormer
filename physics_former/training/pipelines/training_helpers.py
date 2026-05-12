"""
Training Helper Functions

Extracted from cls_pipeline.py to reduce file size and improve organization.
"""

import torch
import time
from typing import Dict, List


def print_training_explainer(epoch: int, batch_idx: int, stage: int = 1):
    """
    Print educational explanations about what's happening during training.
    Helps students understand the training process.
    """
    explanations = {
        # First epoch, first batch
        (0, 0): """
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎓 PHYSICS TRANSFORMER TRAINING - EDUCATIONAL MODE                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WHAT IS HAPPENING:                                                          ║
║  The model is learning to predict how objects move in physics simulations.   ║
║                                                                              ║
║  INPUT:  Current state of objects (position, velocity, rotation)             ║
║  OUTPUT: Predicted change (delta) for next timestep                          ║
║  GOAL:   Minimize difference between predicted and actual physics            ║
║                                                                              ║
║  KEY METRICS TO WATCH:                                                       ║
║  • Loss: How wrong the predictions are (lower = better)                      ║
║  • Grad: Gradient magnitude - how much weights will change                   ║
║  • LR:   Learning rate - step size for weight updates                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
        # After 50 batches
        (0, 50): """
┌─ 📊 TRAINING INSIGHT ─────────────────────────────────────────────────────────┐
│ After 50 batches, the model has seen ~800 physics simulations.               │
│                                                                              │
│ WHAT'S HAPPENING NOW:                                                        │
│ • The model is learning basic patterns (gravity, momentum)                   │
│ • Loss should be decreasing - if not, learning rate may be wrong             │
│ • Gradients should be stable (not inf or 0) - if not, normalization issue    │
└──────────────────────────────────────────────────────────────────────────────┘
""",
        # End of first epoch
        (0, -1): """
┌─ 🏁 EPOCH 1 COMPLETE ─────────────────────────────────────────────────────────┐
│ The model has seen ALL training data once. This is called one "epoch".       │
│                                                                              │
│ WHAT TO EXPECT:                                                              │
│ • Epoch 1 loss is usually high - the model is just starting to learn         │
│ • Each subsequent epoch should have lower loss (model improves)              │
│ • If loss increases, something is wrong (learning rate too high?)            │
│                                                                              │
│ CURRICULUM LEARNING:                                                         │
│ • We start with simple physics (dropping objects)                            │
│ • As the model improves, we add harder scenarios (collisions, stacking)      │
│ • This is like teaching a child: crawl → walk → run                          │
└──────────────────────────────────────────────────────────────────────────────┘
""",
    }
    
    key = (epoch, batch_idx)
    if key in explanations:
        print(explanations[key])
    elif epoch == 0 and batch_idx == -1:  # End of epoch marker
        print(explanations[(0, -1)])


def explain_metric(metric_name: str, value: float) -> str:
    """
    Return an educational explanation for a metric value.
    """
    if metric_name == "loss":
        if value > 10:
            return "Very high - model predictions are far from reality"
        elif value > 1:
            return "High - model is still learning basic patterns"
        elif value > 0.1:
            return "Good - model understands physics reasonably well"
        else:
            return "Excellent - model predictions are very accurate"
    
    elif metric_name == "grad_norm":
        if value == float('inf') or value > 1000:
            return "⚠️ EXPLODING - training unstable, need lower LR or better normalization"
        elif value > 100:
            return "Very high - weights changing rapidly, may be unstable"
        elif value > 10:
            return "High - active learning, weights changing significantly"
        elif value > 0.1:
            return "Normal - healthy gradient flow"
        else:
            return "Low - slow learning, may need higher LR"
    
    elif metric_name == "pred_target_ratio":
        if value > 100:
            return "Model outputs are much smaller than targets - needs more training"
        elif value > 10:
            return "Model is under-predicting - learning to scale up"
        elif value > 2:
            return "Getting closer - model learning the right magnitude"
        else:
            return "Good match - model predicting correct scale"
    
    return ""


def log_training_progress(
    batch_idx: int,
    total_batches: int,
    epoch_losses: List[float],
    epoch_start_time: float,
    batch_size: int,
    batch_start_times: List[float],
    optimizer,
    model,
    metrics: Dict = None,
    last_grad_norm: float = None
):
    """
    Log training progress with progress bar and statistics.
    
    Args:
        batch_idx: Current batch index
        total_batches: Total number of batches
        epoch_losses: List of losses for current epoch
        epoch_start_time: Start time of epoch
        batch_size: Batch size
        batch_start_times: List of batch processing times
        optimizer: Optimizer (for learning rate)
        model: Model (for gradient norm)
        metrics: Optional metrics dict
    """
    from training.common.training_utils import compute_gradient_norm
    
    # Calculate statistics
    recent_losses = epoch_losses[-50:] if len(epoch_losses) >= 50 else epoch_losses
    avg_loss = sum(recent_losses) / len(recent_losses)
    min_loss = min(recent_losses)
    max_loss = max(recent_losses)
    progress_pct = (batch_idx / total_batches) * 100 if total_batches > 0 else 0
    
    # Get current learning rate and use passed gradient norm (before zeroing)
    current_lr = optimizer.param_groups[0]['lr']
    grad_norm = last_grad_norm if last_grad_norm is not None else 0.0
    
    # GPU memory
    gpu_mem_str = ""
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
        gpu_mem_str = f" | GPU: {gpu_mem_allocated:.2f}/{gpu_mem_reserved:.2f}GB"
    
    # Throughput
    elapsed = time.time() - epoch_start_time
    samples_per_sec = (batch_idx * batch_size) / elapsed if elapsed > 0 and batch_size else 0
    
    # ETA
    if len(batch_start_times) >= 10:
        avg_batch_time = sum(batch_start_times[-10:]) / 10
        eta_seconds = avg_batch_time * (total_batches - batch_idx)
        eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds >= 60 else f"{eta_seconds:.0f}s"
    else:
        eta_str = "calculating..."
    
    # Progress bar
    bar_width = 40
    filled = int(bar_width * progress_pct / 100)
    bar = '█' * filled + '░' * (bar_width - filled)
    
    # Print progress - use carriage return to overwrite single line
    batch_time = batch_start_times[-1] if batch_start_times else 0
    
    # Format gradient norm with warning if extreme
    if grad_norm > 100 or grad_norm == float('inf'):
        grad_str = f"Grad: \033[91m{grad_norm:.1f}\033[0m"  # Red for extreme
    elif grad_norm > 10:
        grad_str = f"Grad: \033[93m{grad_norm:.2f}\033[0m"  # Yellow for high
    else:
        grad_str = f"Grad: {grad_norm:.3f}"
    
    # Format loss with trend indicator
    if len(epoch_losses) >= 20:
        early_avg = sum(epoch_losses[:10]) / 10
        recent_avg = sum(epoch_losses[-10:]) / 10
        if recent_avg < early_avg * 0.95:
            trend = "\033[92m↓\033[0m"  # Green down arrow - improving
        elif recent_avg > early_avg * 1.05:
            trend = "\033[91m↑\033[0m"  # Red up arrow - degrading
        else:
            trend = "→"  # Stable
    else:
        trend = ""
    
    # Clear line and print everything on one line
    progress_line = (
        f"\r  [{bar}] {progress_pct:5.1f}% | "
        f"Batch {batch_idx:4d}/{total_batches} | ETA: {eta_str} | "
        f"Loss: {avg_loss:.4f}{trend} | "
        f"LR: {current_lr:.2e} | {grad_str} | "
        f"{samples_per_sec:.1f} samp/s{gpu_mem_str}"
    )
    print(progress_line, end='', flush=True)


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    epoch_losses: List[float],
    epoch_time: float,
    batch_size: int,
    consolidation_count: int = 0,
    cls_memory = None
):
    """
    Print epoch summary statistics.
    
    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        epoch_losses: List of losses for the epoch
        epoch_time: Time taken for epoch
        batch_size: Batch size
        consolidation_count: Number of consolidations performed
        cls_memory: Optional CLS memory system
    """
    # Print newline to move past the progress bar
    print()
    
    if not epoch_losses:
        print(f"\n[WARN]  No batches processed in epoch {epoch+1}")
        return
    
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    min_epoch_loss = min(epoch_losses)
    max_epoch_loss = max(epoch_losses)
    final_loss = epoch_losses[-1]
    
    # Loss trend
    if len(epoch_losses) > 10:
        first_half = sum(epoch_losses[:len(epoch_losses)//2]) / (len(epoch_losses)//2)
        second_half = sum(epoch_losses[len(epoch_losses)//2:]) / (len(epoch_losses) - len(epoch_losses)//2)
        if second_half < first_half * 0.95:
            loss_trend = "[IMPROVE] Improving"
        elif second_half > first_half * 1.05:
            loss_trend = "[WARN]  Degrading"
        else:
            loss_trend = "[STABLE]  Stable"
    else:
        loss_trend = "[INFO]  Too few batches"
    
    # Throughput
    batches_per_sec = len(epoch_losses) / epoch_time if epoch_time > 0 else 0
    total_samples = len(epoch_losses) * batch_size if batch_size else len(epoch_losses) * 32
    samples_per_sec = total_samples / epoch_time if epoch_time > 0 else 0
    
    # Memory stats
    memory_str = ""
    if torch.cuda.is_available():
        gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**3
        memory_str = f"\n  Peak GPU Memory: {gpu_mem_peak:.2f}GB"
    
    print(f"\n{'='*70}")
    print(f"[STATS] EPOCH {epoch+1}/{total_epochs} SUMMARY")
    print(f"{'='*70}")
    print(f"\n[LOSS] Loss Statistics:")
    print(f"  Average: {avg_epoch_loss:.4f} | Final: {final_loss:.4f}")
    print(f"  Min: {min_epoch_loss:.4f} | Max: {max_epoch_loss:.4f}")
    print(f"  Trend: {loss_trend}")
    
    print(f"\n[CYCLE] Consolidation:")
    print(f"  Replays: {consolidation_count} ({consolidation_count/len(epoch_losses)*100:.1f}% of batches)")
    if cls_memory and consolidation_count > 0:
        print(f"  Memory sizes: {cls_memory.get_memory_sizes()}")
    
    print(f"\n[PERF] Performance:")
    print(f"  Epoch Time: {epoch_time:.1f}s")
    print(f"  Throughput: {batches_per_sec:.1f} batches/sec | {samples_per_sec:.1f} samples/sec")
    print(f"  Batches Processed: {len(epoch_losses)}{memory_str}")
    
    # ETA for remaining epochs
    if epoch < total_epochs - 1:
        remaining_epochs = total_epochs - (epoch + 1)
        eta_seconds = epoch_time * remaining_epochs
        eta_minutes = eta_seconds / 60
        print(f"\n[TIME]  Estimated Time Remaining:")
        if eta_minutes < 60:
            print(f"  {eta_minutes:.1f} minutes ({remaining_epochs} epochs left)")
        else:
            eta_hours = eta_minutes / 60
            print(f"  {eta_hours:.1f} hours ({remaining_epochs} epochs left)")
    
    print(f"{'='*70}\n")


def prepare_physics_batch(batch: Dict, constants) -> Dict:
    """
    Prepare batch for physics loss function by adding required keys.
    
    Args:
        batch: Dictionary with keys 'object_states', 'next_states', 'object_mask'
        constants: Constants module with BATCH_KEY_* and MASS_IDX
        
    Returns:
        Modified batch with additional keys including delta_states for position-invariant learning
    """
    from training.utils.tensor_ops import extract_sequence_input_for_prediction
    
    object_states = batch.get(constants.BATCH_KEY_OBJECT_STATES)
    next_states = batch.get(constants.BATCH_KEY_NEXT_STATES)
    
    if object_states is not None and next_states is not None:
        # Handle sequence dimension if present
        if object_states.dim() == 4:
            # Shape: [batch, seq_len, objects, features]
            current_states = extract_sequence_input_for_prediction(object_states)
            masses = object_states[:, 0, :, constants.MASS_IDX:constants.MASS_IDX+1]
        else:
            # Shape: [batch, objects, features]
            current_states = object_states
            masses = object_states[:, :, constants.MASS_IDX:constants.MASS_IDX+1]
        
        # Handle sequence dimension in next_states
        if next_states.dim() == 4:
            target_states = next_states
        else:
            target_states = next_states
        
        batch[constants.BATCH_KEY_TARGET_STATES] = target_states
        batch[constants.BATCH_KEY_CURRENT_STATES] = current_states
        batch[constants.BATCH_KEY_MASSES] = masses
        
        # Compute delta_states if not already in batch (for backward compatibility)
        # Delta = next_state - current_state (position-invariant physics target)
        if not hasattr(constants, 'BATCH_KEY_DELTA_STATES') or constants.BATCH_KEY_DELTA_STATES not in batch:
            delta_states = target_states - current_states
            if hasattr(constants, 'BATCH_KEY_DELTA_STATES'):
                batch[constants.BATCH_KEY_DELTA_STATES] = delta_states
    
    return batch


def compute_improved_loss(loss, embeddings, numbers, aux_heads, aux_loss_fn, contrastive_loss_fn, use_auxiliary, use_contrastive, config):
    """
    Compute loss with advanced features (auxiliary + contrastive).
    
    Args:
        loss: Base loss
        embeddings: Number embeddings
        numbers: Numbers
        aux_heads: Auxiliary task heads
        aux_loss_fn: Auxiliary loss function
        contrastive_loss_fn: Contrastive loss function
        use_auxiliary: Whether to use auxiliary tasks
        use_contrastive: Whether to use contrastive learning
        config: Config object
        
    Returns:
        total_loss, loss_dict
    """
    import logging
    
    losses = {'main': loss}
    total_loss = loss
    
    # Add auxiliary task loss
    if use_auxiliary and aux_heads is not None and embeddings is not None and numbers is not None:
        if len(embeddings) >= 2:
            try:
                from training.improvements.auxiliary_tasks import generate_auxiliary_targets
                from training.constants import LOSS_KEY_AUXILIARY
                
                aux_outputs = {
                    'comparison': aux_heads.forward_comparison(embeddings[:-1], embeddings[1:]),
                    'magnitude': aux_heads.forward_magnitude(embeddings),
                    'parity': aux_heads.forward_parity(embeddings),
                    'digit_sum': aux_heads.forward_digit_sum(embeddings)
                }
                
                aux_targets = generate_auxiliary_targets(
                    torch.stack([numbers[:-1], numbers[1:]], dim=1)
                )
                
                aux_loss, _ = aux_loss_fn(aux_outputs, aux_targets)
                losses[LOSS_KEY_AUXILIARY] = aux_loss
                total_loss += 0.2 * aux_loss
            except (RuntimeError, ValueError, KeyError) as e:
                logging.warning(f"Auxiliary task computation failed: {e}")
    
    # Add contrastive loss
    if use_contrastive and contrastive_loss_fn is not None and embeddings is not None and numbers is not None:
        try:
            from training.constants import LOSS_KEY_CONTRASTIVE
            contrast_loss, _ = contrastive_loss_fn(embeddings, numbers)
            losses[LOSS_KEY_CONTRASTIVE] = contrast_loss
            total_loss += 0.1 * contrast_loss
        except (RuntimeError, ValueError, KeyError) as e:
            logging.warning(f"Contrastive learning computation failed: {e}")
    
    return total_loss, losses


def format_validation_results(task: str, accuracy: float, avg_loss: float, total_samples: int, show_details: bool = True) -> str:
    """
    Format validation results for display.
    
    Args:
        task: Task name
        accuracy: Accuracy value
        avg_loss: Average loss
        total_samples: Number of samples
        show_details: Whether to show detailed info
        
    Returns:
        Formatted string
    """
    if show_details:
        return f"  [OK] {task.capitalize()}: {accuracy:.2%} (loss: {avg_loss:.4f}) [{total_samples} samples]"
    else:
        return f"  {task.capitalize()}: {accuracy:.2%}"


def print_final_training_report(use_cls, cls_memory, stats, validation_history):
    """
    Print final training report with CLS statistics.
    
    Args:
        use_cls: Whether CLS is enabled
        cls_memory: CLS memory system
        stats: Training statistics dict
        validation_history: Validation history dict
    """
    print("\n" + "="*70)
    print("TRAINING COMPLETE - CLS REPORT")
    print("="*70)
    
    # CLS statistics
    if use_cls and cls_memory:
        memory_stats = cls_memory.get_stats()
        
        print("\n[STATS] Hippocampal Buffer Statistics:")
        for task, count in memory_stats['hippocampus'].items():
            print(f"  {task.capitalize()}: {count:,} experiences stored")
        
        print(f"\n[CYCLE] Consolidation Statistics:")
        print(f"  Total consolidations: {stats.get('consolidations', 0):,}")
        for task, count in stats.get('consolidation_by_task', {}).items():
            print(f"  {task.capitalize()}: {count:,} replays")
    
    # Forgetting detection
    if stats.get('forgetting_detected'):
        print(f"\n[WARN]  Forgetting Events Detected: {len(stats['forgetting_detected'])}")
        for event in stats['forgetting_detected']:
            print(f"  Epoch {event['epoch']}: {event['task']} (↓ {event['drop']:.2%})")
    else:
        print("\n[OK] No catastrophic forgetting detected!")
    
    # Final validation
    print("\n[UP] Final Validation Performance:")
    for task, history in validation_history.items():
        if history:
            final_acc = history[-1]
            print(f"  {task.capitalize()}: {final_acc:.2%}")
    
    print("\n" + "="*70)
    print("[OK] CLS-based training complete!")
    print("="*70)


def evaluate_embodied_cognition_helper(model, device, test_data, stage, stats, embodied_available):
    """
    Evaluate embodied cognition metrics.
    
    Args:
        model: The model to evaluate
        device: Device to run on
        test_data: Test data dict or None
        stage: Current stage number
        stats: Stats dict to update
        embodied_available: Whether embodied metrics are available
        
    Returns:
        Results dict or None
    """
    from datetime import datetime
    
    if not embodied_available:
        print("[WARN]  Embodied metrics not available")
        return None
    
    print("\n" + "="*70)
    print(f"EVALUATING EMBODIED COGNITION (Stage {stage})")
    print("="*70)
    
    try:
        from training.embodied_metrics import EmbodiedMetricsAnalyzer
        from training.utils.input_preparation import ModelInputPreparator
        from training.configs.config import TrainingConfig
        
        # Initialize input preparator
        config = TrainingConfig()
        preparator = ModelInputPreparator(config)
        
        analyzer = EmbodiedMetricsAnalyzer(model, device=device)
        
        # Use provided test data or create minimal test set
        if test_data is None:
            print("[WARN]  No test data provided, using minimal evaluation")
            # Create standardized test batch
            test_batch = preparator.create_minimal_test_batch(mode='physics', batch_size=3, device=device)
            test_data = {
                'physics_states': test_batch  # Already in correct tensor format
            }
        
        # Run analyses
        results = {}
        
        if 'grounding_tests' in test_data:
            results['grounding'] = analyzer.physics_grounding_coherence(test_data['grounding_tests'])
        
        if 'intuition_tests' in test_data:
            results['intuition'] = analyzer.physical_intuition_emergence(test_data['intuition_tests'])
        
        if 'physics_states' in test_data:
            try:
                results['geometry'] = analyzer.embodied_representation_geometry(test_data['physics_states'])
            except Exception as geom_error:
                print(f"[WARN]  Skipping geometry analysis: {geom_error}")
                results['geometry'] = {'geometric_fidelity': 0.0, 'error': str(geom_error)}
        
        # Compute overall score
        scores = []
        if 'grounding' in results:
            scores.append(results['grounding'].get('mean_coherence', 0))
        if 'intuition' in results:
            scores.append(results['intuition'].get('overall_intuition', 0))
        if 'geometry' in results:
            scores.append(results['geometry'].get('geometric_fidelity', 0))
        
        results['overall_embodiment'] = sum(scores) / len(scores) if scores else 0.0
        results['stage'] = stage
        results['timestamp'] = datetime.now().isoformat()
        
        # Store in stats
        stats['embodied_metrics'].append(results)
        
        print(f"\n[OK] Overall Embodiment Score: {results['overall_embodiment']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"[WARN]  Error evaluating embodied cognition: {e}")
        return None


def validate_all_tasks_helper(
    model, device, config, validation_history, stats, cls_memory, use_cls,
    temporal_metrics, epoch, validation_dataloaders, trained_tasks, task_names,
    task_physics, task_counting, task_arithmetic, task_symbolic,
    batch_key_next_states, output_key_predicted_states, batch_key_counts,
    output_key_predicted_answer, batch_key_results, batch_key_answers,
    batch_key_object_states, batch_key_object_mask
):
    """
    Multi-task validation to detect catastrophic forgetting.
    
    This is a large helper function extracted from CLSTrainingPipeline.validate_all_tasks.
    """
    from torch.cuda.amp import autocast
    from training.common.training_utils import move_batch_to_device
    
    print(f"\n[SEARCH] Multi-Task Validation (Epoch {epoch+1}):")
    print(f"{'='*70}")
    
    model.eval()
    
    # Import loss function with config weights
    from models.physics_former_full import FullPhysicsLoss
    loss_fn = FullPhysicsLoss(
        prediction_weight=config.prediction_weight,
        energy_weight=config.energy_weight,
        momentum_weight=config.momentum_weight,
        counting_weight=config.counting_weight,
        arithmetic_weight=config.arithmetic_weight,
        symbolic_weight=config.symbolic_weight,
        threshold_weight=getattr(config, 'threshold_weight', 0.3),
        accuracy_threshold=getattr(config, 'accuracy_threshold', 0.1),
        threshold_margin=getattr(config, 'threshold_margin', 0.05)
    )
    
    # Determine which tasks to validate
    if trained_tasks is None:
        tasks_to_validate = task_names
    else:
        tasks_to_validate = trained_tasks
        print(f"[INFO] Validating trained tasks only: {', '.join(tasks_to_validate)}")
    
    with torch.no_grad():
        for task in tasks_to_validate:
            # Skip if no validation data
            if validation_dataloaders is None or task not in validation_dataloaders or validation_dataloaders[task] is None:
                if len(validation_history[task]) > 0:
                    accuracy = validation_history[task][-1]
                    print(f"  [WARN]  {task.capitalize()}: No validation data (using previous: {accuracy:.2%})")
                else:
                    accuracy = 0.0
                    print(f"  [WARN]  {task.capitalize()}: No validation data available (using default: 0%)")
                validation_history[task].append(accuracy)
                continue
            
            # Perform validation
            val_loader = validation_dataloaders[task]
            total_correct = 0
            total_samples = 0
            val_losses = []
            
            # Physics-specific metrics
            all_mse_values = [] if task == task_physics else None
            mse_thresholds = [0.01, 0.05, 0.1, 0.5, 1.0] if task == task_physics else None
            
            # Physical-unit error tracking (in real units: meters, m/s, radians)
            physical_errors = {
                'position': [],      # meters
                'velocity': [],      # m/s
                'orientation': [],   # quaternion error (unitless, 0-2 range)
                'angular_vel': []    # rad/s
            } if task == task_physics else None
            
            # Store example for visualization
            example_pred = None
            example_target = None
            
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                
                # Prepare batch
                if task in [task_physics, task_counting]:
                    import sys
                    batch = prepare_physics_batch(batch, sys.modules['training.constants'])
                
                # Forward pass
                with autocast(enabled=config.mixed_precision and torch.cuda.is_available()):
                    outputs = model(**batch, mode=task)
                    loss, metrics = loss_fn(outputs, batch, mode=task)
                val_losses.append(loss.item())
                
                # Compute accuracy
                if task == task_physics:
                    if batch_key_next_states in batch and output_key_predicted_states in outputs:
                        pred = outputs[output_key_predicted_states]  # Model outputs DELTAS
                        
                        # HYBRID VALIDATION: Match training approach
                        # - Position (0:3): Compare ABSOLUTE positions (current + delta)
                        # - Velocity/Quaternion/AngularVel (3:13): Compare DELTAS
                        
                        # Get current states and target deltas
                        current_states = batch.get('current_states')
                        target_states = batch.get(batch_key_next_states)
                        
                        if 'delta_states' in batch:
                            target_deltas = batch['delta_states']
                        elif current_states is not None and target_states is not None:
                            target_deltas = target_states - current_states
                        else:
                            object_states = batch.get(batch_key_object_states)
                            if object_states is not None and object_states.dim() == 4:
                                target_deltas = object_states[:, 1:] - object_states[:, :-1]
                                current_states = object_states[:, :-1]
                                target_states = object_states[:, 1:]
                            else:
                                target_deltas = target_states
                        
                        # For position: reconstruct absolute positions
                        # pred is delta, so pred_absolute_pos = current_pos + pred_delta_pos
                        if current_states is not None and target_states is not None:
                            pred_absolute_pos = current_states[..., :3] + pred[..., :3]
                            target_absolute_pos = target_states[..., :3]
                            use_absolute_pos = True
                        else:
                            # Fallback to delta comparison
                            pred_absolute_pos = pred[..., :3]
                            target_absolute_pos = target_deltas[..., :3]
                            use_absolute_pos = False
                        
                        from models.physics_former_full import FullPhysicsLoss
                        
                        DYNAMIC_END = 13
                        DELTA_START = 3
                        
                        # Get the running delta scales (for velocity/quaternion/angular velocity)
                        if hasattr(FullPhysicsLoss, '_running_delta_scales'):
                            delta_scales = FullPhysicsLoss._running_delta_scales
                        else:
                            DELTA_FEATURES = 10
                            batch_scales = torch.std(target_deltas[..., DELTA_START:DYNAMIC_END], dim=(0, 1, 2), keepdim=True)
                            min_scales = torch.ones(DELTA_FEATURES, device=batch_scales.device, dtype=batch_scales.dtype)
                            min_scales[0:3] = 1.0
                            min_scales[3:7] = 0.1
                            min_scales[7:10] = 50.0
                            min_scales = min_scales.view(1, 1, 1, DELTA_FEATURES)
                            delta_scales = torch.maximum(batch_scales, min_scales)
                        
                        # Get position scales (for absolute position)
                        if hasattr(FullPhysicsLoss, '_running_pos_scales'):
                            pos_scales = FullPhysicsLoss._running_pos_scales
                        else:
                            batch_scales_pos = torch.std(target_absolute_pos, dim=(0, 1, 2), keepdim=True)
                            min_scales_pos = torch.ones(3, device=batch_scales_pos.device, dtype=batch_scales_pos.dtype) * 1.0
                            min_scales_pos = min_scales_pos.view(1, 1, 1, 3)
                            pos_scales = torch.maximum(batch_scales_pos, min_scales_pos)
                        
                        # Normalize: absolute position + delta velocity/quaternion/angular velocity
                        normalized_pred_pos = pred_absolute_pos / pos_scales
                        normalized_target_pos = target_absolute_pos / pos_scales
                        normalized_pred_delta = pred[..., DELTA_START:DYNAMIC_END] / delta_scales
                        normalized_target_delta = target_deltas[..., DELTA_START:DYNAMIC_END] / delta_scales
                        
                        # Concatenate for MSE computation
                        normalized_pred = torch.cat([normalized_pred_pos, normalized_pred_delta], dim=-1)
                        normalized_target = torch.cat([normalized_target_pos, normalized_target_delta], dim=-1)
                        
                        # Keep target_deltas as 'target' for physical error computation later
                        target = target_deltas
                        
                        # Compute normalized MSE per sample
                        if normalized_pred.dim() == 4:
                            mse_per_sample = ((normalized_pred - normalized_target) ** 2).mean(dim=[1, 2, 3])
                        else:
                            mse_per_sample = ((normalized_pred - normalized_target) ** 2).mean(dim=[1, 2])
                        
                        # Collect MSE values for distribution analysis
                        all_mse_values.extend(mse_per_sample.cpu().tolist())
                        
                        # Count correct with original threshold
                        correct = (mse_per_sample < 0.1).sum().item()
                        total_correct += correct
                        total_samples += pred.shape[0]
                        
                        # Store first example for visualization
                        # For position: store ABSOLUTE positions (current + delta)
                        # For velocity/quaternion/angular velocity: store deltas
                        if example_pred is None:
                            example_pred_raw = pred[0].detach().cpu()  # [seq, objects, features] - deltas
                            example_target_raw = target[0].detach().cpu()  # deltas
                            
                            # Reconstruct absolute positions for display
                            if current_states is not None and target_states is not None:
                                current_pos = current_states[0, ..., :3].detach().cpu()  # [seq, objects, 3]
                                target_abs_pos = target_states[0, ..., :3].detach().cpu()
                                pred_abs_pos = current_pos + example_pred_raw[..., :3]
                                
                                # Create hybrid display tensors: absolute pos + delta velocity/quat/angvel
                                example_pred = example_pred_raw.clone()
                                example_pred[..., :3] = pred_abs_pos
                                example_target = example_target_raw.clone()
                                example_target[..., :3] = target_abs_pos
                                example_is_absolute_pos = True
                            else:
                                example_pred = example_pred_raw
                                example_target = example_target_raw
                                example_is_absolute_pos = False
                        
                        # Compute physical-unit errors (RAW, not normalized)
                        # State vector: [pos(0-2), vel(3-5), quat(6-9), ang_vel(10-12), ...]
                        # pred and target are deltas: [batch, seq, objects, 35]
                        if physical_errors is not None:
                            # Position error (meters) - indices 0:3
                            pos_error = torch.sqrt(((pred[..., :3] - target[..., :3]) ** 2).sum(dim=-1))
                            physical_errors['position'].extend(pos_error.mean(dim=[1, 2]).cpu().tolist())
                            
                            # Velocity error (m/s) - indices 3:6
                            vel_error = torch.sqrt(((pred[..., 3:6] - target[..., 3:6]) ** 2).sum(dim=-1))
                            physical_errors['velocity'].extend(vel_error.mean(dim=[1, 2]).cpu().tolist())
                            
                            # Orientation error (quaternion distance, 0-2 range) - indices 6:10
                            # Quaternion distance: 1 - |q1 · q2| (0 = identical, 1 = 90°, 2 = 180°)
                            pred_quat = pred[..., 6:10]
                            target_quat = target[..., 6:10]
                            # For deltas, we compute magnitude of quaternion change
                            quat_error = torch.sqrt((pred_quat ** 2).sum(dim=-1)) - torch.sqrt((target_quat ** 2).sum(dim=-1))
                            quat_error = torch.abs(quat_error)
                            physical_errors['orientation'].extend(quat_error.mean(dim=[1, 2]).cpu().tolist())
                            
                            # Angular velocity error (rad/s) - indices 10:13
                            ang_vel_error = torch.sqrt(((pred[..., 10:13] - target[..., 10:13]) ** 2).sum(dim=-1))
                            physical_errors['angular_vel'].extend(ang_vel_error.mean(dim=[1, 2]).cpu().tolist())
                
                elif task in [task_counting, task_arithmetic, task_symbolic]:
                    if task == task_arithmetic:
                        if isinstance(outputs, tuple) and len(outputs) == 2:
                            digit_logits, length_logits = outputs
                            pred = _decode_compositional_output(digit_logits, length_logits, device)
                            target = batch[batch_key_results]
                            correct = (pred == target).sum().item()
                            total_correct += correct
                            total_samples += pred.shape[0]
                    elif task == task_symbolic:
                        if isinstance(outputs, tuple) and len(outputs) == 2:
                            digit_logits, length_logits = outputs
                            pred = _decode_compositional_output(digit_logits, length_logits, device)
                            target = batch[batch_key_answers]
                            correct = (pred == target).sum().item()
                            total_correct += correct
                            total_samples += pred.shape[0]
                    elif task == task_counting:
                        if batch_key_counts in batch and output_key_predicted_answer in outputs:
                            pred = outputs[output_key_predicted_answer].argmax(dim=-1)
                            target = batch[batch_key_counts]
                            correct = (pred == target).sum().item()
                            total_correct += correct
                            total_samples += pred.shape[0]
                    
                    # Log temporal metrics
                    if temporal_metrics is not None:
                        if task == task_counting and batch_key_object_states in batch:
                            if 'pred' in locals() and 'target' in locals():
                                temporal_metrics.log_counting_result(
                                    object_states=batch[batch_key_object_states],
                                    predicted=pred,
                                    target=target,
                                    object_mask=batch.get(batch_key_object_mask)
                                )
                        elif task == task_arithmetic:
                            if 'group_a_states' in batch and 'group_b_states' in batch:
                                if 'pred' in locals() and 'target' in locals():
                                    temporal_metrics.log_arithmetic_result(
                                        group_a_states=batch['group_a_states'],
                                        group_b_states=batch['group_b_states'],
                                        predicted=pred,
                                        target=target,
                                        operation=batch.get('operation', 'unknown')
                                    )
                
                if total_samples >= 1000:
                    break
            
            # Compute accuracy
            if total_samples > 0:
                accuracy = total_correct / total_samples
                avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
            else:
                accuracy = 0.0
                avg_loss = 0.0
                print(f"  [WARN]  {task.capitalize()}: No samples processed during validation!")
            
            # Physics-specific MSE analysis
            if task == task_physics and all_mse_values:
                import numpy as np
                mse_array = np.array(all_mse_values)
                mean_mse = np.mean(mse_array)
                median_mse = np.median(mse_array)
                p25_mse = np.percentile(mse_array, 25)
                p75_mse = np.percentile(mse_array, 75)
                p90_mse = np.percentile(mse_array, 90)
                
                # Compute accuracy at different thresholds
                threshold_accuracies = {}
                for threshold in mse_thresholds:
                    threshold_accuracies[threshold] = (mse_array < threshold).mean()
                
                # For physics, use threshold-based accuracy (MSE < 0.1) for curriculum progression
                # This replaces the meaningless 0.0 accuracy from total_correct/total_samples
                accuracy = threshold_accuracies.get(0.1, 0.0)
            
            validation_history[task].append(accuracy)
            
            # Detect forgetting
            forgetting_threshold = getattr(config, 'forgetting_threshold', 0.1)
            if len(validation_history[task]) > 1:
                prev_acc = validation_history[task][-2]
                if accuracy < prev_acc - forgetting_threshold:
                    print(f"  [WARN]  {task.capitalize()}: {accuracy:.2%} (loss: {avg_loss:.4f}) "
                          f"(↓ {(prev_acc - accuracy):.2%} - FORGETTING DETECTED!)")
                    stats['forgetting_detected'].append({
                        'epoch': epoch,
                        'task': task,
                        'drop': prev_acc - accuracy
                    })
                    if use_cls:
                        cls_memory.update_performance(task, accuracy)
                else:
                    print(f"  [OK] {task.capitalize()}: {accuracy:.2%} (loss: {avg_loss:.4f}) [{total_samples} samples]")
            else:
                print(f"  [OK] {task.capitalize()}: {accuracy:.2%} (loss: {avg_loss:.4f}) [{total_samples} samples]")
            
            # Print detailed MSE metrics for physics
            if task == task_physics and all_mse_values:
                print(f"       MSE Distribution: mean={mean_mse:.4f}, median={median_mse:.4f}, p90={p90_mse:.4f}")
                print(f"       Accuracy @ Thresholds: ", end="")
                threshold_strs = [f"<{t}: {threshold_accuracies[t]:.1%}" for t in mse_thresholds]
                print(" | ".join(threshold_strs))
                
                # Print physical-unit errors (more interpretable than normalized MSE)
                if physical_errors and len(physical_errors['position']) > 0:
                    import numpy as np
                    pos_err = np.array(physical_errors['position'])
                    vel_err = np.array(physical_errors['velocity'])
                    ori_err = np.array(physical_errors['orientation'])
                    ang_err = np.array(physical_errors['angular_vel'])
                    
                    print(f"       Physical Errors (per timestep):")
                    print(f"         Position:  mean={np.mean(pos_err)*100:.1f}cm, median={np.median(pos_err)*100:.1f}cm, p90={np.percentile(pos_err, 90)*100:.1f}cm")
                    print(f"         Velocity:  mean={np.mean(vel_err):.2f}m/s, median={np.median(vel_err):.2f}m/s, p90={np.percentile(vel_err, 90):.2f}m/s")
                    print(f"         Rotation:  mean={np.mean(ori_err):.3f}, median={np.median(ori_err):.3f}, p90={np.percentile(ori_err, 90):.3f}")
                    print(f"         AngVel:    mean={np.mean(ang_err):.1f}rad/s, median={np.median(ang_err):.1f}rad/s, p90={np.percentile(ang_err, 90):.1f}rad/s")
                
                # Print example prediction vs target (first sample, first object, multiple timesteps)
                if 'example_pred' in dir() and example_pred is not None:
                    # Check if we have absolute position display
                    is_abs_pos = 'example_is_absolute_pos' in dir() and example_is_absolute_pos
                    
                    # All 13 dynamic features grouped by type
                    # Position shows ABSOLUTE values, others show DELTAS
                    pos_label = 'Position-ABS (m)' if is_abs_pos else 'Position-Δ (m)'
                    feature_groups = [
                        (pos_label, ['pos_x', 'pos_y', 'pos_z'], [0, 1, 2]),
                        ('Velocity-Δ (m/s)', ['vel_x', 'vel_y', 'vel_z'], [3, 4, 5]),
                        ('Quaternion-Δ', ['quat_x', 'quat_y', 'quat_z', 'quat_w'], [6, 7, 8, 9]),
                        ('AngVel-Δ (rad/s)', ['ang_x', 'ang_y', 'ang_z'], [10, 11, 12])
                    ]
                    
                    num_timesteps = min(5, example_pred.shape[0])
                    mode_str = "HYBRID: pos=absolute, vel/quat/angvel=delta" if is_abs_pos else "ALL DELTAS"
                    print(f"\n       Example Prediction (Object 0, {num_timesteps} timesteps) [{mode_str}]:")
                    print(f"       {'Feature':<12} {'Target':>12} {'Predicted':>12} {'Error':>10} {'Ratio':>8}")
                    print(f"       {'-'*56}")
                    
                    for group_name, feat_names, feat_indices in feature_groups:
                        print(f"       --- {group_name} ---")
                        for t in range(num_timesteps):
                            print(f"       t={t}:", end="")
                            for f_name, f_idx in zip(feat_names, feat_indices):
                                if f_idx >= example_pred.shape[-1]:
                                    continue
                                tgt = example_target[t, 0, f_idx].item()
                                prd = example_pred[t, 0, f_idx].item()
                                err = abs(tgt - prd)
                                # Compute ratio (how much model under/over-predicts)
                                if abs(tgt) > 1e-6:
                                    ratio = prd / tgt
                                    ratio_str = f"{ratio:.2f}x"
                                else:
                                    ratio_str = "N/A"
                                print(f" {f_name}={tgt:>7.3f}→{prd:>7.3f}({ratio_str})", end="")
                            print()  # newline after each timestep
                    
                    # Summary: average ratio per feature type across all timesteps
                    print(f"\n       Feature Summary (avg across {num_timesteps} timesteps):")
                    for group_name, feat_names, feat_indices in feature_groups:
                        ratios = []
                        errors = []
                        for t in range(num_timesteps):
                            for f_idx in feat_indices:
                                if f_idx >= example_pred.shape[-1]:
                                    continue
                                tgt = example_target[t, 0, f_idx].item()
                                prd = example_pred[t, 0, f_idx].item()
                                errors.append(abs(tgt - prd))
                                if abs(tgt) > 1e-6:
                                    ratios.append(prd / tgt)
                        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
                        avg_err = sum(errors) / len(errors) if errors else 0
                        status = "✓" if 0.8 <= avg_ratio <= 1.2 else "⚠" if 0.5 <= avg_ratio <= 2.0 else "✗"
                        print(f"         {status} {group_name:<20}: avg_ratio={avg_ratio:.2f}x, avg_err={avg_err:.4f}")
            
            # Print temporal metrics
            if task in [task_counting, task_arithmetic] and temporal_metrics is not None:
                temporal_metrics.print_report(task)
                impact = temporal_metrics.detect_rnn_impact(task, threshold=5.0)
                if impact.get('motion_dependent'):
                    print(f"\n[WARN] WARNING: {task.capitalize()} shows motion-dependency!")
                    print(f"   Difference: {impact['motion_difference']:.2f}%")
                    print("   RNN temporal context may be affecting performance.\n")
                
                import os
                metrics_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'metrics')
                os.makedirs(metrics_dir, exist_ok=True)
                metrics_file = os.path.join(metrics_dir, f'{task}_temporal_metrics_epoch{epoch}.json')
                temporal_metrics.save_metrics(metrics_file)
                temporal_metrics.reset()
    
    model.train()
    print(f"{'='*70}")
    
    # Return physics accuracy for curriculum progression
    if task_physics in validation_history and len(validation_history[task_physics]) > 0:
        return {'physics_accuracy': validation_history[task_physics][-1]}
    return {'physics_accuracy': 0.0}


def _decode_compositional_output(digit_logits, length_logits, device):
    """Helper to decode compositional output to numbers."""
    digits = torch.argmax(digit_logits, dim=-1)
    length = torch.argmax(length_logits, dim=-1) + 1
    
    batch_size = digits.shape[0]
    pred_numbers = []
    PAD_TOKEN = 10
    SIGN_TOKEN = 11
    
    for i in range(batch_size):
        num_digits = length[i].item()
        digit_list = digits[i, :num_digits].tolist()
        
        is_negative = False
        if len(digit_list) > 0 and digit_list[0] == SIGN_TOKEN:
            is_negative = True
            digit_list = digit_list[1:]
        
        digit_list = [d if d < 10 else 0 for d in digit_list]
        
        if len(digit_list) == 0:
            number = 0
        else:
            number = sum(d * (10 ** (len(digit_list) - 1 - idx)) 
                        for idx, d in enumerate(digit_list))
        
        if is_negative:
            number = -number
        
        pred_numbers.append(number)
    
    return torch.tensor(pred_numbers, device=device)
