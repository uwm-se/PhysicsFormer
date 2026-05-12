"""
Common utilities shared across training pipelines.

This module provides reusable components for:
- Metrics logging integration
- Validation
- Checkpoint management
- Training utilities

Note: Data loading and curriculum utilities have been archived.
Use run_physics_training.py for physics-only training.
"""

from .metrics_integration import (
    setup_metrics_logger,
    log_training_metrics,
    log_dataset_metrics,
    log_model_metrics,
    finalize_metrics
)
from .validation import detect_forgetting
from .checkpointing import save_checkpoint, load_checkpoint
from .training_utils import (
    compute_gradient_norm,
    freeze_encoder,
    unfreeze_all,
    print_training_progress
)
from training.progressive_curriculum import ProgressiveCurriculum

def print_curriculum_state(curriculum):
    """Print current curriculum state."""
    print(f"  Schema level: {curriculum.current_schema_level}/13")
    print(f"  Sequence length: {curriculum.current_seq_length}")
    print(f"  Phase: {curriculum.current_phase}")

def calculate_batch_size_for_seq_length(seq_length):
    """Calculate recommended batch size based on sequence length."""
    if seq_length <= 32:
        return 16
    elif seq_length <= 64:
        return 8
    elif seq_length <= 128:
        return 4
    elif seq_length <= 256:
        return 2
    else:
        return 1

__all__ = [
    'setup_metrics_logger',
    'log_training_metrics',
    'log_dataset_metrics',
    'log_model_metrics',
    'finalize_metrics',
    'detect_forgetting',
    'save_checkpoint',
    'load_checkpoint',
    'compute_gradient_norm',
    'freeze_encoder',
    'unfreeze_all',
    'print_training_progress',
    'print_curriculum_state',
    'calculate_batch_size_for_seq_length',
    'ProgressiveCurriculum',
]
