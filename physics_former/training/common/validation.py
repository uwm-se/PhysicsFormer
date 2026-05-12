"""
Validation utilities for multi-task learning.

Provides forgetting detection and cross-task validation.

NOTE: The actual validation is performed by validate_all_tasks_helper() 
in training/pipelines/training_helpers.py. This file contains only utility functions.
"""

import torch
from collections import defaultdict


def detect_forgetting(validation_history, task, current_accuracy, threshold=0.1):
    """
    Detect if catastrophic forgetting occurred for a task.
    
    Args:
        validation_history: Dict of validation history per task
        task: Task name
        current_accuracy: Current accuracy for the task
        threshold: Forgetting threshold (default: 10% drop)
    
    Returns:
        dict: {
            'forgetting_detected': bool,
            'drop': float,
            'previous_accuracy': float,
            'current_accuracy': float
        }
    """
    result = {
        'forgetting_detected': False,
        'drop': 0.0,
        'previous_accuracy': None,
        'current_accuracy': current_accuracy
    }
    
    if task not in validation_history or len(validation_history[task]) < 2:
        return result
    
    # Get previous accuracy
    prev_acc = validation_history[task][-2]
    result['previous_accuracy'] = prev_acc
    
    # Calculate drop
    drop = prev_acc - current_accuracy
    result['drop'] = drop
    
    # Detect forgetting
    if drop > threshold:
        result['forgetting_detected'] = True
    
    return result


def get_best_accuracy(validation_history, task):
    """
    Get best accuracy achieved for a task.
    
    Args:
        validation_history: Dict of validation history per task
        task: Task name
    
    Returns:
        float: Best accuracy or None if no history
    """
    if task not in validation_history or not validation_history[task]:
        return None
    
    return max(validation_history[task])


def get_forgetting_events(validation_history, threshold=0.1):
    """
    Get all forgetting events across all tasks.
    
    Args:
        validation_history: Dict of validation history per task
        threshold: Forgetting threshold
    
    Returns:
        list: List of forgetting events with details
    """
    events = []
    
    for task, history in validation_history.items():
        if len(history) < 2:
            continue
        
        for i in range(1, len(history)):
            prev_acc = history[i-1]
            curr_acc = history[i]
            drop = prev_acc - curr_acc
            
            if drop > threshold:
                events.append({
                    'task': task,
                    'epoch': i,
                    'previous_accuracy': prev_acc,
                    'current_accuracy': curr_acc,
                    'drop': drop
                })
    
    return events


def print_validation_summary(validation_history):
    """
    Print summary of validation performance across all tasks.
    
    Args:
        validation_history: Dict of validation history per task
    """
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for task, history in validation_history.items():
        if not history:
            continue
        
        final_acc = history[-1]
        best_acc = max(history)
        worst_acc = min(history)
        
        print(f"\n{task.capitalize()}:")
        print(f"  Final: {final_acc:.2%}")
        print(f"  Best: {best_acc:.2%}")
        print(f"  Worst: {worst_acc:.2%}")
        
        # Check if performance degraded
        if len(history) > 1:
            initial_acc = history[0]
            if final_acc < initial_acc - 0.05:  # 5% drop
                print(f"  WARNING:  Performance degraded by {(initial_acc - final_acc):.2%}")
    
    print("="*70)


def create_validation_loaders(dataloaders, validation_split=0.1):
    """
    Create validation loaders from training dataloaders by splitting datasets.
    
    Args:
        dataloaders: Dict of training dataloaders
        validation_split: Fraction of data to use for validation (default: 10%)
    
    Returns:
        tuple: (train_loaders, val_loaders) - Updated training loaders and new validation loaders
    """
    from torch.utils.data import DataLoader, random_split

    train_loaders = {}
    val_loaders = {}
    
    for task_name, dataloader in dataloaders.items():
        if dataloader is None:
            train_loaders[task_name] = None
            val_loaders[task_name] = None
            continue
        
        dataset = dataloader.dataset
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        # Split dataset
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Get collate function if needed. Counting and arithmetic datasets
        # are not part of the physics-only Stage-1 snapshot, so import
        # their collate_fn lazily and raise a clear error if somehow
        # requested.
        collate_fn = None
        if task_name == 'counting':
            raise RuntimeError(
                "counting dataset is not included in this snapshot "
                "(physics_former_best.pt was trained physics-only). "
                "Restore training/datasets/counting_dataset.py from the full "
                "physics_former repository to enable counting validation."
            )
        elif task_name == 'arithmetic':
            raise RuntimeError(
                "arithmetic dataset is not included in this snapshot "
                "(physics_former_best.pt was trained physics-only). "
                "Restore training/datasets/arithmetic_dataset.py from the full "
                "physics_former repository to enable arithmetic validation."
            )
        
        # Create new dataloaders
        train_loaders[task_name] = DataLoader(
            train_dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            pin_memory=getattr(dataloader, 'pin_memory', False),
            collate_fn=collate_fn
        )
        
        val_loaders[task_name] = DataLoader(
            val_dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            pin_memory=getattr(dataloader, 'pin_memory', False),
            collate_fn=collate_fn
        )
    
    return train_loaders, val_loaders
