"""
Checkpoint management utilities.

Provides consistent checkpoint saving/loading across pipelines.
"""

import torch
from pathlib import Path
from datetime import datetime


def save_checkpoint(
    model,
    optimizer,
    stage,
    epoch,
    path,
    scheduler=None,
    validation_history=None,
    stats=None,
    **metadata
):
    """
    Save checkpoint with metadata.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        stage: Current stage number
        epoch: Current epoch
        path: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        validation_history: Validation history dict (optional)
        stats: Training statistics dict (optional)
        **metadata: Additional metadata to save
    
    Returns:
        Path: Path where checkpoint was saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'stage': stage,
        'epoch': epoch,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add optional components
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if validation_history is not None:
        checkpoint['validation_history'] = validation_history
    
    if stats is not None:
        checkpoint['stats'] = stats
    
    # Add any additional metadata
    checkpoint.update(metadata)
    
    # Save
    torch.save(checkpoint, path)
    
    print(f"PASS: Checkpoint saved: {path}")
    
    return path


def load_checkpoint(
    path,
    model=None,
    optimizer=None,
    scheduler=None,
    device='cuda',
    strict=True
):
    """
    Load checkpoint with validation.
    
    Args:
        path: Path to checkpoint
        model: PyTorch model (optional, will load state if provided)
        optimizer: Optimizer (optional, will load state if provided)
        scheduler: Scheduler (optional, will load state if provided)
        device: Device to load checkpoint to
        strict: Whether to strictly enforce state dict keys match
    
    Returns:
        dict: Checkpoint dictionary with all saved data
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    # Load checkpoint
    # PyTorch 2.6+ defaults to weights_only=True, but our checkpoints contain
    # numpy arrays and other metadata that require weights_only=False
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Load model state
    if model is not None and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Handle torch.compile() prefix mismatch
        # Checkpoints saved with compiled models have "_orig_mod." prefix
        sample_key = next(iter(state_dict.keys()), "")
        has_orig_mod_prefix = sample_key.startswith("_orig_mod.")
        
        model_sample_key = next(iter(model.state_dict().keys()), "")
        model_has_prefix = model_sample_key.startswith("_orig_mod.")
        
        if has_orig_mod_prefix and not model_has_prefix:
            # Strip _orig_mod. prefix from checkpoint keys
            print("[INFO] Stripping '_orig_mod.' prefix from checkpoint keys (compiled -> uncompiled)")
            state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        elif not has_orig_mod_prefix and model_has_prefix:
            # Add _orig_mod. prefix to checkpoint keys
            print("[INFO] Adding '_orig_mod.' prefix to checkpoint keys (uncompiled -> compiled)")
            state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}
        
        # If strict=False, filter out keys with size mismatches (for architecture changes)
        if not strict:
            model_state = model.state_dict()
            filtered_state = {}
            skipped_keys = []
            
            for key, value in state_dict.items():
                if key in model_state:
                    if value.shape == model_state[key].shape:
                        filtered_state[key] = value
                    else:
                        skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state[key].shape})")
                else:
                    skipped_keys.append(f"{key} (not in model)")
            
            if skipped_keys:
                print(f"[INFO] Skipped {len(skipped_keys)} incompatible keys:")
                for key in skipped_keys[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(skipped_keys) > 5:
                    print(f"  ... and {len(skipped_keys) - 5} more")
            
            model.load_state_dict(filtered_state, strict=False)
        else:
            model.load_state_dict(state_dict, strict=strict)
        
        print(f"PASS: Model state loaded from: {path}")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"PASS: Optimizer state loaded")
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"PASS: Scheduler state loaded")
    
    # Print checkpoint info
    if 'stage' in checkpoint and 'epoch' in checkpoint:
        print(f"  Stage: {checkpoint['stage']}, Epoch: {checkpoint['epoch']}")
    
    if 'timestamp' in checkpoint:
        print(f"  Saved: {checkpoint['timestamp']}")
    
    return checkpoint


def get_latest_checkpoint(checkpoint_dir, pattern="*.pt"):
    """
    Get the most recent checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern for checkpoint files
    
    Returns:
        Path: Path to latest checkpoint or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    return latest


def list_checkpoints(checkpoint_dir, pattern="*.pt"):
    """
    List all checkpoints in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern for checkpoint files
    
    Returns:
        list: List of checkpoint paths sorted by modification time
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return checkpoints


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5, pattern="*.pt"):
    """
    Remove old checkpoints, keeping only the most recent N.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        pattern: Glob pattern for checkpoint files
    
    Returns:
        int: Number of checkpoints deleted
    """
    checkpoints = list_checkpoints(checkpoint_dir, pattern)
    
    if len(checkpoints) <= keep_last_n:
        return 0
    
    # Delete old checkpoints
    to_delete = checkpoints[keep_last_n:]
    deleted_count = 0
    
    for checkpoint_path in to_delete:
        try:
            checkpoint_path.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"WARNING:  Could not delete {checkpoint_path}: {e}")
    
    if deleted_count > 0:
        print(f"🗑️  Cleaned up {deleted_count} old checkpoints")
    
    return deleted_count


def save_stage_checkpoint(
    model,
    optimizer,
    stage,
    epoch,
    checkpoint_dir,
    **metadata
):
    """
    Save stage-specific checkpoint with standard naming.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        stage: Stage number (1-4)
        epoch: Epoch number
        checkpoint_dir: Directory to save checkpoint
        **metadata: Additional metadata
    
    Returns:
        Path: Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"stage{stage}_epoch{epoch}.pt"
    
    return save_checkpoint(
        model=model,
        optimizer=optimizer,
        stage=stage,
        epoch=epoch,
        path=checkpoint_path,
        **metadata
    )


def save_final_checkpoint(
    model,
    optimizer,
    stage,
    checkpoint_dir,
    **metadata
):
    """
    Save final checkpoint for a stage.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        stage: Stage number (1-4)
        checkpoint_dir: Directory to save checkpoint
        **metadata: Additional metadata
    
    Returns:
        Path: Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"stage{stage}_final.pt"
    
    return save_checkpoint(
        model=model,
        optimizer=optimizer,
        stage=stage,
        epoch=metadata.get('epoch', 0),
        path=checkpoint_path,
        **metadata
    )
