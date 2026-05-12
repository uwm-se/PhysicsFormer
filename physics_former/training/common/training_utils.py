"""
Training utilities shared across pipelines.

Common functions for gradient computation, model manipulation, and progress printing.
"""

import torch


def compute_gradient_norm(model):
    """
    Compute total gradient norm across all model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        float: Total gradient norm
    """
    total_norm = 0.0
    nan_params = []
    inf_params = []
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            # Check for NaN/Inf in gradients
            if torch.isnan(p.grad).any():
                nan_params.append(name)
            if torch.isinf(p.grad).any():
                inf_params.append(name)
            
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    # Report NaN/Inf parameters
    if nan_params:
        print(f"\n[ERROR] NaN gradients in {len(nan_params)} parameters:")
        for name in nan_params[:5]:  # Show first 5
            print(f"  - {name}")
        if len(nan_params) > 5:
            print(f"  ... and {len(nan_params) - 5} more")
    
    if inf_params:
        print(f"\n[ERROR] Inf gradients in {len(inf_params)} parameters:")
        for name in inf_params[:5]:  # Show first 5
            print(f"  - {name}")
        if len(inf_params) > 5:
            print(f"  ... and {len(inf_params) - 5} more")
    
    total_norm = total_norm ** 0.5
    return total_norm


def freeze_encoder(model):
    """
    Freeze encoder for schema protection.
    
    Prevents overwriting physics grounding in later stages.
    
    Args:
        model: PyTorch model with encoder and transformer_layers
    """
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    if hasattr(model, 'transformer_layers'):
        for param in model.transformer_layers.parameters():
            param.requires_grad = False
    
    print("🔒 Encoder frozen (schema protection active)")


def unfreeze_all(model):
    """
    Unfreeze all model parameters.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True
    
    print("🔓 All parameters trainable")


def freeze_parameters(model, parameter_names):
    """
    Freeze specific parameters by name.
    
    Args:
        model: PyTorch model
        parameter_names: List of parameter names to freeze
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(pname in name for pname in parameter_names):
            param.requires_grad = False
            frozen_count += 1
    
    print(f"🔒 Frozen {frozen_count} parameters")


def count_parameters(model, trainable_only=False):
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
    
    Returns:
        int: Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_training_progress(
    batch_idx,
    total_batches,
    loss,
    learning_rate=None,
    grad_norm=None,
    consolidations=None,
    gpu_memory_gb=None,
    **kwargs
):
    """
    Print standardized training progress.
    
    Args:
        batch_idx: Current batch index
        total_batches: Total number of batches
        loss: Current loss
        learning_rate: Current learning rate
        grad_norm: Gradient norm
        consolidations: Number of consolidations (CLS only)
        gpu_memory_gb: GPU memory usage in GB
        **kwargs: Additional metrics to print
    """
    progress_pct = (batch_idx / total_batches) * 100 if total_batches > 0 else 0
    
    # Build progress string
    progress_str = f"  [{progress_pct:5.1f}%] Batch {batch_idx:4d}/{total_batches} | Loss: {loss:.4f}"
    
    if learning_rate is not None:
        progress_str += f" | LR: {learning_rate:.2e}"
    
    if grad_norm is not None:
        progress_str += f" | Grad: {grad_norm:.2f}"
    
    if consolidations is not None:
        progress_str += f" | Consolidations: {consolidations}"
    
    if gpu_memory_gb is not None:
        progress_str += f" | GPU: {gpu_memory_gb:.1f}GB"
    
    # Add any additional metrics
    for key, value in kwargs.items():
        if isinstance(value, float):
            progress_str += f" | {key}: {value:.4f}"
        else:
            progress_str += f" | {key}: {value}"
    
    # Use carriage return for in-place update
    print(f"\r{progress_str}", end='', flush=True)


def print_epoch_summary(
    epoch,
    total_epochs,
    avg_loss,
    epoch_time,
    batches_per_sec=None,
    consolidations=None,
    **kwargs
):
    """
    Print end-of-epoch summary.
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        avg_loss: Average loss for epoch
        epoch_time: Epoch duration in seconds
        batches_per_sec: Throughput in batches/sec
        consolidations: Number of consolidations (CLS only)
        **kwargs: Additional metrics
    """
    print(f"\n📊 Epoch {epoch+1}/{total_epochs} Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    
    if consolidations is not None:
        print(f"  Consolidations: {consolidations}")
    
    if batches_per_sec is not None:
        print(f"  Time: {epoch_time:.1f}s ({batches_per_sec:.1f} batches/sec)")
    else:
        print(f"  Time: {epoch_time:.1f}s")
    
    # Print additional metrics
    for key, value in kwargs.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def calculate_eta(current_epoch, total_epochs, epoch_time):
    """
    Calculate estimated time remaining.
    
    Args:
        current_epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        epoch_time: Time for current epoch in seconds
    
    Returns:
        str: Formatted ETA string
    """
    if current_epoch >= total_epochs - 1:
        return "Complete"
    
    remaining_epochs = total_epochs - (current_epoch + 1)
    eta_seconds = epoch_time * remaining_epochs
    eta_minutes = eta_seconds / 60
    
    if eta_minutes < 60:
        return f"~{eta_minutes:.1f} minutes"
    else:
        return f"~{eta_minutes/60:.1f} hours"


def move_batch_to_device(batch, device):
    """
    Move batch to specified device.
    
    Args:
        batch: Dictionary of tensors
        device: Target device
    
    Returns:
        dict: Batch with tensors moved to device
    """
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
