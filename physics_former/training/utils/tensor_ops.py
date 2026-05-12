"""
Safe Tensor Operations

Provides numerically stable tensor operations with proper error handling.
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def safe_masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute masked mean with protection against division by zero.
    
    Args:
        tensor: Input tensor [batch, seq_len, hidden_dim]
        mask: Binary mask [batch, seq_len]
        dim: Dimension to reduce (default: 1)
        eps: Epsilon for numerical stability
    
    Returns:
        Masked mean [batch, hidden_dim]
    
    Example:
        >>> tensor = torch.randn(4, 10, 128)
        >>> mask = torch.randint(0, 2, (4, 10))
        >>> result = safe_masked_mean(tensor, mask, dim=1)
        >>> result.shape
        torch.Size([4, 128])
    """
    mask_expanded = mask.unsqueeze(-1).float()
    masked_sum = (tensor * mask_expanded).sum(dim=dim)
    mask_sum = mask_expanded.sum(dim=dim).clamp(min=eps)
    return masked_sum / mask_sum


def safe_masked_sum(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1
) -> torch.Tensor:
    """
    Compute masked sum.
    
    Args:
        tensor: Input tensor
        mask: Binary mask
        dim: Dimension to reduce
    
    Returns:
        Masked sum
    """
    mask_expanded = mask.unsqueeze(-1).float()
    return (tensor * mask_expanded).sum(dim=dim)


def safe_division(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Safe division with protection against division by zero.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        eps: Epsilon for numerical stability
    
    Returns:
        numerator / (denominator + eps)
    """
    return numerator / (denominator + eps)


def safe_sqrt(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Safe square root with protection against negative values.
    
    Args:
        tensor: Input tensor
        eps: Epsilon for numerical stability
    
    Returns:
        sqrt(max(tensor, eps))
    """
    return torch.sqrt(torch.clamp(tensor, min=eps))


def safe_log(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Safe logarithm with protection against zero/negative values.
    
    Args:
        tensor: Input tensor
        eps: Epsilon for numerical stability
    
    Returns:
        log(max(tensor, eps))
    """
    return torch.log(torch.clamp(tensor, min=eps))


def safe_normalize(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Safe L2 normalization.
    
    Args:
        tensor: Input tensor
        dim: Dimension to normalize
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor
    """
    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    return tensor / (norm + eps)


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute softmax with masking.
    
    Args:
        logits: Input logits [batch, seq_len, vocab]
        mask: Binary mask [batch, seq_len]
        dim: Dimension for softmax
    
    Returns:
        Masked softmax probabilities
    """
    # Set masked positions to large negative value
    masked_logits = logits.masked_fill(~mask.bool(), float('-inf'))
    return torch.softmax(masked_logits, dim=dim)


def compute_masked_loss(
    loss_per_element: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute loss with masking.
    
    Args:
        loss_per_element: Loss for each element [batch, seq_len]
        mask: Binary mask [batch, seq_len]
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Masked loss
    """
    masked_loss = loss_per_element * mask.float()
    
    if reduction == 'none':
        return masked_loss
    elif reduction == 'sum':
        return masked_loss.sum()
    elif reduction == 'mean':
        return masked_loss.sum() / (mask.sum() + 1e-8)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def gradient_clipping_norm(
    parameters,
    max_norm: float = 1.0,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by norm and return the total norm.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum norm
        norm_type: Type of norm (2.0 for L2)
    
    Returns:
        Total norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)


def check_for_nan(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Check if tensor contains NaN values.
    
    Args:
        tensor: Tensor to check
        name: Name for error message
    
    Returns:
        True if contains NaN, False otherwise
    
    Raises:
        ValueError: If NaN detected
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    return False


def check_for_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Check if tensor contains Inf values.
    
    Args:
        tensor: Tensor to check
        name: Name for error message
    
    Returns:
        True if contains Inf, False otherwise
    
    Raises:
        ValueError: If Inf detected
    """
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")
    return False


def safe_tensor_operation(func):
    """
    Decorator to check for NaN/Inf after tensor operations.
    
    Example:
        @safe_tensor_operation
        def my_function(x):
            return x / x.sum()
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            check_for_nan(result, func.__name__)
            check_for_inf(result, func.__name__)
        return result
    return wrapper


def align_sequence_to_predictions(
    tensor: torch.Tensor,
    predicted_seq_len: int,
    seq_dim: int = 1
) -> torch.Tensor:
    """
    Align a sequence tensor to match prediction sequence length.
    
    When models predict next states, they produce seq_len-1 predictions from seq_len inputs.
    This function adjusts input sequences (like masks) to match prediction length.
    
    Args:
        tensor: Input tensor with sequence dimension
        predicted_seq_len: Target sequence length (typically seq_len - 1)
        seq_dim: Dimension index for sequence (default: 1 for [batch, seq_len, ...])
    
    Returns:
        Tensor with adjusted sequence length
    
    Example:
        >>> mask = torch.ones(4, 192, 25)  # [batch, seq_len, objects]
        >>> predictions_len = 191  # Model predicts seq_len - 1
        >>> aligned_mask = align_sequence_to_predictions(mask, predictions_len)
        >>> aligned_mask.shape
        torch.Size([4, 191, 25])
    """
    current_seq_len = tensor.shape[seq_dim]
    
    logger.debug(
        f"align_sequence_to_predictions: tensor_shape={tuple(tensor.shape)}, "
        f"current_seq_len={current_seq_len}, predicted_seq_len={predicted_seq_len}, seq_dim={seq_dim}"
    )
    
    # If already aligned, return as-is
    if current_seq_len == predicted_seq_len:
        logger.debug("Sequence already aligned, returning as-is")
        return tensor
    
    # If tensor has one extra timestep, remove the last one
    if current_seq_len == predicted_seq_len + 1:
        # Build slice to remove last timestep
        slices = [slice(None)] * tensor.ndim
        slices[seq_dim] = slice(None, -1)  # All except last
        result = tensor[tuple(slices)]
        logger.debug(f"Aligned sequence by removing last timestep: {tuple(tensor.shape)} -> {tuple(result.shape)}")
        return result
    
    # Unexpected mismatch
    error_msg = (
        f"Cannot align sequence: tensor has {current_seq_len} timesteps "
        f"but predictions have {predicted_seq_len}. Expected difference of 0 or 1. "
        f"Tensor shape: {tuple(tensor.shape)}, seq_dim: {seq_dim}"
    )
    logger.error(error_msg)
    raise ValueError(error_msg)


def prepare_sequence_mask_for_loss(
    mask: Optional[torch.Tensor],
    predicted_states: torch.Tensor,
    seq_dim: int = 1
) -> Optional[torch.Tensor]:
    """
    Prepare object mask for loss computation with sequence predictions.
    
    Handles alignment and expansion of masks to match predicted states shape.
    This is a common pattern when computing masked losses on sequence predictions.
    
    Args:
        mask: Object mask [batch, seq_len, objects] or [batch, objects] or None
        predicted_states: Predicted states [batch, seq_len-1, objects, features] or [batch, objects, features]
        seq_dim: Sequence dimension index (default: 1)
    
    Returns:
        Expanded mask [batch, seq_len-1, objects, 1] or [batch, objects, 1] or None
    
    Example:
        >>> mask = torch.ones(4, 192, 25)
        >>> predictions = torch.randn(4, 191, 25, 6)
        >>> expanded_mask = prepare_sequence_mask_for_loss(mask, predictions)
        >>> expanded_mask.shape
        torch.Size([4, 191, 25, 1])
    """
    logger.debug(
        f"prepare_sequence_mask_for_loss: "
        f"mask_shape={tuple(mask.shape) if mask is not None else None}, "
        f"predictions_shape={tuple(predicted_states.shape)}, seq_dim={seq_dim}"
    )
    
    if mask is None:
        logger.debug("Mask is None, returning None")
        return None
    
    is_sequence = predicted_states.dim() == 4
    
    if is_sequence and mask.dim() == 3:
        # Sequence case: align mask to prediction length
        logger.debug("Processing sequence mask (4D predictions, 3D mask)")
        pred_seq_len = predicted_states.shape[seq_dim]
        mask = align_sequence_to_predictions(mask, pred_seq_len, seq_dim=seq_dim)
        # Expand: [batch, seq_len-1, objects] -> [batch, seq_len-1, objects, 1]
        result = mask.unsqueeze(-1).float()
        logger.debug(f"Expanded sequence mask: {tuple(mask.shape)} -> {tuple(result.shape)}")
        return result
    
    elif not is_sequence and mask.dim() == 2:
        # Single timestep case
        logger.debug("Processing single timestep mask (3D predictions, 2D mask)")
        # Expand: [batch, objects] -> [batch, objects, 1]
        result = mask.unsqueeze(-1).float()
        logger.debug(f"Expanded single timestep mask: {tuple(mask.shape)} -> {tuple(result.shape)}")
        return result
    
    else:
        # Dimension mismatch - cannot safely align
        logger.warning(
            f"Dimension mismatch in prepare_sequence_mask_for_loss: "
            f"predictions.dim()={predicted_states.dim()}, mask.dim()={mask.dim()}. "
            f"Expected (4D predictions + 3D mask) or (3D predictions + 2D mask). "
            f"Returning None (unmasked loss will be computed)."
        )
        return None


def extract_sequence_input_for_prediction(
    states: torch.Tensor,
    seq_dim: int = 1
) -> torch.Tensor:
    """
    Extract input states for next-state prediction from sequences.
    
    For autoregressive prediction, we use timesteps 0 to seq_len-2 to predict
    timesteps 1 to seq_len-1. This function extracts the input portion.
    
    Args:
        states: State sequence [batch, seq_len, objects, features]
        seq_dim: Sequence dimension index (default: 1)
    
    Returns:
        Input states [batch, seq_len-1, objects, features]
    
    Example:
        >>> states = torch.randn(4, 192, 25, 6)
        >>> inputs = extract_sequence_input_for_prediction(states)
        >>> inputs.shape
        torch.Size([4, 191, 25, 6])
    """
    logger.debug(
        f"extract_sequence_input_for_prediction: "
        f"states_shape={tuple(states.shape)}, seq_dim={seq_dim}"
    )
    
    if states.dim() < 3:
        error_msg = f"Expected at least 3D tensor, got {states.dim()}D with shape {tuple(states.shape)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Remove last timestep
    slices = [slice(None)] * states.ndim
    slices[seq_dim] = slice(None, -1)
    result = states[tuple(slices)]
    
    logger.debug(f"Extracted input sequence: {tuple(states.shape)} -> {tuple(result.shape)}")
    return result


def extract_last_timestep(
    tensor: torch.Tensor,
    seq_dim: int = 1
) -> torch.Tensor:
    """
    Extract last timestep from sequence tensor.
    
    Useful for extracting final state from sequences for classification or pooling.
    
    Args:
        tensor: Sequence tensor [batch, seq_len, ...] or non-sequence
        seq_dim: Sequence dimension index (default: 1)
    
    Returns:
        Last timestep [batch, ...] or original if not a sequence
    
    Example:
        >>> embeddings = torch.randn(4, 192, 25, 256)
        >>> last = extract_last_timestep(embeddings)
        >>> last.shape
        torch.Size([4, 25, 256])
    """
    logger.debug(f"extract_last_timestep: tensor_shape={tuple(tensor.shape)}, seq_dim={seq_dim}")
    
    if tensor.dim() >= 3:
        slices = [slice(None)] * tensor.ndim
        slices[seq_dim] = -1
        result = tensor[tuple(slices)]
        logger.debug(f"Extracted last timestep: {tuple(tensor.shape)} -> {tuple(result.shape)}")
        return result
    else:
        logger.debug(f"Tensor is not a sequence (dim < 3): {tuple(tensor.shape)}")
        return tensor


def extract_last_timestep_with_mask(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    seq_dim: int = 1
) -> tuple:
    """
    Extract last timestep from both tensor and mask.
    
    Convenience function for handling tensor and mask together.
    
    Args:
        tensor: Sequence tensor [batch, seq_len, ...]
        mask: Optional mask [batch, seq_len, ...] or None
        seq_dim: Sequence dimension index
    
    Returns:
        (tensor_last, mask_last) tuple
    
    Example:
        >>> embeddings = torch.randn(4, 192, 25, 256)
        >>> mask = torch.ones(4, 192, 25)
        >>> emb_last, mask_last = extract_last_timestep_with_mask(embeddings, mask)
        >>> emb_last.shape, mask_last.shape
        (torch.Size([4, 25, 256]), torch.Size([4, 25]))
    """
    logger.debug(
        f"extract_last_timestep_with_mask: "
        f"tensor_shape={tuple(tensor.shape)}, "
        f"mask_shape={tuple(mask.shape) if mask is not None else None}"
    )
    
    tensor_last = extract_last_timestep(tensor, seq_dim)
    
    if mask is not None:
        mask_last = extract_last_timestep(mask, seq_dim)
        logger.debug(f"Extracted both: tensor={tuple(tensor_last.shape)}, mask={tuple(mask_last.shape)}")
        return tensor_last, mask_last
    
    logger.debug(f"Extracted tensor only: {tuple(tensor_last.shape)}")
    return tensor_last, None


def masked_pool_embeddings(
    embeddings: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = 1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Pool embeddings with optional masking.
    
    This is a convenience wrapper around safe_masked_mean with better naming
    for the common use case of pooling object embeddings.
    
    Args:
        embeddings: [batch, objects, hidden_dim] or [batch, seq_len, objects, hidden_dim]
        mask: [batch, objects] or [batch, seq_len, objects] or None
        dim: Dimension to pool over (default: 1 for objects)
        eps: Epsilon for numerical stability
    
    Returns:
        Pooled embeddings [batch, hidden_dim] or [batch, objects, hidden_dim]
    
    Example:
        >>> embeddings = torch.randn(4, 25, 256)
        >>> mask = torch.ones(4, 25)
        >>> pooled = masked_pool_embeddings(embeddings, mask)
        >>> pooled.shape
        torch.Size([4, 256])
    """
    logger.debug(
        f"masked_pool_embeddings: embeddings={tuple(embeddings.shape)}, "
        f"mask={tuple(mask.shape) if mask is not None else None}, dim={dim}"
    )
    
    if mask is not None:
        result = safe_masked_mean(embeddings, mask, dim=dim, eps=eps)
        logger.debug(f"Masked pooling result: {tuple(result.shape)}")
    else:
        result = embeddings.mean(dim=dim)
        logger.debug(f"Unmasked pooling result: {tuple(result.shape)}")
    
    return result


def compute_masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
    use_huber: bool = True,
    huber_delta: float = 1.0
) -> torch.Tensor:
    """
    Compute loss with optional masking and automatic sequence alignment.
    
    Uses Huber loss by default for robustness to outliers (prevents gradient explosion
    on extreme physics values like high-speed collisions, chaos scenarios).
    
    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional mask (will be aligned to predictions if needed)
        reduction: 'mean', 'sum', or 'none'
        use_huber: If True, use Huber loss instead of MSE (default: True)
        huber_delta: Threshold for Huber loss (default: 1.0)
    
    Returns:
        Loss (scalar or per-element depending on reduction)
    
    Example:
        >>> predictions = torch.randn(4, 191, 25, 6)
        >>> targets = torch.randn(4, 191, 25, 6)
        >>> mask = torch.ones(4, 192, 25)  # Will be auto-aligned
        >>> loss = compute_masked_mse_loss(predictions, targets, mask)
    """
    logger.debug(
        f"compute_masked_mse_loss: pred={tuple(predictions.shape)}, "
        f"target={tuple(targets.shape)}, mask={tuple(mask.shape) if mask is not None else None}, "
        f"reduction={reduction}, use_huber={use_huber}"
    )
    
    # Prepare mask if provided (handles sequence alignment)
    if mask is not None:
        mask_expanded = prepare_sequence_mask_for_loss(mask, predictions)
        if mask_expanded is not None:
            # Compute element-wise loss (Huber or MSE)
            if use_huber:
                loss_per_element = torch.nn.functional.huber_loss(
                    predictions, targets, reduction='none', delta=huber_delta
                )
            else:
                loss_per_element = (predictions - targets) ** 2
            
            masked_loss = loss_per_element * mask_expanded
            
            if reduction == 'sum':
                result = masked_loss.sum()
            elif reduction == 'mean':
                result = masked_loss.sum() / mask_expanded.sum()
            else:  # 'none'
                result = masked_loss
            
            logger.debug(f"Masked {'Huber' if use_huber else 'MSE'} loss: {result.item() if reduction != 'none' else 'per-element'}")
            return result
    
    # Unmasked loss
    if use_huber:
        result = torch.nn.functional.huber_loss(predictions, targets, reduction=reduction, delta=huber_delta)
    else:
        result = torch.nn.functional.mse_loss(predictions, targets, reduction=reduction)
    logger.debug(f"Unmasked {'Huber' if use_huber else 'MSE'} loss: {result.item() if reduction != 'none' else 'per-element'}")
    return result


__all__ = [
    'safe_masked_mean',
    'safe_masked_sum',
    'safe_division',
    'safe_sqrt',
    'safe_log',
    'safe_normalize',
    'masked_softmax',
    'compute_masked_loss',
    'gradient_clipping_norm',
    'check_for_nan',
    'check_for_inf',
    'safe_tensor_operation',
    'align_sequence_to_predictions',
    'prepare_sequence_mask_for_loss',
    'extract_sequence_input_for_prediction',
    'extract_last_timestep',
    'extract_last_timestep_with_mask',
    'masked_pool_embeddings',
    'compute_masked_mse_loss',
]
