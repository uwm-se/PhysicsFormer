"""
Training Diagnostics and Fixes

Utilities to diagnose and fix common training issues:
- Gradient explosion detection
- Data normalization verification
- Loss scale analysis
- Model weight initialization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


def diagnose_gradients(model: nn.Module, max_norm: float = 1.0) -> Dict[str, float]:
    """
    Diagnose gradient health.
    
    Args:
        model: PyTorch model
        max_norm: Expected maximum gradient norm
        
    Returns:
        Dictionary with gradient statistics
    """
    total_norm = 0.0
    max_grad = 0.0
    min_grad = float('inf')
    num_params = 0
    nan_count = 0
    inf_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            max_grad = max(max_grad, param.grad.abs().max().item())
            min_grad = min(min_grad, param.grad.abs().min().item())
            num_params += 1
            
            # Check for NaN/Inf
            if torch.isnan(param.grad).any():
                nan_count += 1
                print(f"  ⚠️  NaN gradient in {name}")
            if torch.isinf(param.grad).any():
                inf_count += 1
                print(f"  ⚠️  Inf gradient in {name}")
    
    total_norm = total_norm ** 0.5
    
    stats = {
        'total_norm': total_norm,
        'max_grad': max_grad,
        'min_grad': min_grad,
        'num_params': num_params,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'is_healthy': total_norm < max_norm * 10 and nan_count == 0 and inf_count == 0
    }
    
    # Print diagnosis
    print("\n" + "="*70)
    print("GRADIENT DIAGNOSTICS")
    print("="*70)
    print(f"Total Norm: {total_norm:.2f} (target: <{max_norm*10:.1f})")
    print(f"Max Gradient: {max_grad:.4f}")
    print(f"Min Gradient: {min_grad:.4f}")
    print(f"Parameters with gradients: {num_params}")
    
    if nan_count > 0:
        print(f"❌ NaN gradients detected: {nan_count} parameters")
    if inf_count > 0:
        print(f"❌ Inf gradients detected: {inf_count} parameters")
    
    if total_norm > max_norm * 10:
        print(f"⚠️  GRADIENT EXPLOSION DETECTED!")
        print(f"   Recommendation: Reduce learning rate or increase warmup")
    elif stats['is_healthy']:
        print(f"✅ Gradients are healthy")
    
    print("="*70 + "\n")
    
    return stats


def diagnose_data_scale(batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    Diagnose data scaling issues.
    
    Args:
        batch: Training batch
        
    Returns:
        Dictionary with statistics for each tensor
    """
    stats = {}
    
    print("\n" + "="*70)
    print("DATA SCALE DIAGNOSTICS")
    print("="*70)
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_stats = {
                'mean': value.float().mean().item(),
                'std': value.float().std().item(),
                'min': value.float().min().item(),
                'max': value.float().max().item(),
                'shape': tuple(value.shape)
            }
            stats[key] = tensor_stats
            
            print(f"\n{key}:")
            print(f"  Shape: {tensor_stats['shape']}")
            print(f"  Range: [{tensor_stats['min']:.4f}, {tensor_stats['max']:.4f}]")
            print(f"  Mean: {tensor_stats['mean']:.4f}, Std: {tensor_stats['std']:.4f}")
            
            # Check for issues
            if abs(tensor_stats['mean']) > 100:
                print(f"  ⚠️  Large mean value - consider normalization")
            if tensor_stats['std'] > 100:
                print(f"  ⚠️  Large variance - consider normalization")
            if tensor_stats['max'] > 1000:
                print(f"  ⚠️  Very large values - may cause gradient explosion")
    
    print("="*70 + "\n")
    
    return stats


def diagnose_loss_scale(
    loss: torch.Tensor,
    loss_components: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, float]:
    """
    Diagnose loss scale issues.
    
    Args:
        loss: Total loss
        loss_components: Dictionary of individual loss components
        
    Returns:
        Dictionary with loss statistics
    """
    stats = {
        'total_loss': loss.item()
    }
    
    print("\n" + "="*70)
    print("LOSS SCALE DIAGNOSTICS")
    print("="*70)
    print(f"Total Loss: {loss.item():.4f}")
    
    if loss.item() > 1000:
        print(f"❌ LOSS EXPLOSION DETECTED!")
        print(f"   Recommendation: Check data normalization and loss weights")
    elif loss.item() > 100:
        print(f"⚠️  High loss value - may indicate scaling issues")
    else:
        print(f"✅ Loss scale is reasonable")
    
    if loss_components:
        print(f"\nLoss Components:")
        for name, component_loss in loss_components.items():
            value = component_loss.item()
            stats[name] = value
            print(f"  {name}: {value:.4f}")
            
            if value > 1000:
                print(f"    ⚠️  Component is very large - reduce weight")
    
    print("="*70 + "\n")
    
    return stats


def initialize_weights_properly(model: nn.Module, scale: float = 0.02):
    """
    Properly initialize model weights for stable training.
    
    Args:
        model: PyTorch model
        scale: Initialization scale
    """
    print("\n" + "="*70)
    print("WEIGHT INITIALIZATION")
    print("="*70)
    
    initialized_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            initialized_count += 1
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=scale)
            initialized_count += 1
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            initialized_count += 1
    
    print(f"✅ Initialized {initialized_count} modules with scale={scale}")
    print("="*70 + "\n")


def verify_gradient_clipping(
    model: nn.Module,
    max_norm: float,
    before_clip: Optional[float] = None
) -> Tuple[float, bool]:
    """
    Verify gradient clipping is working.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        before_clip: Gradient norm before clipping (optional)
        
    Returns:
        Tuple of (current_norm, was_clipped)
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    was_clipped = before_clip is not None and before_clip > max_norm
    
    if was_clipped:
        print(f"✂️  Gradient clipped: {before_clip:.2f} -> {total_norm:.2f} (max: {max_norm})")
    
    return total_norm, was_clipped


def suggest_hyperparameters(
    current_loss: float,
    gradient_norm: float,
    learning_rate: float
) -> Dict[str, float]:
    """
    Suggest hyperparameter adjustments based on training state.
    
    Args:
        current_loss: Current loss value
        gradient_norm: Current gradient norm
        learning_rate: Current learning rate
        
    Returns:
        Dictionary with suggested hyperparameters
    """
    suggestions = {}
    
    print("\n" + "="*70)
    print("HYPERPARAMETER SUGGESTIONS")
    print("="*70)
    
    # Loss-based suggestions
    if current_loss > 1000:
        suggestions['learning_rate'] = learning_rate * 0.1
        print(f"❌ Loss too high ({current_loss:.1f})")
        print(f"   Reduce LR: {learning_rate:.2e} -> {suggestions['learning_rate']:.2e}")
        suggestions['energy_weight'] = 0.01
        suggestions['momentum_weight'] = 0.01
        print(f"   Reduce physics loss weights to 0.01")
    
    # Gradient-based suggestions
    if gradient_norm > 10:
        suggestions['max_grad_norm'] = 0.5
        print(f"⚠️  Gradients too large ({gradient_norm:.1f})")
        print(f"   Reduce max_grad_norm to 0.5")
        if 'learning_rate' not in suggestions:
            suggestions['learning_rate'] = learning_rate * 0.5
            print(f"   Reduce LR: {learning_rate:.2e} -> {suggestions['learning_rate']:.2e}")
    
    # Warmup suggestions
    if gradient_norm > 5 or current_loss > 500:
        suggestions['warmup_steps'] = 5000
        print(f"⚠️  Training unstable")
        print(f"   Increase warmup_steps to 5000")
    
    if not suggestions:
        print(f"✅ Current hyperparameters look good!")
    
    print("="*70 + "\n")
    
    return suggestions


def run_full_diagnostics(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss: torch.Tensor,
    loss_components: Optional[Dict[str, torch.Tensor]],
    learning_rate: float,
    max_grad_norm: float
) -> Dict[str, any]:
    """
    Run complete training diagnostics.
    
    Args:
        model: PyTorch model
        batch: Training batch
        loss: Total loss
        loss_components: Individual loss components
        learning_rate: Current learning rate
        max_grad_norm: Maximum gradient norm
        
    Returns:
        Dictionary with all diagnostic results
    """
    print("\n" + "="*70)
    print("RUNNING FULL TRAINING DIAGNOSTICS")
    print("="*70 + "\n")
    
    results = {}
    
    # 1. Data scale
    results['data_stats'] = diagnose_data_scale(batch)
    
    # 2. Loss scale
    results['loss_stats'] = diagnose_loss_scale(loss, loss_components)
    
    # 3. Gradient health
    results['gradient_stats'] = diagnose_gradients(model, max_grad_norm)
    
    # 4. Suggestions
    results['suggestions'] = suggest_hyperparameters(
        current_loss=loss.item(),
        gradient_norm=results['gradient_stats']['total_norm'],
        learning_rate=learning_rate
    )
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    issues = []
    if results['loss_stats']['total_loss'] > 1000:
        issues.append("Loss explosion")
    if results['gradient_stats']['total_norm'] > max_grad_norm * 10:
        issues.append("Gradient explosion")
    if results['gradient_stats']['nan_count'] > 0:
        issues.append("NaN gradients")
    if results['gradient_stats']['inf_count'] > 0:
        issues.append("Inf gradients")
    
    if issues:
        print(f"❌ Issues detected: {', '.join(issues)}")
        print(f"\n⚠️  TRAINING IS UNSTABLE - STOP AND FIX ISSUES")
    else:
        print(f"✅ No critical issues detected")
        print(f"   Training can continue")
    
    print("="*70 + "\n")
    
    return results
