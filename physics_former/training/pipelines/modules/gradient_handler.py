"""
Gradient Handling Module

Manages gradient clipping, NaN/Inf detection, and gradient statistics.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GradientHandler:
    """Handles gradient clipping and safety checks."""
    
    def __init__(
        self,
        max_grad_norm: float = 10.0,
        max_grad_value: float = 1.0,
        detect_anomalies: bool = True
    ):
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value
        self.detect_anomalies = detect_anomalies
    
    def clip_and_check(self, model: nn.Module) -> Tuple[float, List[str]]:
        """
        Clip gradients and check for NaN/Inf.
        
        Returns:
            (gradient_norm, list of parameters with bad gradients)
        """
        # Clip by value first
        if self.max_grad_value:
            torch.nn.utils.clip_grad_value_(
                model.parameters(),
                clip_value=self.max_grad_value
            )
        
        # Clip by norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_grad_norm
        )
        
        # Check for NaN/Inf and zero them out
        bad_params = []
        if self.detect_anomalies:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        bad_params.append(name)
                        param.grad.zero_()
        
        return grad_norm.item(), bad_params
    
    def compute_gradient_stats(self, model: nn.Module) -> dict:
        """Compute gradient statistics for monitoring."""
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if not grad_norms:
            return {'mean': 0, 'max': 0, 'min': 0}
        
        return {
            'mean': sum(grad_norms) / len(grad_norms),
            'max': max(grad_norms),
            'min': min(grad_norms)
        }
