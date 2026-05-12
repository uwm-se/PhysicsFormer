"""
Optimizer and Scheduler Factory

Creates optimizers and learning rate schedulers with various configurations.
"""

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """Factory for creating optimizers and schedulers."""
    
    @staticmethod
    def create_optimizer(
        model,
        optimizer_type: str = 'adamw',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs
    ):
        """
        Create optimizer.
        
        Args:
            model: Model to optimize
            optimizer_type: 'adamw', 'adam', or 'sgd'
            learning_rate: Learning rate
            weight_decay: Weight decay
            **kwargs: Additional optimizer-specific arguments
        """
        if optimizer_type.lower() == 'adamw':
            optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=kwargs.get('momentum', 0.9),
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        logger.info(f"Created {optimizer_type.upper()} optimizer (lr={learning_rate})")
        return optimizer
    
    @staticmethod
    def create_scheduler(
        optimizer,
        scheduler_type: str = 'cosine',
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        min_lr: float = 1e-7,
        **kwargs
    ):
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: 'cosine', 'linear', or 'none'
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
            **kwargs: Additional scheduler-specific arguments
        """
        if scheduler_type.lower() == 'none':
            return None
        
        # Create warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Create main scheduler
        if scheduler_type.lower() == 'cosine':
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=min_lr
            )
        elif scheduler_type.lower() == 'linear':
            main_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=min_lr / optimizer.param_groups[0]['lr'],
                total_iters=total_steps - warmup_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        # Combine warmup + main
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        logger.info(f"Created {scheduler_type} scheduler with {warmup_steps} warmup steps")
        return scheduler
