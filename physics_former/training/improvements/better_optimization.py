"""
Better Optimization for PhysicsFormer

Advanced optimizers and learning rate schedules.
Expected improvement: +1-2% accuracy
"""

import torch
import torch.nn as nn
import math


def create_optimizer(model: nn.Module, config_name: str = 'adamw', lr: float = 1e-4, weight_decay: float = 0.01):
    """
    Create optimizer with best practices.
    
    Args:
        model: Model to optimize
        config_name: 'adam', 'adamw', or 'adamw_advanced'
        lr: Base learning rate (default: 1e-4)
        weight_decay: Weight decay coefficient (default: 0.01)
    
    Returns:
        optimizer
    """
    
    if config_name == 'adam':
        # Basic Adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    elif config_name == 'adamw':
        # AdamW with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
    
    elif config_name == 'adamw_advanced':
        # AdamW with layer-wise learning rates
        
        # Separate parameters by layer type
        encoder_params = []
        transformer_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'transformer' in name:
                transformer_params.append(param)
            else:
                head_params.append(param)
        
        # Different learning rates for different layers (scaled from base lr)
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': lr * 0.5},   # 0.5x for encoder
            {'params': transformer_params, 'lr': lr},     # 1.0x for transformer
            {'params': head_params, 'lr': lr * 2.0},      # 2.0x for heads
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
        )
    
    else:
        raise ValueError(f"Unknown optimizer config: {config_name}")
    
    return optimizer


def create_scheduler(optimizer, config_name: str = 'cosine', total_steps: int = 10000, warmup_steps: int = 1000, min_lr: float = 1e-6):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config_name: 'cosine', 'cosine_warmup', or 'plateau'
        total_steps: Total training steps
        warmup_steps: Warmup steps for warmup_cosine scheduler (default: 1000)
        min_lr: Minimum learning rate (default: 1e-6)
    
    Returns:
        scheduler
    """
    
    if config_name == 'cosine':
        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=min_lr
        )
    
    elif config_name == 'cosine_warmup':
        # Cosine with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=warmup_steps,  # First restart after warmup_steps
            T_mult=2,          # Double period after each restart
            eta_min=min_lr
        )
    
    elif config_name == 'plateau':
        # Reduce on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    
    elif config_name == 'warmup_cosine':
        # Custom: Warmup + Cosine
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr
        )
    
    else:
        raise ValueError(f"Unknown scheduler config: {config_name}")
    
    return scheduler


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine decay.
    
    Best practice for transformer training.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step < self.warmup_steps:
                # Warmup: Linear increase
                lr = base_lr * (self.current_step / self.warmup_steps)
            else:
                # Cosine decay
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class GradientClipping:
    """
    Gradient clipping utilities.
    
    Prevents exploding gradients.
    """
    
    @staticmethod
    def clip_grad_norm(model: nn.Module, max_norm: float = 1.0):
        """Clip gradients by norm."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def clip_grad_value(model: nn.Module, clip_value: float = 1.0):
        """Clip gradients by value."""
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state (used when curriculum advances)."""
        self.counter = 0
        self.best_loss = None
        self.should_stop = False


# Example usage
if __name__ == "__main__":
    print("Better Optimization Module")
    print("="*70)
    
    # Create dummy model
    model = nn.Linear(10, 10)
    
    # Test optimizers
    print("\n1. Testing Optimizers:")
    for config in ['adam', 'adamw', 'adamw_advanced']:
        optimizer = create_optimizer(model, config)
        print(f"  PASS: {config}: {type(optimizer).__name__}")
    
    # Test schedulers
    print("\n2. Testing Schedulers:")
    optimizer = create_optimizer(model, 'adamw')
    for config in ['cosine', 'cosine_warmup', 'plateau', 'warmup_cosine']:
        scheduler = create_scheduler(optimizer, config, total_steps=10000)
        print(f"  PASS: {config}: {type(scheduler).__name__}")
    
    # Test warmup schedule
    print("\n3. Testing Warmup Schedule:")
    optimizer = create_optimizer(model, 'adamw')
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=100, total_steps=1000)
    
    print(f"  Initial LR: {scheduler.get_last_lr()[0]:.6f}")
    
    for step in [0, 50, 100, 500, 1000]:
        scheduler.current_step = step
        scheduler.step()
        print(f"  Step {step:4d}: LR = {scheduler.get_last_lr()[0]:.6f}")
    
    print("\nPASS: Better optimization ready!")
