"""
Focal Loss for Classification

Focuses training on hard examples by down-weighting easy examples.
Particularly useful for imbalanced datasets or when some classes are harder to learn.

Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t is the model's estimated probability for the true class
    - gamma is the focusing parameter (higher = more focus on hard examples)
    - alpha_t is a weighting factor for class imbalance
    
    Args:
        gamma: Focusing parameter (default: 2.0)
            - gamma = 0: equivalent to cross-entropy
            - gamma > 0: focuses more on hard examples
        alpha: Class weights (default: None)
            - None: no class weighting
            - float: same weight for all classes
            - Tensor: per-class weights
        label_smoothing: Label smoothing factor (default: 0.0)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    """
    
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes] - raw model outputs
            targets: [batch_size] - ground truth class indices
        
        Returns:
            loss: scalar tensor
        """
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probability of true class
        batch_size = logits.size(0)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [batch_size]
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross-entropy loss (without reduction)
        ce_loss = F.cross_entropy(
            logits, 
            targets, 
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Per-class alpha
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that adjusts gamma based on training progress.
    
    Starts with high gamma (focus on hard examples) and gradually
    decreases to standard cross-entropy as training progresses.
    
    Args:
        gamma_start: Initial gamma value (default: 3.0)
        gamma_end: Final gamma value (default: 0.5)
        total_epochs: Total training epochs for gamma schedule (default: 100)
        alpha: Class weights (default: None)
        label_smoothing: Label smoothing factor (default: 0.0)
    """
    
    def __init__(self, gamma_start=3.0, gamma_end=0.5, total_epochs=100, 
                 alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.total_epochs = total_epochs
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Update current epoch for gamma scheduling."""
        self.current_epoch = epoch
    
    def get_current_gamma(self):
        """Compute current gamma based on epoch."""
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * progress
        return gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        
        Returns:
            loss: scalar tensor
        """
        gamma = self.get_current_gamma()
        
        # Use FocalLoss with current gamma
        focal_loss = FocalLoss(
            gamma=gamma,
            alpha=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='mean'
        )
        
        return focal_loss(logits, targets)


def test_focal_loss():
    """Test focal loss implementation."""
    print("Testing Focal Loss...")
    
    # Create dummy data
    batch_size = 8
    num_classes = 21  # 0-20 for counting
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test standard focal loss
    focal_loss = FocalLoss(gamma=2.0, label_smoothing=0.1)
    loss = focal_loss(logits, targets)
    print(f"✓ Focal Loss: {loss.item():.4f}")
    
    # Test with alpha weighting
    alpha = torch.ones(num_classes)
    alpha[15:] = 2.0  # Higher weight for counts 15-20
    focal_loss_weighted = FocalLoss(gamma=2.0, alpha=alpha, label_smoothing=0.1)
    loss_weighted = focal_loss_weighted(logits, targets)
    print(f"✓ Weighted Focal Loss: {loss_weighted.item():.4f}")
    
    # Test adaptive focal loss
    adaptive_loss = AdaptiveFocalLoss(gamma_start=3.0, gamma_end=0.5, total_epochs=100)
    
    for epoch in [0, 50, 100]:
        adaptive_loss.set_epoch(epoch)
        loss = adaptive_loss(logits, targets)
        gamma = adaptive_loss.get_current_gamma()
        print(f"✓ Epoch {epoch:3d}: gamma={gamma:.2f}, loss={loss.item():.4f}")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_focal_loss()
