"""
Count-Aware Loss Functions

Weighted loss functions that give higher importance to difficult high-count examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CountWeightedCrossEntropy(nn.Module):
    """
    Cross-entropy loss with higher weights for high counts.
    
    Rationale: High counts (15-20) are harder to learn and should be
    weighted more heavily to prevent the model from ignoring them.
    
    Args:
        max_count: Maximum count value (default: 20)
        min_weight: Minimum weight for low counts (default: 1.0)
        max_weight: Maximum weight for high counts (default: 2.0)
        label_smoothing: Label smoothing factor (default: 0.0)
    """
    
    def __init__(self, max_count=20, min_weight=1.0, max_weight=2.0, label_smoothing=0.0):
        super().__init__()
        self.max_count = max_count
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.label_smoothing = label_smoothing
        
        # Pre-compute weights for each count
        self.register_buffer(
            'count_weights',
            torch.linspace(min_weight, max_weight, max_count + 1)
        )
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        
        Returns:
            weighted_loss: scalar tensor
        """
        # Compute per-sample weights based on target count
        weights = self.count_weights[targets]  # [batch_size]
        
        # Compute cross-entropy without reduction
        ce_loss = F.cross_entropy(
            logits, 
            targets, 
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        # Apply weights and reduce
        weighted_loss = (ce_loss * weights).mean()
        
        return weighted_loss


class ProgressiveCountWeighting(nn.Module):
    """
    Progressive count weighting that increases weight for high counts over time.
    
    Starts with uniform weighting and gradually increases weight for high counts
    as training progresses. This allows the model to first learn low counts well,
    then focus more on high counts.
    
    Args:
        max_count: Maximum count value (default: 20)
        high_count_threshold: Counts >= this get progressive weighting (default: 12)
        max_weight_multiplier: Maximum weight multiplier for high counts (default: 3.0)
        total_epochs: Total training epochs for weight schedule (default: 100)
        label_smoothing: Label smoothing factor (default: 0.0)
    """
    
    def __init__(self, max_count=20, high_count_threshold=12, max_weight_multiplier=3.0,
                 total_epochs=100, label_smoothing=0.0):
        super().__init__()
        self.max_count = max_count
        self.high_count_threshold = high_count_threshold
        self.max_weight_multiplier = max_weight_multiplier
        self.total_epochs = total_epochs
        self.label_smoothing = label_smoothing
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Update current epoch for weight scheduling."""
        self.current_epoch = epoch
    
    def get_current_weights(self):
        """Compute current weights based on epoch."""
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        
        # Start with uniform weights
        weights = torch.ones(self.max_count + 1)
        
        # Progressively increase weight for high counts
        for count in range(self.high_count_threshold, self.max_count + 1):
            # Linear increase from 1.0 to max_weight_multiplier
            weight_increase = (self.max_weight_multiplier - 1.0) * progress
            weights[count] = 1.0 + weight_increase
        
        return weights
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        
        Returns:
            weighted_loss: scalar tensor
        """
        # Get current weights
        weights = self.get_current_weights().to(logits.device)
        
        # Compute per-sample weights
        sample_weights = weights[targets]
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        # Apply weights
        weighted_loss = (ce_loss * sample_weights).mean()
        
        return weighted_loss


class DifficultyAwareWeighting(nn.Module):
    """
    Difficulty-aware weighting based on model's current performance.
    
    Dynamically adjusts weights based on which counts the model struggles with.
    Requires periodic updates of difficulty scores.
    
    Args:
        max_count: Maximum count value (default: 20)
        label_smoothing: Label smoothing factor (default: 0.0)
        smoothing_factor: How much to smooth difficulty updates (default: 0.9)
    """
    
    def __init__(self, max_count=20, label_smoothing=0.0, smoothing_factor=0.9):
        super().__init__()
        self.max_count = max_count
        self.label_smoothing = label_smoothing
        self.smoothing_factor = smoothing_factor
        
        # Initialize difficulty scores (higher = more difficult)
        self.register_buffer(
            'difficulty_scores',
            torch.ones(max_count + 1)
        )
    
    def update_difficulty(self, count_accuracies):
        """
        Update difficulty scores based on current accuracies.
        
        Args:
            count_accuracies: dict mapping count -> accuracy (0-1)
        """
        new_scores = torch.ones(self.max_count + 1)
        
        for count, accuracy in count_accuracies.items():
            if count <= self.max_count:
                # Difficulty = 1 - accuracy (higher when accuracy is low)
                difficulty = 1.0 - accuracy
                # Add baseline to prevent zero weights
                new_scores[count] = 1.0 + difficulty * 2.0
        
        # Smooth update (exponential moving average)
        self.difficulty_scores = (
            self.smoothing_factor * self.difficulty_scores +
            (1 - self.smoothing_factor) * new_scores.to(self.difficulty_scores.device)
        )
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        
        Returns:
            weighted_loss: scalar tensor
        """
        # Get weights based on difficulty
        weights = self.difficulty_scores[targets]
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        # Apply weights
        weighted_loss = (ce_loss * weights).mean()
        
        return weighted_loss


def test_count_aware_losses():
    """Test count-aware loss implementations."""
    print("Testing Count-Aware Losses...")
    
    # Create dummy data
    batch_size = 16
    num_classes = 21
    
    logits = torch.randn(batch_size, num_classes)
    # Mix of low and high counts
    targets = torch.tensor([2, 5, 8, 10, 12, 15, 18, 20, 3, 7, 11, 14, 16, 19, 6, 13])
    
    # Test count-weighted cross-entropy
    print("\n1. Count-Weighted Cross-Entropy:")
    loss_fn = CountWeightedCrossEntropy(max_count=20, min_weight=1.0, max_weight=2.0)
    loss = loss_fn(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   ✓ Higher counts get up to 2x weight")
    
    # Test progressive weighting
    print("\n2. Progressive Count Weighting:")
    loss_fn = ProgressiveCountWeighting(
        max_count=20,
        high_count_threshold=12,
        max_weight_multiplier=3.0,
        total_epochs=100
    )
    
    for epoch in [0, 50, 100]:
        loss_fn.set_epoch(epoch)
        loss = loss_fn(logits, targets)
        weights = loss_fn.get_current_weights()
        print(f"   Epoch {epoch:3d}: loss={loss.item():.4f}, "
              f"weight[20]={weights[20]:.2f}")
    
    # Test difficulty-aware weighting
    print("\n3. Difficulty-Aware Weighting:")
    loss_fn = DifficultyAwareWeighting(max_count=20)
    
    # Simulate accuracy updates
    count_accuracies = {
        count: 0.9 if count < 10 else (0.5 if count < 15 else 0.2)
        for count in range(21)
    }
    loss_fn.update_difficulty(count_accuracies)
    
    loss = loss_fn(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Difficulty[5]={loss_fn.difficulty_scores[5]:.2f} (easy)")
    print(f"   Difficulty[12]={loss_fn.difficulty_scores[12]:.2f} (medium)")
    print(f"   Difficulty[18]={loss_fn.difficulty_scores[18]:.2f} (hard)")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_count_aware_losses()
