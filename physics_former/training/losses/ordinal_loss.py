"""
Ordinal Regression Loss for Counting

Encourages the model to learn the ordinal structure of counts,
where 13 is closer to 14 than to 2. This helps with generalization
to higher counts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss that penalizes predictions based on 
    distance from the true count.
    
    Combines cross-entropy (for classification) with L1 distance loss
    (for ordinal structure).
    """
    
    def __init__(self, max_count=20, ordinal_weight=0.5, label_smoothing=0.0):
        """
        Args:
            max_count: Maximum count value (default: 20)
            ordinal_weight: Weight for ordinal loss component (default: 0.5)
            label_smoothing: Label smoothing for cross-entropy (default: 0.0)
        """
        super().__init__()
        self.max_count = max_count
        self.ordinal_weight = ordinal_weight
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch, num_classes] - raw logits from model
            targets: [batch] - true count labels
            
        Returns:
            loss: Combined classification + ordinal loss
        """
        # Classification loss (cross-entropy)
        ce_loss = F.cross_entropy(logits, targets, 
                                  label_smoothing=self.label_smoothing)
        
        # Ordinal loss (L1 distance)
        probs = F.softmax(logits, dim=-1)
        count_values = torch.arange(logits.size(-1), device=logits.device).float()
        
        # Expected count (soft prediction using probabilities)
        predicted_count = (probs * count_values).sum(dim=-1)
        
        # L1 distance between predicted and true count
        ordinal_loss = F.l1_loss(predicted_count, targets.float())
        
        # Combine losses
        total_loss = ce_loss + self.ordinal_weight * ordinal_loss
        
        return total_loss


class AdaptiveOrdinalLoss(nn.Module):
    """
    Adaptive ordinal loss that increases ordinal weight for higher counts.
    
    Rationale: High counts (15-20) benefit more from ordinal structure
    than low counts (1-5).
    """
    
    def __init__(self, max_count=20, base_ordinal_weight=0.3, 
                 high_count_threshold=10, high_ordinal_weight=0.8,
                 label_smoothing=0.0):
        """
        Args:
            max_count: Maximum count value
            base_ordinal_weight: Weight for low counts
            high_count_threshold: Counts >= this get higher weight
            high_ordinal_weight: Weight for high counts
            label_smoothing: Label smoothing for cross-entropy
        """
        super().__init__()
        self.max_count = max_count
        self.base_ordinal_weight = base_ordinal_weight
        self.high_count_threshold = high_count_threshold
        self.high_ordinal_weight = high_ordinal_weight
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch, num_classes]
            targets: [batch]
            
        Returns:
            loss: Adaptive ordinal loss
        """
        batch_size = targets.size(0)
        
        # Classification loss
        ce_loss = F.cross_entropy(logits, targets, 
                                  label_smoothing=self.label_smoothing,
                                  reduction='none')
        
        # Ordinal loss
        probs = F.softmax(logits, dim=-1)
        count_values = torch.arange(logits.size(-1), device=logits.device).float()
        predicted_count = (probs * count_values).sum(dim=-1)
        ordinal_loss = F.l1_loss(predicted_count, targets.float(), reduction='none')
        
        # Adaptive weighting based on count
        ordinal_weights = torch.where(
            targets >= self.high_count_threshold,
            torch.tensor(self.high_ordinal_weight, device=targets.device),
            torch.tensor(self.base_ordinal_weight, device=targets.device)
        )
        
        # Combine with adaptive weights
        total_loss = ce_loss + ordinal_weights * ordinal_loss
        
        return total_loss.mean()


class MonotonicOrdinalLoss(nn.Module):
    """
    Ordinal loss with monotonicity constraint.
    
    Encourages cumulative probabilities to be monotonic:
    P(count <= k) should increase with k.
    """
    
    def __init__(self, max_count=20, ordinal_weight=0.5, 
                 monotonic_weight=0.1, label_smoothing=0.0):
        """
        Args:
            max_count: Maximum count value
            ordinal_weight: Weight for ordinal L1 loss
            monotonic_weight: Weight for monotonicity constraint
            label_smoothing: Label smoothing for cross-entropy
        """
        super().__init__()
        self.max_count = max_count
        self.ordinal_weight = ordinal_weight
        self.monotonic_weight = monotonic_weight
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch, num_classes]
            targets: [batch]
            
        Returns:
            loss: Ordinal loss with monotonicity
        """
        # Classification loss
        ce_loss = F.cross_entropy(logits, targets, 
                                  label_smoothing=self.label_smoothing)
        
        # Ordinal loss
        probs = F.softmax(logits, dim=-1)
        count_values = torch.arange(logits.size(-1), device=logits.device).float()
        predicted_count = (probs * count_values).sum(dim=-1)
        ordinal_loss = F.l1_loss(predicted_count, targets.float())
        
        # Monotonicity constraint
        # Cumulative probabilities should be non-decreasing
        cumsum_probs = torch.cumsum(probs, dim=-1)
        
        # Penalize violations: cumsum[i] < cumsum[i-1]
        violations = F.relu(cumsum_probs[:, :-1] - cumsum_probs[:, 1:])
        monotonic_loss = violations.sum(dim=-1).mean()
        
        # Combine losses
        total_loss = (ce_loss + 
                     self.ordinal_weight * ordinal_loss + 
                     self.monotonic_weight * monotonic_loss)
        
        return total_loss


class RankConsistencyLoss(nn.Module):
    """
    Rank consistency loss for ordinal regression.
    
    Ensures that if count_a < count_b, then predicted_a < predicted_b.
    """
    
    def __init__(self, max_count=20, ordinal_weight=0.5, 
                 rank_weight=0.2, label_smoothing=0.0):
        """
        Args:
            max_count: Maximum count value
            ordinal_weight: Weight for ordinal L1 loss
            rank_weight: Weight for rank consistency
            label_smoothing: Label smoothing
        """
        super().__init__()
        self.max_count = max_count
        self.ordinal_weight = ordinal_weight
        self.rank_weight = rank_weight
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch, num_classes]
            targets: [batch]
            
        Returns:
            loss: Ordinal loss with rank consistency
        """
        batch_size = targets.size(0)
        
        # Classification loss
        ce_loss = F.cross_entropy(logits, targets, 
                                  label_smoothing=self.label_smoothing)
        
        # Ordinal loss
        probs = F.softmax(logits, dim=-1)
        count_values = torch.arange(logits.size(-1), device=logits.device).float()
        predicted_count = (probs * count_values).sum(dim=-1)
        ordinal_loss = F.l1_loss(predicted_count, targets.float())
        
        # Rank consistency loss
        # For pairs where target_i < target_j, ensure predicted_i < predicted_j
        rank_loss = 0.0
        if batch_size > 1:
            # Create pairwise comparisons
            target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # [batch, batch]
            pred_diff = predicted_count.unsqueeze(1) - predicted_count.unsqueeze(0)
            
            # Penalize rank violations
            # If target_i < target_j but predicted_i >= predicted_j
            violations = F.relu(-pred_diff * torch.sign(target_diff))
            rank_loss = violations.mean()
        
        # Combine losses
        total_loss = (ce_loss + 
                     self.ordinal_weight * ordinal_loss + 
                     self.rank_weight * rank_loss)
        
        return total_loss
