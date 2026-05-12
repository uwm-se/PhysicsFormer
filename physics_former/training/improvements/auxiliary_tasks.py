"""
Auxiliary Tasks for PhysicsFormer

Additional tasks that improve number representations.
Expected improvement: +3-5% accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryTaskHeads(nn.Module):
    """
    Additional task heads for richer learning.
    
    Tasks:
    1. Comparison: Which number is larger?
    2. Ordering: Sort numbers
    3. Magnitude: Is number small/medium/large?
    4. Parity: Is number even or odd?
    """
    
    def __init__(self, hidden_dim: int = 128, max_objects: int = 100):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        
        # 1. Comparison head
        self.comparison_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)  # greater, equal, less
        )
        
        # 2. Ordering head (for sequences)
        self.ordering_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, max_objects + 1)
        )
        
        # 3. Magnitude head
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5)  # very_small, small, medium, large, very_large
        )
        
        # 4. Parity head
        self.parity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # even, odd
        )
        
        # 5. Digit sum head (auxiliary for compositional understanding)
        self.digit_sum_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 46)  # digit sums 0-45 (max for 99)
        )
    
    def forward_comparison(self, embed_a, embed_b):
        """
        Compare two numbers.
        
        Args:
            embed_a: Embedding of first number
            embed_b: Embedding of second number
        
        Returns:
            logits: [batch, 3] - (greater, equal, less)
        """
        combined = torch.cat([embed_a, embed_b], dim=-1)
        return self.comparison_head(combined)
    
    def forward_magnitude(self, embed):
        """
        Classify magnitude of number.
        
        Args:
            embed: Number embedding
        
        Returns:
            logits: [batch, 5] - magnitude class
        """
        return self.magnitude_head(embed)
    
    def forward_parity(self, embed):
        """
        Classify parity (even/odd).
        
        Args:
            embed: Number embedding
        
        Returns:
            logits: [batch, 2] - (even, odd)
        """
        return self.parity_head(embed)
    
    def forward_digit_sum(self, embed):
        """
        Predict sum of digits.
        
        Args:
            embed: Number embedding
        
        Returns:
            logits: [batch, 46] - digit sum
        """
        return self.digit_sum_head(embed)


class AuxiliaryTaskLoss(nn.Module):
    """
    Combined loss for auxiliary tasks.
    """
    
    def __init__(
        self,
        comparison_weight: float = 0.2,
        magnitude_weight: float = 0.1,
        parity_weight: float = 0.1,
        digit_sum_weight: float = 0.1
    ):
        super().__init__()
        
        self.comparison_weight = comparison_weight
        self.magnitude_weight = magnitude_weight
        self.parity_weight = parity_weight
        self.digit_sum_weight = digit_sum_weight
    
    def forward(self, outputs, targets):
        """
        Compute auxiliary task losses.
        
        Args:
            outputs: Dict of auxiliary task outputs
            targets: Dict of auxiliary task targets
        
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        
        # Comparison loss
        if 'comparison' in outputs and 'comparison' in targets:
            losses['comparison'] = F.cross_entropy(
                outputs['comparison'],
                targets['comparison']
            )
        
        # Magnitude loss
        if 'magnitude' in outputs and 'magnitude' in targets:
            losses['magnitude'] = F.cross_entropy(
                outputs['magnitude'],
                targets['magnitude']
            )
        
        # Parity loss
        if 'parity' in outputs and 'parity' in targets:
            losses['parity'] = F.cross_entropy(
                outputs['parity'],
                targets['parity']
            )
        
        # Digit sum loss
        if 'digit_sum' in outputs and 'digit_sum' in targets:
            losses['digit_sum'] = F.cross_entropy(
                outputs['digit_sum'],
                targets['digit_sum']
            )
        
        # Total loss
        total_loss = (
            self.comparison_weight * losses.get('comparison', 0) +
            self.magnitude_weight * losses.get('magnitude', 0) +
            self.parity_weight * losses.get('parity', 0) +
            self.digit_sum_weight * losses.get('digit_sum', 0)
        )
        
        return total_loss, losses


def generate_auxiliary_targets(numbers):
    """
    Generate targets for auxiliary tasks.
    
    Args:
        numbers: Tensor of numbers [batch, 2] for (num1, num2)
    
    Returns:
        Dict of targets
    """
    num1, num2 = numbers[:, 0], numbers[:, 1]
    
    targets = {}
    
    # Comparison target
    comparison = torch.zeros_like(num1)
    comparison[num1 > num2] = 0  # greater
    comparison[num1 == num2] = 1  # equal
    comparison[num1 < num2] = 2  # less
    targets['comparison'] = comparison
    
    # Magnitude target (for num1)
    magnitude = torch.zeros_like(num1)
    magnitude[num1 < 10] = 0      # very_small
    magnitude[(num1 >= 10) & (num1 < 25)] = 1  # small
    magnitude[(num1 >= 25) & (num1 < 50)] = 2  # medium
    magnitude[(num1 >= 50) & (num1 < 75)] = 3  # large
    magnitude[num1 >= 75] = 4     # very_large
    targets['magnitude'] = magnitude
    
    # Parity target (for num1)
    parity = num1 % 2  # 0 for even, 1 for odd
    targets['parity'] = parity
    
    # Digit sum target (for num1)
    digit_sum = torch.zeros_like(num1)
    for i, n in enumerate(num1):
        digit_sum[i] = sum(int(d) for d in str(n.item()))
    targets['digit_sum'] = digit_sum
    
    return targets


# Example usage
if __name__ == "__main__":
    print("Auxiliary Tasks Module")
    print("="*70)
    
    # Create auxiliary heads
    aux_heads = AuxiliaryTaskHeads(hidden_dim=128, max_objects=100)
    
    print("\nAuxiliary task heads:")
    print(f"  1. Comparison: {aux_heads.comparison_head}")
    print(f"  2. Magnitude: {aux_heads.magnitude_head}")
    print(f"  3. Parity: {aux_heads.parity_head}")
    print(f"  4. Digit sum: {aux_heads.digit_sum_head}")
    
    # Test forward passes
    print("\nTesting forward passes:")
    batch_size = 4
    hidden_dim = 128
    
    embed_a = torch.randn(batch_size, hidden_dim)
    embed_b = torch.randn(batch_size, hidden_dim)
    
    comparison_out = aux_heads.forward_comparison(embed_a, embed_b)
    print(f"  Comparison output: {comparison_out.shape}")
    
    magnitude_out = aux_heads.forward_magnitude(embed_a)
    print(f"  Magnitude output: {magnitude_out.shape}")
    
    parity_out = aux_heads.forward_parity(embed_a)
    print(f"  Parity output: {parity_out.shape}")
    
    digit_sum_out = aux_heads.forward_digit_sum(embed_a)
    print(f"  Digit sum output: {digit_sum_out.shape}")
    
    # Test target generation
    print("\nTesting target generation:")
    numbers = torch.tensor([[5, 3], [10, 10], [2, 8], [7, 4]])
    targets = generate_auxiliary_targets(numbers)
    
    print(f"  Numbers: {numbers.tolist()}")
    print(f"  Comparison: {targets['comparison'].tolist()}")
    print(f"  Magnitude: {targets['magnitude'].tolist()}")
    print(f"  Parity: {targets['parity'].tolist()}")
    print(f"  Digit sum: {targets['digit_sum'].tolist()}")
    
    print("\nPASS: Auxiliary tasks ready!")
