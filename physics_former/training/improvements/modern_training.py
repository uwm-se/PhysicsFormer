"""
Modern Training Improvements for PhysicsFormer

Implements state-of-the-art techniques from LLaMA, GPT-4, and recent research:

1. EMA (Exponential Moving Average) - Smoothed model weights for better generalization
2. Cosine LR Schedule with Warmup - Standard modern LR schedule
3. SwiGLU Activation - Gated activation from LLaMA/PaLM
4. Grouped Query Attention (GQA) - Memory-efficient attention
5. Pre-Norm Architecture - Better gradient flow
6. Gradient Checkpointing - Trade compute for memory
7. Label Smoothing - Prevent overconfident predictions
8. Data Augmentation - Object permutation, noise injection
9. Multi-Scale Temporal Loss - Penalize errors at multiple horizons
10. Contrastive Learning - Learn physics-aware representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, Dict, List, Tuple
from torch.optim.lr_scheduler import LambdaLR


class EMAModel:
    """
    Exponential Moving Average of model weights.
    
    Maintains a smoothed copy of model weights that often generalizes better.
    Used in DALL-E, Stable Diffusion, and many SOTA models.
    
    Usage:
        ema = EMAModel(model, decay=0.999)
        # During training:
        ema.update(model)
        # For evaluation:
        ema.apply_shadow(model)  # Use EMA weights
        ema.restore(model)       # Restore original weights
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update EMA weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1.0 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model (backup original first)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self) -> Dict:
        return {'shadow': self.shadow, 'decay': self.decay}
    
    def load_state_dict(self, state_dict: Dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1
):
    """
    Cosine learning rate schedule with linear warmup.
    
    Standard in modern transformers (GPT, LLaMA, etc.)
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Steps for linear warmup (typically 5-10% of total)
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles (0.5 = half cycle, decay to min)
        min_lr_ratio: Minimum LR as ratio of initial LR (0.1 = decay to 10%)
    """
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # Scale to [min_lr_ratio, 1.0]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


class SwiGLU(nn.Module):
    """
    SwiGLU activation from LLaMA/PaLM.
    
    SwiGLU(x) = Swish(xW) ⊙ (xV)
    
    More expressive than ReLU, used in most modern LLMs.
    Note: Requires 3x hidden dim instead of 4x for same param count.
    """
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.w3 = nn.Linear(in_features, hidden_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) from LLaMA 2.
    
    Uses fewer K/V heads than Q heads, reducing memory while maintaining quality.
    Example: 16 Q heads, 4 K/V heads = 4x memory reduction for K/V cache.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(output)


class PreNormTransformerLayer(nn.Module):
    """
    Pre-Norm Transformer Layer (used in GPT, LLaMA).
    
    x = x + Attention(Norm(x))
    x = x + FFN(Norm(x))
    
    Better gradient flow than Post-Norm, especially for deep networks.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_swiglu: bool = True
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        if use_swiglu:
            swiglu_hidden = int(ff_dim * 2 / 3)
            self.ff = SwiGLU(hidden_dim, swiglu_hidden, hidden_dim)
        else:
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, hidden_dim)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        
        x = x + self.dropout(self.ff(self.norm2(x)))
        
        return x


def enable_gradient_checkpointing(model: nn.Module) -> int:
    """
    Enable gradient checkpointing on transformer layers.
    
    Trades compute for memory - recomputes activations during backward pass.
    Can reduce memory by 50-70% at cost of ~20% slower training.
    
    Returns number of layers modified.
    """
    from torch.utils.checkpoint import checkpoint
    
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'transformer_layers'):
            original_forward = module.forward
            
            def checkpointed_forward(self, *args, **kwargs):
                def custom_forward(*inputs):
                    return original_forward(*inputs)
                return checkpoint(custom_forward, *args, use_reentrant=False, **kwargs)
            
            module.forward = checkpointed_forward.__get__(module, type(module))
            count += 1
    
    return count


class LabelSmoothingMSE(nn.Module):
    """
    Label smoothing for continuous targets (physics predictions).
    
    Instead of hard MSE targets, adds small noise to prevent overconfidence.
    Helps with generalization on continuous prediction tasks.
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.training and self.smoothing > 0:
            noise = torch.randn_like(target) * self.smoothing * target.std()
            target = target + noise
        
        return F.mse_loss(pred, target)


class PhysicsDataAugmentation:
    """
    Data augmentation for physics sequences.
    
    Augmentations that preserve physics validity:
    1. Object permutation (order invariance)
    2. Small noise injection
    3. Time reversal (for reversible physics)
    """
    
    def __init__(
        self,
        permute_prob: float = 0.3,
        noise_std: float = 0.01,
        time_reverse_prob: float = 0.1
    ):
        self.permute_prob = permute_prob
        self.noise_std = noise_std
        self.time_reverse_prob = time_reverse_prob
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'object_states' not in batch:
            return batch
        
        states = batch['object_states']
        batch_size = states.shape[0]
        
        if torch.rand(1).item() < self.permute_prob:
            for i in range(batch_size):
                num_objects = states.shape[2] if states.dim() == 4 else states.shape[1]
                perm = torch.randperm(num_objects)
                if states.dim() == 4:
                    states[i] = states[i, :, perm, :]
                else:
                    states[i] = states[i, perm, :]
        
        if self.noise_std > 0:
            noise = torch.randn_like(states) * self.noise_std
            noise[..., 6:10] = 0
            states = states + noise
        
        if torch.rand(1).item() < self.time_reverse_prob:
            if states.dim() == 4:
                states = states.flip(1)
                states[..., 3:6] = -states[..., 3:6]
                if 'next_states' in batch:
                    batch['next_states'] = batch['next_states'].flip(1)
                    batch['next_states'][..., 3:6] = -batch['next_states'][..., 3:6]
        
        batch['object_states'] = states
        return batch


class MultiScaleTemporalLoss(nn.Module):
    """
    Multi-scale temporal loss for physics prediction.
    
    Penalizes errors at multiple time horizons:
    L = L(t+1) + w1*L(t+5) + w2*L(t+10) + ...
    
    Encourages learning both short-term and long-term dynamics.
    """
    
    def __init__(
        self,
        horizons: List[int] = [1, 5, 10],
        weights: List[float] = [1.0, 0.5, 0.25]
    ):
        super().__init__()
        self.horizons = horizons
        self.weights = weights
    
    def forward(
        self,
        pred_sequence: torch.Tensor,
        target_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_sequence: [batch, seq_len, objects, features]
            target_sequence: [batch, seq_len, objects, features]
        """
        total_loss = 0.0
        seq_len = pred_sequence.shape[1]
        
        for horizon, weight in zip(self.horizons, self.weights):
            if horizon < seq_len:
                pred_at_horizon = pred_sequence[:, horizon-1::horizon, :, :]
                target_at_horizon = target_sequence[:, horizon-1::horizon, :, :]
                
                min_len = min(pred_at_horizon.shape[1], target_at_horizon.shape[1])
                if min_len > 0:
                    loss = F.mse_loss(
                        pred_at_horizon[:, :min_len],
                        target_at_horizon[:, :min_len]
                    )
                    total_loss = total_loss + weight * loss
        
        return total_loss


class PhysicsContrastiveLoss(nn.Module):
    """
    Contrastive learning for physics representations.
    
    Learn to distinguish:
    - Same schema, different params = positive pairs (similar physics)
    - Different schema = negative pairs (different physics)
    
    Helps learn physics-aware embeddings.
    """
    
    def __init__(self, temperature: float = 0.07, hidden_dim: int = 256):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        schema_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, hidden_dim] - Scene embeddings
            schema_ids: [batch] - Schema IDs for each sample
        """
        projected = F.normalize(self.projection(embeddings), dim=-1)
        
        similarity = torch.matmul(projected, projected.T) / self.temperature
        
        same_schema = schema_ids.unsqueeze(0) == schema_ids.unsqueeze(1)
        same_schema.fill_diagonal_(False)
        
        labels = same_schema.float()
        
        loss = F.binary_cross_entropy_with_logits(similarity, labels)
        
        return loss


def apply_all_improvements(
    model: nn.Module,
    optimizer,
    config,
    num_training_steps: int,
    verbose: bool = True
) -> Dict:
    """
    Apply all modern training improvements.
    
    Returns dict with:
        - ema: EMAModel instance
        - scheduler: LR scheduler
        - augmentation: Data augmentation
        - multi_scale_loss: Multi-scale temporal loss
        - contrastive_loss: Contrastive loss module
    """
    improvements = {}
    applied = []
    
    if getattr(config, 'use_ema', True):
        ema_decay = getattr(config, 'ema_decay', 0.999)
        improvements['ema'] = EMAModel(model, decay=ema_decay)
        applied.append(f"EMA (decay={ema_decay})")
    
    if getattr(config, 'use_cosine_schedule', True):
        warmup_ratio = getattr(config, 'warmup_ratio', 0.05)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        min_lr_ratio = getattr(config, 'min_lr_ratio', 0.1)
        
        improvements['scheduler'] = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio
        )
        applied.append(f"Cosine LR (warmup={num_warmup_steps}, min_ratio={min_lr_ratio})")
    
    if getattr(config, 'use_gradient_checkpointing', False):
        count = enable_gradient_checkpointing(model)
        if count > 0:
            applied.append(f"Gradient Checkpointing ({count} modules)")
    
    if getattr(config, 'use_data_augmentation', True):
        improvements['augmentation'] = PhysicsDataAugmentation(
            permute_prob=getattr(config, 'augment_permute_prob', 0.3),
            noise_std=getattr(config, 'augment_noise_std', 0.01),
            time_reverse_prob=getattr(config, 'augment_time_reverse_prob', 0.1)
        )
        applied.append("Data Augmentation")
    
    if getattr(config, 'use_multi_scale_loss', True):
        improvements['multi_scale_loss'] = MultiScaleTemporalLoss(
            horizons=getattr(config, 'temporal_horizons', [1, 5, 10]),
            weights=getattr(config, 'temporal_weights', [1.0, 0.5, 0.25])
        )
        applied.append("Multi-Scale Temporal Loss")
    
    if getattr(config, 'use_contrastive_loss', False):
        hidden_dim = getattr(config, 'hidden_dim', 512)
        improvements['contrastive_loss'] = PhysicsContrastiveLoss(
            temperature=getattr(config, 'contrastive_temperature', 0.07),
            hidden_dim=hidden_dim
        ).to(next(model.parameters()).device)
        applied.append("Contrastive Learning")
    
    if getattr(config, 'use_label_smoothing', True):
        improvements['label_smoothing'] = LabelSmoothingMSE(
            smoothing=getattr(config, 'label_smoothing', 0.1)
        )
        applied.append(f"Label Smoothing ({getattr(config, 'label_smoothing', 0.1)})")
    
    if verbose:
        if applied:
            print(f"[MODERN TRAINING] Applied: {', '.join(applied)}")
        else:
            print("[MODERN TRAINING] No improvements applied")
    
    return improvements
