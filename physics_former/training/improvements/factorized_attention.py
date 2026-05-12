"""
Factorized Attention Module

Separates object-level and temporal reasoning into distinct attention operations,
similar to TimeSformer/ViViT architectures for video understanding.

Based on attention-physics correlation analysis showing:
- Early layers (L0) handle proximity/force correlations
- Deep layers (L5) show inverse correlations (compositional reasoning)
- Factorization should make these roles more explicit and interpretable

Expected Impact: High
Implementation Effort: High
Risk: Medium

Benefits:
- Explicit separation of "how objects interact" vs "how objects evolve"
- Improved interpretability (can analyze spatial vs temporal heads separately)
- Potentially better generalization to different object counts or sequence lengths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SpatialAttention(nn.Module):
    """
    Attention over objects at each timestep.
    
    Captures: How objects relate to each other (proximity, forces, collisions)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_physics_bias: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_physics_bias = use_physics_bias
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        if use_physics_bias:
            self.physics_bias_net = nn.Sequential(
                nn.Linear(5, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, num_heads),
                nn.Tanh()
            )
        
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)
        
        if self.use_physics_bias:
            for module in self.physics_bias_net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        pairwise_features: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Spatial attention over objects.
        
        Args:
            x: [batch, objects, d_model]
            pairwise_features: [batch, objects, objects, 5] - physics features
            object_mask: [batch, objects]
            
        Returns:
            output: [batch, objects, d_model]
            attention_weights: [batch, heads, objects, objects]
        """
        batch_size, num_objects, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, num_objects, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, num_objects, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, num_objects, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if self.use_physics_bias and pairwise_features is not None:
            pairwise_flat = pairwise_features.view(-1, pairwise_features.size(-1))
            physics_bias = self.physics_bias_net(pairwise_flat)
            physics_bias = physics_bias.view(batch_size, num_objects, num_objects, self.num_heads)
            physics_bias = physics_bias.permute(0, 3, 1, 2)
            attn_scores = attn_scores + physics_bias
        
        if object_mask is not None:
            # Handle 2D or 3D object_mask
            if object_mask.dim() == 3:
                object_mask = object_mask[:, 0, :]
            mask = object_mask.unsqueeze(1).unsqueeze(2).bool()
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        attn_scores = torch.clamp(attn_scores, min=-50, max=50)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, num_objects, self.d_model)
        output = self.out_proj(output)
        
        return output, attn_weights


class TemporalAttention(nn.Module):
    """
    Attention over timesteps for each object.
    
    Captures: How each object evolves over time (trajectories, momentum)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        use_causal_mask: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_causal_mask = use_causal_mask
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.temporal_bias = nn.Embedding(2 * max_seq_len - 1, num_heads)
        nn.init.normal_(self.temporal_bias.weight, mean=0, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)
    
    def _get_relative_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute relative position indices."""
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions + self.max_seq_len - 1
        relative_positions = relative_positions.clamp(0, 2 * self.max_seq_len - 2)
        return relative_positions
    
    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Temporal attention over timesteps.
        
        Args:
            x: [batch, timesteps, d_model]
            temporal_mask: [batch, timesteps] - which timesteps are valid
            
        Returns:
            output: [batch, timesteps, d_model]
            attention_weights: [batch, heads, timesteps, timesteps]
        """
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        relative_positions = self._get_relative_positions(seq_len, x.device)
        temporal_bias = self.temporal_bias(relative_positions)
        temporal_bias = temporal_bias.permute(2, 0, 1).unsqueeze(0)
        attn_scores = attn_scores + temporal_bias
        
        if self.use_causal_mask:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if temporal_mask is not None:
            mask = temporal_mask.unsqueeze(1).unsqueeze(2).bool()
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        attn_scores = torch.clamp(attn_scores, min=-50, max=50)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output, attn_weights


class FactorizedPhysicsBlock(nn.Module):
    """
    Factorized attention block for physics sequences.
    
    Applies spatial attention (object interactions) and temporal attention
    (object evolution) separately, then combines with feed-forward network.
    
    Architecture:
    1. Spatial attention: [batch*T, O, D] -> how objects interact at each timestep
    2. Temporal attention: [batch*O, T, D] -> how each object evolves over time
    3. Feed-forward: Standard transformer FFN
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 16,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        num_objects: int = 20,
        num_timesteps: int = 256,
        use_physics_bias: bool = True,
        use_causal_mask: bool = False,
        attention_order: str = 'spatial_first'
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Total attention heads (split between spatial and temporal)
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            num_objects: Maximum number of objects
            num_timesteps: Maximum sequence length
            use_physics_bias: Add physics-based attention bias to spatial attention
            use_causal_mask: Use causal mask for temporal attention
            attention_order: 'spatial_first' or 'temporal_first'
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_objects = num_objects
        self.num_timesteps = num_timesteps
        self.attention_order = attention_order
        
        spatial_heads = num_heads // 2
        temporal_heads = num_heads - spatial_heads
        
        self.spatial_attn = SpatialAttention(
            d_model=d_model,
            num_heads=spatial_heads,
            dropout=dropout,
            use_physics_bias=use_physics_bias
        )
        self.spatial_norm = nn.LayerNorm(d_model)
        
        self.temporal_attn = TemporalAttention(
            d_model=d_model,
            num_heads=temporal_heads,
            dropout=dropout,
            max_seq_len=num_timesteps,
            use_causal_mask=use_causal_mask
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        pairwise_features: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with factorized attention.
        
        Args:
            x: [batch, timesteps, objects, d_model] or [batch, timesteps * objects, d_model]
            pairwise_features: [batch, timesteps, objects, objects, 5] or None
            object_mask: [batch, timesteps, objects] or [batch, objects]
            return_attention: If True, return attention weights for analysis
            
        Returns:
            output: Same shape as input
            attention_info: Dictionary with spatial and temporal attention weights (if requested)
        """
        is_4d = x.dim() == 4
        
        if is_4d:
            B, T, O, D = x.shape
        else:
            B, S, D = x.shape
            T = self.num_timesteps
            O = S // T
            x = x.view(B, T, O, D)
            if pairwise_features is not None and pairwise_features.dim() == 4:
                pairwise_features = pairwise_features.unsqueeze(1).expand(-1, T, -1, -1, -1)
        
        attention_info = {} if return_attention else None
        
        if self.attention_order == 'spatial_first':
            x, attention_info = self._spatial_then_temporal(
                x, pairwise_features, object_mask, return_attention
            )
        else:
            x, attention_info = self._temporal_then_spatial(
                x, pairwise_features, object_mask, return_attention
            )
        
        x_flat = x.view(B * T * O, D)
        ffn_out = self.ffn(x_flat)
        x = self.ffn_norm(x.view(B * T * O, D) + ffn_out).view(B, T, O, D)
        
        if not is_4d:
            x = x.view(B, T * O, D)
        
        return x, attention_info
    
    def _spatial_then_temporal(
        self,
        x: torch.Tensor,
        pairwise_features: Optional[torch.Tensor],
        object_mask: Optional[torch.Tensor],
        return_attention: bool
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Apply spatial attention first, then temporal."""
        B, T, O, D = x.shape
        attention_info = {} if return_attention else None
        
        x_spatial = x.view(B * T, O, D)
        
        if pairwise_features is not None:
            pf_spatial = pairwise_features.view(B * T, O, O, -1)
        else:
            pf_spatial = None
        
        if object_mask is not None:
            if object_mask.dim() == 3:
                om_spatial = object_mask.view(B * T, O)
            else:
                om_spatial = object_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, O)
        else:
            om_spatial = None
        
        spatial_out, spatial_attn = self.spatial_attn(x_spatial, pf_spatial, om_spatial)
        x = self.spatial_norm(x.view(B * T, O, D) + self.dropout(spatial_out)).view(B, T, O, D)
        
        if return_attention:
            attention_info['spatial'] = spatial_attn.view(B, T, -1, O, O)
        
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * O, T, D)
        
        temporal_out, temporal_attn = self.temporal_attn(x_temporal)
        temporal_out = temporal_out.view(B, O, T, D).permute(0, 2, 1, 3)
        x = self.temporal_norm(x + self.dropout(temporal_out))
        
        if return_attention:
            attention_info['temporal'] = temporal_attn.view(B, O, -1, T, T)
        
        return x, attention_info
    
    def _temporal_then_spatial(
        self,
        x: torch.Tensor,
        pairwise_features: Optional[torch.Tensor],
        object_mask: Optional[torch.Tensor],
        return_attention: bool
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Apply temporal attention first, then spatial."""
        B, T, O, D = x.shape
        attention_info = {} if return_attention else None
        
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * O, T, D)
        
        temporal_out, temporal_attn = self.temporal_attn(x_temporal)
        temporal_out = temporal_out.view(B, O, T, D).permute(0, 2, 1, 3)
        x = self.temporal_norm(x + self.dropout(temporal_out))
        
        if return_attention:
            attention_info['temporal'] = temporal_attn.view(B, O, -1, T, T)
        
        x_spatial = x.view(B * T, O, D)
        
        if pairwise_features is not None:
            pf_spatial = pairwise_features.view(B * T, O, O, -1)
        else:
            pf_spatial = None
        
        if object_mask is not None:
            if object_mask.dim() == 3:
                om_spatial = object_mask.view(B * T, O)
            else:
                om_spatial = object_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, O)
        else:
            om_spatial = None
        
        spatial_out, spatial_attn = self.spatial_attn(x_spatial, pf_spatial, om_spatial)
        x = self.spatial_norm(x.view(B * T, O, D) + self.dropout(spatial_out)).view(B, T, O, D)
        
        if return_attention:
            attention_info['spatial'] = spatial_attn.view(B, T, -1, O, O)
        
        return x, attention_info


class FactorizedPhysicsEncoder(nn.Module):
    """
    Full encoder using factorized attention blocks.
    
    Replaces the standard transformer encoder with factorized attention
    for improved physics understanding and interpretability.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 16,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        num_objects: int = 20,
        num_timesteps: int = 256,
        use_physics_bias: bool = True
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            FactorizedPhysicsBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                num_objects=num_objects,
                num_timesteps=num_timesteps,
                use_physics_bias=use_physics_bias,
                attention_order='spatial_first' if i % 2 == 0 else 'temporal_first'
            )
            for i in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        pairwise_features: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through all factorized layers.
        
        Args:
            x: [batch, timesteps, objects, d_model]
            pairwise_features: [batch, timesteps, objects, objects, 5]
            object_mask: [batch, timesteps, objects] or [batch, objects]
            return_attention: If True, return attention weights from all layers
            
        Returns:
            output: [batch, timesteps, objects, d_model]
            all_attention: List of attention info dicts (if requested)
        """
        all_attention = [] if return_attention else None
        
        for layer in self.layers:
            x, attn_info = layer(x, pairwise_features, object_mask, return_attention)
            if return_attention:
                all_attention.append(attn_info)
        
        x = self.final_norm(x)
        
        return x, all_attention


def create_factorized_encoder(
    d_model: int = 256,
    num_layers: int = 6,
    num_heads: int = 16,
    ff_dim: int = 1024,
    dropout: float = 0.1,
    num_objects: int = 20,
    num_timesteps: int = 256
) -> FactorizedPhysicsEncoder:
    """
    Factory function to create a factorized physics encoder.
    
    Args:
        d_model: Model dimension
        num_layers: Number of factorized blocks
        num_heads: Total attention heads per block
        ff_dim: Feed-forward dimension
        dropout: Dropout rate
        num_objects: Maximum number of objects
        num_timesteps: Maximum sequence length
        
    Returns:
        FactorizedPhysicsEncoder instance
    """
    return FactorizedPhysicsEncoder(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        num_objects=num_objects,
        num_timesteps=num_timesteps
    )
