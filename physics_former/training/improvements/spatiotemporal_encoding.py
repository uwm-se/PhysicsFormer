"""
Spatiotemporal Positional Encoding Module

Provides an inductive bias for object identity preservation by combining:
1. Temporal encoding (sinusoidal) - when in the sequence
2. Spatial encoding (learnable) - which object

Based on attention-physics correlation analysis showing:
- L0-H0 serves as implicit identity preserver (physics-agnostic, r≈0)
- Adding explicit spatial encoding should free L0-H0 capacity for physics

Expected Impact: Medium
Implementation Effort: Medium
Risk: Low
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class SpatioTemporalEncoding(nn.Module):
    """
    Combined spatial and temporal positional encoding for physics sequences.
    
    Features:
    - Temporal: Sinusoidal encoding (standard transformer) for timestep position
    - Spatial: Learnable per-object embedding for object identity
    - Learnable combination weights (alpha) to balance temporal vs spatial
    
    This provides explicit object identity information, reducing the burden
    on attention heads to learn identity preservation implicitly.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_objects: int = 20,
        max_timesteps: int = 256,
        temporal_ratio: float = 0.5,
        learnable_alpha: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model hidden dimension
            max_objects: Maximum number of objects in scene
            max_timesteps: Maximum sequence length
            temporal_ratio: Fraction of d_model for temporal encoding (rest for spatial)
            learnable_alpha: If True, learn combination weight; else use fixed 0.5
            dropout: Dropout rate for encoding
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_objects = max_objects
        self.max_timesteps = max_timesteps
        
        temporal_dim = int(d_model * temporal_ratio)
        spatial_dim = d_model - temporal_dim
        
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        
        temporal_pe = self._create_sinusoidal_encoding(max_timesteps, temporal_dim)
        self.register_buffer('temporal_pe', temporal_pe)
        
        self.spatial_pe = nn.Embedding(max_objects, spatial_dim)
        nn.init.normal_(self.spatial_pe.weight, mean=0, std=0.02)
        
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('alpha', torch.tensor(0.5))
        
        self.projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        return pe
    
    def forward(
        self,
        x: torch.Tensor,
        object_ids: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add spatiotemporal encoding to input embeddings.
        
        Args:
            x: [batch, seq_len, d_model] or [batch, seq_len, objects, d_model]
            object_ids: [batch, seq_len] or [batch, seq_len, objects] - which object each token represents
            timesteps: [batch, seq_len] or [batch, seq_len, objects] - which timestep each token is from
            
        Returns:
            Encoded tensor with same shape as input
        """
        is_4d = x.dim() == 4
        
        if is_4d:
            batch_size, seq_len, num_objects, d_model = x.shape
            
            if timesteps is None:
                timesteps = torch.arange(seq_len, device=x.device)
                timesteps = timesteps.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_objects)
            
            if object_ids is None:
                object_ids = torch.arange(num_objects, device=x.device)
                object_ids = object_ids.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            
            timesteps_clamped = timesteps.clamp(0, self.max_timesteps - 1)
            object_ids_clamped = object_ids.clamp(0, self.max_objects - 1)
            
            temporal = self.temporal_pe[timesteps_clamped]
            spatial = self.spatial_pe(object_ids_clamped)
            
            encoding = torch.cat([temporal, spatial], dim=-1)
            
        else:
            batch_size, seq_len, d_model = x.shape
            
            if timesteps is None:
                timesteps = torch.arange(seq_len, device=x.device)
                timesteps = timesteps.unsqueeze(0).expand(batch_size, -1)
            
            timesteps_clamped = timesteps.clamp(0, self.max_timesteps - 1)
            temporal = self.temporal_pe[timesteps_clamped]
            
            if object_ids is not None:
                object_ids_clamped = object_ids.clamp(0, self.max_objects - 1)
                spatial = self.spatial_pe(object_ids_clamped)
                encoding = torch.cat([temporal, spatial], dim=-1)
            else:
                zero_spatial = torch.zeros(
                    batch_size, seq_len, self.spatial_dim,
                    device=x.device, dtype=x.dtype
                )
                encoding = torch.cat([temporal, zero_spatial], dim=-1)
        
        encoding = self.projection(encoding)
        encoding = self.dropout(encoding)
        
        alpha = torch.sigmoid(self.alpha)
        output = self.layer_norm(x + alpha * encoding)
        
        return output
    
    def get_encoding_only(
        self,
        batch_size: int,
        seq_len: int,
        num_objects: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Get raw spatiotemporal encoding without input.
        
        Useful for visualization and analysis.
        
        Returns:
            encoding: [batch, seq_len, objects, d_model]
        """
        timesteps = torch.arange(seq_len, device=device)
        timesteps = timesteps.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_objects)
        
        object_ids = torch.arange(num_objects, device=device)
        object_ids = object_ids.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        temporal = self.temporal_pe[timesteps]
        spatial = self.spatial_pe(object_ids)
        
        encoding = torch.cat([temporal, spatial], dim=-1)
        encoding = self.projection(encoding)
        
        return encoding


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for physics sequences.
    
    Instead of absolute positions, encodes relative distances between:
    - Objects (spatial distance)
    - Timesteps (temporal distance)
    
    This is more physics-appropriate since physics laws are translation-invariant.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_relative_distance: int = 32,
        num_heads: int = 16
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_relative_distance = max_relative_distance
        self.num_heads = num_heads
        
        self.temporal_bias = nn.Embedding(
            2 * max_relative_distance + 1,
            num_heads
        )
        
        self.spatial_bias = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, num_heads)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.temporal_bias.weight, mean=0, std=0.02)
        for module in self.spatial_bias.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
    
    def compute_temporal_bias(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute temporal relative position bias.
        
        Returns:
            bias: [1, num_heads, seq_len, seq_len]
        """
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        relative_positions = relative_positions.clamp(
            -self.max_relative_distance,
            self.max_relative_distance
        )
        relative_positions = relative_positions + self.max_relative_distance
        
        bias = self.temporal_bias(relative_positions)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        
        return bias
    
    def compute_spatial_bias(
        self,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial relative position bias based on physical distances.
        
        Args:
            positions: [batch, objects, 3] - object positions
            
        Returns:
            bias: [batch, num_heads, objects, objects]
        """
        distances = torch.cdist(positions, positions)
        distances = distances.unsqueeze(-1)
        
        bias = self.spatial_bias(distances)
        bias = bias.permute(0, 3, 1, 2)
        
        return bias
    
    def forward(
        self,
        attention_scores: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Add relative position bias to attention scores.
        
        Args:
            attention_scores: [batch, heads, seq, seq] or [batch, heads, objects, objects]
            positions: [batch, objects, 3] - for spatial bias
            seq_len: Sequence length for temporal bias
            
        Returns:
            Modified attention scores with position bias added
        """
        if seq_len is not None:
            temporal_bias = self.compute_temporal_bias(seq_len, attention_scores.device)
            attention_scores = attention_scores + temporal_bias
        
        if positions is not None:
            spatial_bias = self.compute_spatial_bias(positions)
            attention_scores = attention_scores + spatial_bias
        
        return attention_scores


def create_spatiotemporal_encoding(
    d_model: int = 256,
    max_objects: int = 20,
    max_timesteps: int = 256,
    encoding_type: str = 'absolute'
) -> nn.Module:
    """
    Factory function to create spatiotemporal encoding.
    
    Args:
        d_model: Model hidden dimension
        max_objects: Maximum number of objects
        max_timesteps: Maximum sequence length
        encoding_type: 'absolute' or 'relative'
        
    Returns:
        Encoding module
    """
    if encoding_type == 'absolute':
        return SpatioTemporalEncoding(
            d_model=d_model,
            max_objects=max_objects,
            max_timesteps=max_timesteps
        )
    elif encoding_type == 'relative':
        return RelativePositionalEncoding(
            d_model=d_model,
            max_relative_distance=32,
            num_heads=16
        )
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
