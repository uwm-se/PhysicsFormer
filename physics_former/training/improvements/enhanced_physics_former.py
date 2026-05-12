"""
Enhanced PhysicsFormer with Physics-Informed Architecture

Integrates all three architectural improvements from the attention-physics correlation analysis:
1. Physics Auxiliary Loss - Explicit dynamics prediction objectives
2. Spatiotemporal Encoding - Object identity + temporal position
3. Factorized Attention - Separate spatial (object) and temporal attention

This module provides a drop-in replacement for FullPhysicsFormer with improved:
- Physics understanding (auxiliary loss reinforces dynamics)
- Object identity preservation (spatiotemporal encoding)
- Interpretability (factorized attention separates spatial vs temporal reasoning)
- Generalization (explicit inductive biases for physics)

Usage:
    from physics_former.training.improvements import EnhancedPhysicsFormer
    
    model = EnhancedPhysicsFormer(
        state_dim=35,
        hidden_dim=256,
        num_layers=6,
        num_heads=16,
        use_auxiliary_loss=True,
        use_spatiotemporal_encoding=True,
        use_factorized_attention=True,
        aux_loss_weight=0.1
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List

from .physics_auxiliary_loss import PhysicsAuxiliaryLoss
from .spatiotemporal_encoding import SpatioTemporalEncoding
from .factorized_attention import FactorizedPhysicsBlock, FactorizedPhysicsEncoder


class EnhancedStateEncoder(nn.Module):
    """
    State encoder with optional spatiotemporal encoding.
    
    Encodes object states and pairwise relationships, with explicit
    spatial (object identity) and temporal (timestep) position encoding.
    """
    
    def __init__(
        self,
        state_dim: int = 35,
        hidden_dim: int = 256,
        max_objects: int = 20,
        max_timesteps: int = 256,
        dropout: float = 0.1,
        use_spatiotemporal_encoding: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_spatiotemporal_encoding = use_spatiotemporal_encoding
        
        self.object_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.pairwise_encoder = nn.Sequential(
            nn.Linear(state_dim * 2 + 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        if use_spatiotemporal_encoding:
            self.spatiotemporal_encoding = SpatioTemporalEncoding(
                d_model=hidden_dim,
                max_objects=max_objects,
                max_timesteps=max_timesteps,
                dropout=dropout
            )
    
    def compute_pairwise_features(self, object_states: torch.Tensor) -> torch.Tensor:
        """Compute physics-relevant pairwise features."""
        positions = object_states[..., :3]
        velocities = object_states[..., 3:6]
        
        pos_diff = positions.unsqueeze(-2) - positions.unsqueeze(-3)
        vel_diff = velocities.unsqueeze(-2) - velocities.unsqueeze(-3)
        
        distance = torch.norm(pos_diff, dim=-1, keepdim=True)
        closing_vel = -torch.sum(pos_diff * vel_diff, dim=-1, keepdim=True) / (distance + 1e-6)
        
        pairwise_features = torch.cat([distance, closing_vel, vel_diff], dim=-1)
        return pairwise_features
    
    def forward(
        self,
        object_states: torch.Tensor,
        object_ids: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode object states with optional spatiotemporal encoding.
        
        Args:
            object_states: [batch, objects, features] or [batch, seq_len, objects, features]
            object_ids: Optional object identity indices
            timesteps: Optional timestep indices
            
        Returns:
            object_embeddings: Encoded object representations
            pairwise_embeddings: Encoded pairwise relationships
            pairwise_features: Raw pairwise physics features
        """
        is_sequence = object_states.dim() == 4
        
        if is_sequence:
            batch_size, seq_len, num_objects, state_dim = object_states.shape
            
            states_flat = object_states.view(batch_size * seq_len, num_objects, state_dim)
            object_embeddings_flat = self.object_encoder(states_flat)
            pairwise_features_flat = self.compute_pairwise_features(states_flat)
            
            obj_i = states_flat.unsqueeze(2).expand(-1, -1, num_objects, -1)
            obj_j = states_flat.unsqueeze(1).expand(-1, num_objects, -1, -1)
            pairwise_input = torch.cat([obj_i, obj_j, pairwise_features_flat], dim=-1)
            pairwise_embeddings_flat = self.pairwise_encoder(
                pairwise_input.view(-1, pairwise_input.size(-1))
            ).view(batch_size * seq_len, num_objects, num_objects, self.hidden_dim)
            
            object_embeddings = object_embeddings_flat.view(batch_size, seq_len, num_objects, self.hidden_dim)
            pairwise_embeddings = pairwise_embeddings_flat.view(batch_size, seq_len, num_objects, num_objects, self.hidden_dim)
            pairwise_features = pairwise_features_flat.view(batch_size, seq_len, num_objects, num_objects, -1)
            
            if self.use_spatiotemporal_encoding:
                object_embeddings = self.spatiotemporal_encoding(
                    object_embeddings, object_ids, timesteps
                )
        else:
            batch_size, num_objects, state_dim = object_states.shape
            
            object_embeddings = self.object_encoder(object_states)
            pairwise_features = self.compute_pairwise_features(object_states)
            
            obj_i = object_states.unsqueeze(2).expand(-1, -1, num_objects, -1)
            obj_j = object_states.unsqueeze(1).expand(-1, num_objects, -1, -1)
            pairwise_input = torch.cat([obj_i, obj_j, pairwise_features], dim=-1)
            pairwise_embeddings = self.pairwise_encoder(
                pairwise_input.view(-1, pairwise_input.size(-1))
            ).view(batch_size, num_objects, num_objects, self.hidden_dim)
        
        return object_embeddings, pairwise_embeddings, pairwise_features


class EnhancedPhysicsFormer(nn.Module):
    """
    PhysicsFormer with all three architectural improvements integrated.
    
    Improvements:
    1. Physics Auxiliary Loss: Adds explicit dynamics prediction objectives
    2. Spatiotemporal Encoding: Provides object identity and temporal position
    3. Factorized Attention: Separates spatial (object) and temporal reasoning
    
    These improvements are based on attention-physics correlation analysis showing:
    - L0-H8 serves as primary physics integrator (r=0.63 proximity)
    - L5-H4 exhibits inverse correlations (compositional reasoning)
    - Many heads in layers 3-5 appear underutilized
    """
    
    def __init__(
        self,
        state_dim: int = 35,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 16,
        ff_dim: int = 1024,
        max_objects: int = 20,
        max_timesteps: int = 256,
        num_schema_classes: int = 87,
        dropout: float = 0.1,
        use_auxiliary_loss: bool = True,
        use_spatiotemporal_encoding: bool = True,
        use_factorized_attention: bool = True,
        aux_loss_weight: float = 0.1
    ):
        """
        Args:
            state_dim: Dimension of object state vector
            hidden_dim: Model hidden dimension
            num_layers: Number of transformer/factorized layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            max_objects: Maximum number of objects
            max_timesteps: Maximum sequence length
            num_schema_classes: Number of physics schema classes
            dropout: Dropout rate
            use_auxiliary_loss: Enable physics auxiliary loss heads
            use_spatiotemporal_encoding: Enable spatiotemporal positional encoding
            use_factorized_attention: Use factorized (spatial+temporal) attention
            aux_loss_weight: Weight for auxiliary physics loss (λ)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.max_timesteps = max_timesteps
        self.use_auxiliary_loss = use_auxiliary_loss
        self.use_factorized_attention = use_factorized_attention
        self.aux_loss_weight = aux_loss_weight
        
        self.encoder = EnhancedStateEncoder(
            state_dim=state_dim * 2,
            hidden_dim=hidden_dim,
            max_objects=max_objects,
            max_timesteps=max_timesteps,
            dropout=dropout,
            use_spatiotemporal_encoding=use_spatiotemporal_encoding
        )
        
        if use_factorized_attention:
            self.transformer = FactorizedPhysicsEncoder(
                d_model=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                num_objects=max_objects,
                num_timesteps=max_timesteps,
                use_physics_bias=True
            )
        else:
            from ..models.physics_former_full import EnhancedTransformerLayer
            self.transformer_layers = nn.ModuleList([
                EnhancedTransformerLayer(hidden_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ])
        
        self.physics_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, state_dim)
        )
        
        self.schema_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_schema_classes)
        )
        
        self.counting_attention = nn.Linear(hidden_dim, 1)
        self.counting_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, max_objects + 1)
        )
        
        if use_auxiliary_loss:
            self.aux_heads = PhysicsAuxiliaryLoss(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                dropout=dropout
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Physics-informed weight initialization."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'physics_decoder' in name and '8' in name:
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode_physics(
        self,
        object_states: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Encode physical scene with optional attention analysis.
        
        Args:
            object_states: [batch, objects, features] or [batch, seq_len, objects, features]
            object_mask: [batch, objects] or [batch, seq_len, objects]
            return_attention: If True, return attention weights for analysis
            
        Returns:
            object_embeddings: Encoded representations
            attention_info: Dictionary with attention weights (if requested)
        """
        is_sequence = object_states.dim() == 4
        
        if is_sequence:
            batch_size, seq_len, num_objects, state_dim = object_states.shape
            
            velocity = object_states[:, 1:] - object_states[:, :-1]
            zero_velocity = torch.zeros(batch_size, 1, num_objects, state_dim,
                                       device=object_states.device, dtype=object_states.dtype)
            velocity_padded = torch.cat([zero_velocity, velocity], dim=1)
            augmented_states = torch.cat([object_states, velocity_padded], dim=-1)
        else:
            batch_size, num_objects, state_dim = object_states.shape
            zero_velocity = torch.zeros_like(object_states)
            augmented_states = torch.cat([object_states, zero_velocity], dim=-1)
        
        object_embeddings, pairwise_embeddings, pairwise_features = self.encoder(augmented_states)
        
        if self.use_factorized_attention:
            object_embeddings, attention_info = self.transformer(
                object_embeddings, pairwise_features, object_mask, return_attention
            )
        else:
            attention_info = None
            if is_sequence:
                batch_size, seq_len, num_objects, hidden_dim = object_embeddings.shape
                embeddings_flat = object_embeddings.view(batch_size * seq_len, num_objects, hidden_dim)
                pairwise_flat = pairwise_features.view(batch_size * seq_len, num_objects, num_objects, -1)
                
                if object_mask is not None and object_mask.dim() == 3:
                    mask_flat = object_mask.view(batch_size * seq_len, num_objects)
                else:
                    mask_flat = None
                
                for layer in self.transformer_layers:
                    embeddings_flat = layer(embeddings_flat, pairwise_flat, mask_flat)
                
                object_embeddings = embeddings_flat.view(batch_size, seq_len, num_objects, hidden_dim)
            else:
                for layer in self.transformer_layers:
                    object_embeddings = layer(object_embeddings, pairwise_features, object_mask)
        
        return object_embeddings, attention_info
    
    def forward_physics(
        self,
        object_states: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Physics prediction with optional auxiliary loss.
        
        Args:
            object_states: [batch, objects, features] or [batch, seq_len, objects, features]
            object_mask: [batch, objects] or [batch, seq_len, objects]
            targets: Optional targets for auxiliary loss computation
            return_attention: If True, return attention weights
            
        Returns:
            Dictionary with:
            - predicted_states: Next state predictions
            - schema_logits: Schema classification logits
            - object_embeddings: Encoded representations
            - aux_predictions: Auxiliary physics predictions (if enabled)
            - aux_loss: Auxiliary loss value (if targets provided)
            - attention_info: Attention weights (if requested)
        """
        is_sequence = object_states.dim() == 4
        
        object_embeddings, attention_info = self.encode_physics(
            object_states, object_mask, return_attention
        )
        
        if is_sequence:
            batch_size, seq_len, num_objects, hidden_dim = object_embeddings.shape
            
            if seq_len < 2:
                raise ValueError(f"Sequence too short: got {seq_len}, need at least 2")
            
            embeddings_for_prediction = object_embeddings[:, :-1]
            embeddings_flat = embeddings_for_prediction.contiguous().view(
                batch_size * (seq_len - 1), num_objects, hidden_dim
            )
            predicted_flat = self.physics_decoder(embeddings_flat)
            predicted_states = predicted_flat.view(batch_size, seq_len - 1, num_objects, -1)
            
            last_embeddings = object_embeddings[:, -1]
            if object_mask is not None and object_mask.dim() == 3:
                last_mask = object_mask[:, -1]
            else:
                last_mask = object_mask
        else:
            predicted_states = self.physics_decoder(object_embeddings)
            last_embeddings = object_embeddings
            last_mask = object_mask
        
        if last_mask is not None:
            mask_expanded = last_mask.unsqueeze(-1).float()
            pooled = (last_embeddings * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            pooled = last_embeddings.mean(dim=1)
        
        schema_logits = self.schema_classifier(pooled)
        
        outputs = {
            'predicted_states': predicted_states,
            'schema_logits': schema_logits,
            'object_embeddings': object_embeddings
        }
        
        if return_attention:
            outputs['attention_info'] = attention_info
        
        if self.use_auxiliary_loss:
            aux_predictions = self.aux_heads(object_embeddings, object_mask)
            outputs['aux_predictions'] = aux_predictions
            
            if targets is not None:
                aux_targets = self._extract_aux_targets(object_states, targets)
                aux_loss, aux_components = self.aux_heads.compute_loss(
                    aux_predictions, aux_targets, 
                    last_mask if object_mask is not None else None
                )
                outputs['aux_loss'] = aux_loss * self.aux_loss_weight
                outputs['aux_loss_components'] = aux_components
        
        return outputs
    
    def forward_counting(
        self,
        object_embeddings: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Count objects using attention-based pooling.
        
        Args:
            object_embeddings: [batch, objects, hidden] or [batch, seq_len, objects, hidden]
            object_mask: [batch, objects] or [batch, seq_len, objects]
            
        Returns:
            count_logits: [batch, max_objects + 1]
        """
        if object_embeddings.dim() == 4:
            object_embeddings = object_embeddings[:, -1]
            if object_mask is not None and object_mask.dim() == 3:
                object_mask = object_mask[:, -1]
        
        attention_scores = self.counting_attention(object_embeddings)
        
        if object_mask is not None:
            mask_bool = object_mask.bool() if object_mask.dtype != torch.bool else object_mask
            attention_scores = attention_scores.masked_fill(~mask_bool.unsqueeze(-1), -1e4)
        
        attention_weights = F.softmax(attention_scores, dim=1)
        
        if object_mask is not None:
            attention_weights = attention_weights * mask_bool.unsqueeze(-1).float()
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        pooled = (object_embeddings * attention_weights).sum(dim=1)
        count_logits = self.counting_head(pooled)
        
        return count_logits
    
    def _extract_aux_targets(
        self,
        object_states: torch.Tensor,
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Extract auxiliary targets from states and provided targets."""
        aux_targets = {}
        
        if 'next_states' in targets or 'target_states' in targets:
            next_states = targets.get('next_states', targets.get('target_states'))
            if next_states is not None:
                if next_states.dim() == 4:
                    aux_targets['next_positions'] = next_states[:, -1, :, :3]
                    aux_targets['next_velocities'] = next_states[:, -1, :, 3:6]
                else:
                    aux_targets['next_positions'] = next_states[:, :, :3]
                    aux_targets['next_velocities'] = next_states[:, :, 3:6]
        
        if object_states.dim() == 4 and object_states.shape[1] >= 3:
            v1 = object_states[:, -2, :, 3:6]
            v2 = object_states[:, -1, :, 3:6]
            aux_targets['accelerations'] = v2 - v1
        
        return aux_targets
    
    def forward(
        self,
        object_states: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        task: str = 'physics',
        targets: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass for all tasks.
        
        Args:
            object_states: Input object states
            object_mask: Object validity mask
            task: Task type ('physics', 'counting')
            targets: Optional targets for loss computation
            return_attention: If True, return attention weights
            
        Returns:
            Task-specific outputs dictionary
        """
        if task == 'physics':
            return self.forward_physics(object_states, object_mask, targets, return_attention)
        elif task == 'counting':
            object_embeddings, _ = self.encode_physics(object_states, object_mask)
            count_logits = self.forward_counting(object_embeddings, object_mask)
            return {'count_logits': count_logits, 'object_embeddings': object_embeddings}
        else:
            raise ValueError(f"Unknown task: {task}")


def create_enhanced_physics_former(
    state_dim: int = 35,
    hidden_dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 16,
    improvements: Optional[List[str]] = None,
    aux_loss_weight: float = 0.1
) -> EnhancedPhysicsFormer:
    """
    Factory function to create EnhancedPhysicsFormer with selected improvements.
    
    Args:
        state_dim: State dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        improvements: List of improvements to enable:
            - 'auxiliary_loss': Physics auxiliary loss heads
            - 'spatiotemporal': Spatiotemporal positional encoding
            - 'factorized': Factorized (spatial+temporal) attention
            If None, enables all improvements.
        aux_loss_weight: Weight for auxiliary loss
        
    Returns:
        EnhancedPhysicsFormer instance
    """
    if improvements is None:
        improvements = ['auxiliary_loss', 'spatiotemporal', 'factorized']
    
    return EnhancedPhysicsFormer(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_auxiliary_loss='auxiliary_loss' in improvements,
        use_spatiotemporal_encoding='spatiotemporal' in improvements,
        use_factorized_attention='factorized' in improvements,
        aux_loss_weight=aux_loss_weight
    )
