"""
Physics Auxiliary Loss Module

Reinforces physics understanding by adding explicit dynamics prediction objectives
alongside the primary QA/prediction task.

Based on attention-physics correlation analysis showing:
- L0-H8 serves as primary physics integrator (r=0.63 proximity, r=0.47 force)
- Many heads in layers 3-5 appear underutilized
- Adding auxiliary physics loss should improve middle layer utilization

Expected Impact: Medium-High
Implementation Effort: Low
Risk: Low
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PhysicsAuxiliaryLoss(nn.Module):
    """
    Auxiliary physics prediction heads for multi-task learning.
    
    Adds explicit supervision for:
    1. Next-step position prediction (dynamics)
    2. Velocity prediction (motion understanding)
    3. Acceleration prediction (force understanding)
    4. Collision prediction (interaction detection)
    
    These auxiliary tasks reinforce physics understanding in the shared encoder.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        state_dim: int = 35,
        num_aux_tasks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self.acceleration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self.collision_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        object_embeddings: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary physics predictions.
        
        Args:
            object_embeddings: [batch, objects, hidden] or [batch, seq_len, objects, hidden]
            object_mask: [batch, objects] or [batch, seq_len, objects]
            
        Returns:
            Dictionary with auxiliary predictions:
            - position_pred: [batch, objects, 3]
            - velocity_pred: [batch, objects, 3]
            - acceleration_pred: [batch, objects, 3]
            - collision_pred: [batch, objects, objects] (pairwise collision probability)
        """
        is_sequence = object_embeddings.dim() == 4
        
        if is_sequence:
            batch_size, seq_len, num_objects, hidden_dim = object_embeddings.shape
            embeddings = object_embeddings[:, -1]
        else:
            batch_size, num_objects, hidden_dim = object_embeddings.shape
            embeddings = object_embeddings
        
        position_pred = self.position_head(embeddings)
        velocity_pred = self.velocity_head(embeddings)
        acceleration_pred = self.acceleration_head(embeddings)
        
        obj_i = embeddings.unsqueeze(2).expand(-1, -1, num_objects, -1)
        obj_j = embeddings.unsqueeze(1).expand(-1, num_objects, -1, -1)
        pairwise = torch.cat([obj_i, obj_j], dim=-1)
        
        collision_pred = self.collision_head(pairwise).squeeze(-1)
        
        return {
            'position_pred': position_pred,
            'velocity_pred': velocity_pred,
            'acceleration_pred': acceleration_pred,
            'collision_pred': collision_pred
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        object_mask: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute auxiliary physics loss.
        
        Args:
            predictions: Dictionary from forward()
            targets: Dictionary with ground truth:
                - next_positions: [batch, objects, 3]
                - next_velocities: [batch, objects, 3]
                - accelerations: [batch, objects, 3]
                - collision_matrix: [batch, objects, objects] (binary)
            object_mask: [batch, objects]
            loss_weights: Optional weights for each loss component
                
        Returns:
            total_loss: Scalar tensor
            loss_components: Dictionary with individual loss values
        """
        if loss_weights is None:
            loss_weights = {
                'position': 1.0,
                'velocity': 0.5,
                'acceleration': 0.3,
                'collision': 0.2
            }
        
        loss_components = {}
        
        if object_mask is not None:
            # Handle 2D or 3D object_mask
            if object_mask.dim() == 3:
                object_mask = object_mask[:, 0, :]
            mask = object_mask.unsqueeze(-1).float()
            num_active = mask.sum().clamp(min=1)
        else:
            mask = torch.ones_like(predictions['position_pred'][..., :1])
            num_active = mask.numel()
        
        if 'next_positions' in targets:
            pos_diff = (predictions['position_pred'] - targets['next_positions']) ** 2
            pos_loss = (pos_diff * mask).sum() / num_active
            loss_components['position'] = pos_loss
        
        if 'next_velocities' in targets:
            vel_diff = (predictions['velocity_pred'] - targets['next_velocities']) ** 2
            vel_loss = (vel_diff * mask).sum() / num_active
            loss_components['velocity'] = vel_loss
        
        if 'accelerations' in targets:
            acc_diff = (predictions['acceleration_pred'] - targets['accelerations']) ** 2
            acc_loss = (acc_diff * mask).sum() / num_active
            loss_components['acceleration'] = acc_loss
        
        if 'collision_matrix' in targets:
            collision_targets = targets['collision_matrix'].float()
            collision_loss = F.binary_cross_entropy(
                predictions['collision_pred'],
                collision_targets,
                reduction='mean'
            )
            loss_components['collision'] = collision_loss
        
        total_loss = sum(
            loss_weights.get(name, 1.0) * loss 
            for name, loss in loss_components.items()
        )
        
        return total_loss, loss_components


class PhysicsFormerWithAuxLoss(nn.Module):
    """
    Wrapper that adds auxiliary physics loss to any PhysicsFormer model.
    
    Usage:
        base_model = FullPhysicsFormer(...)
        model = PhysicsFormerWithAuxLoss(base_model, aux_loss_weight=0.1)
        
        # Training
        outputs = model(batch)
        total_loss = outputs['primary_loss'] + outputs['aux_loss']
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        aux_loss_weight: float = 0.1,
        hidden_dim: int = 256,
        state_dim: int = 35
    ):
        super().__init__()
        
        self.base_model = base_model
        self.aux_loss_weight = aux_loss_weight
        
        self.aux_heads = PhysicsAuxiliaryLoss(
            hidden_dim=hidden_dim,
            state_dim=state_dim
        )
    
    def forward(
        self,
        object_states: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        task: str = 'physics'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with auxiliary physics predictions.
        
        Args:
            object_states: Input states
            object_mask: Object mask
            targets: Optional targets for loss computation
            task: Task type ('physics', 'counting', etc.)
            
        Returns:
            Dictionary with outputs and losses
        """
        if task == 'physics':
            predicted_states, schema_logits, object_embeddings = self.base_model.forward_physics(
                object_states, object_mask
            )
            
            aux_predictions = self.aux_heads(object_embeddings, object_mask)
            
            outputs = {
                'predicted_states': predicted_states,
                'schema_logits': schema_logits,
                'object_embeddings': object_embeddings,
                'aux_predictions': aux_predictions
            }
            
            if targets is not None:
                aux_targets = self._extract_aux_targets(object_states, targets)
                aux_loss, aux_components = self.aux_heads.compute_loss(
                    aux_predictions, aux_targets, object_mask
                )
                outputs['aux_loss'] = aux_loss * self.aux_loss_weight
                outputs['aux_loss_components'] = aux_components
            
            return outputs
        else:
            return self.base_model(object_states, object_mask, task=task)
    
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


def create_physics_aux_loss(
    base_model: nn.Module,
    aux_loss_weight: float = 0.1,
    hidden_dim: int = 256,
    state_dim: int = 35
) -> PhysicsFormerWithAuxLoss:
    """
    Factory function to wrap a model with auxiliary physics loss.
    
    Args:
        base_model: The base PhysicsFormer model
        aux_loss_weight: Weight for auxiliary loss (λ in paper, typically 0.1-0.5)
        hidden_dim: Hidden dimension of the model
        state_dim: State dimension
        
    Returns:
        Wrapped model with auxiliary physics loss
    """
    return PhysicsFormerWithAuxLoss(
        base_model=base_model,
        aux_loss_weight=aux_loss_weight,
        hidden_dim=hidden_dim,
        state_dim=state_dim
    )
