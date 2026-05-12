"""
Causal Training for PhysicsFormer

Adds causal intervention training to enable true counterfactual reasoning.
The key insight: Train with object dropout and intervention-aware loss so
the model learns CAUSAL relationships, not just correlations.

Training Strategy:
1. Object Dropout: Randomly remove objects during training
2. Intervention Loss: Penalize when unrelated objects are affected
3. Causal Masking: Track which objects SHOULD be affected by removal
4. Contrastive Learning: Compare original vs counterfactual trajectories

This addresses the CLEVRER-Humans gap (98.5% → 54%) by teaching the model
to distinguish correlation from causation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class CausalTrainingConfig:
    dropout_prob: float = 0.3
    intervention_weight: float = 1.0
    contrastive_weight: float = 0.5
    causal_margin: float = 0.1
    collision_threshold: float = 2.0
    velocity_threshold: float = 0.1


class CausalGraphBuilder:
    """
    Builds causal graphs from physics states to determine which objects
    causally affect which other objects.
    
    An object A causally affects object B if:
    1. A and B collide (or come within collision distance)
    2. A's trajectory intersects B's future position
    3. A transfers momentum to B
    """
    
    POSITION_IDX = slice(0, 3)
    VELOCITY_IDX = slice(3, 6)
    MASS_IDX = 10
    RADIUS_IDX = 13
    
    def __init__(self, config: CausalTrainingConfig):
        self.config = config
    
    def build_causal_graph(
        self,
        states: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Build causal adjacency matrix from physics states (VECTORIZED for GPU).
        
        Args:
            states: [batch, seq_len, num_objects, state_dim]
            mask: [batch, num_objects]
        
        Returns:
            causal_graph: [batch, num_objects, num_objects]
                          causal_graph[b, i, j] = 1 if object i causally affects object j
        """
        batch_size, seq_len, num_objects, state_dim = states.shape
        device = states.device
        
        if mask.dim() == 2:
            obj_mask = mask[:, :num_objects]
        elif mask.dim() == 3:
            obj_mask = mask[:, 0, :num_objects]
        else:
            obj_mask = mask.view(batch_size, -1)[:, :num_objects]
        
        positions = states[:, :, :, self.POSITION_IDX]
        
        if state_dim > 5:
            velocities = states[:, :, :, self.VELOCITY_IDX]
        else:
            velocities = positions[:, 1:] - positions[:, :-1]
            velocities = torch.cat([velocities, velocities[:, -1:]], dim=1)
        
        if state_dim > self.RADIUS_IDX:
            radii = states[:, 0, :, self.RADIUS_IDX]
        else:
            radii = torch.full((batch_size, num_objects), 0.5, device=device)
        
        pos_i = positions.unsqueeze(3)
        pos_j = positions.unsqueeze(2)
        
        diff = pos_i - pos_j
        distances = torch.norm(diff, dim=-1)
        
        radii_i = radii.unsqueeze(2)
        radii_j = radii.unsqueeze(1)
        collision_dist = radii_i + radii_j + self.config.collision_threshold
        
        collisions = distances < collision_dist.unsqueeze(1)
        
        collisions_any_time = collisions.any(dim=1)
        
        vel_magnitude = torch.norm(velocities, dim=-1)
        fast_objects = vel_magnitude > self.config.velocity_threshold
        
        future_positions = positions + velocities * 5
        future_pos_i = future_positions.unsqueeze(3)
        future_diff = future_pos_i - pos_j
        future_distances = torch.norm(future_diff, dim=-1)
        future_collisions = future_distances < collision_dist.unsqueeze(1)
        
        future_collisions_masked = future_collisions & fast_objects.unsqueeze(-1)
        future_any_time = future_collisions_masked.any(dim=1)
        
        causal_graph = (collisions_any_time | future_any_time).float()
        
        eye = torch.eye(num_objects, device=device).unsqueeze(0)
        causal_graph = causal_graph * (1 - eye)
        
        valid_mask = obj_mask.unsqueeze(2) * obj_mask.unsqueeze(1)
        causal_graph = causal_graph * valid_mask
        
        causal_graph = torch.max(causal_graph, causal_graph.transpose(1, 2))
        
        return causal_graph
    
    def get_affected_objects(
        self,
        causal_graph: torch.Tensor,
        removed_object: int
    ) -> torch.Tensor:
        """
        Get mask of objects that SHOULD be affected by removing an object.
        
        Uses transitive closure: if A affects B and B affects C, then A affects C.
        
        Args:
            causal_graph: [batch, num_objects, num_objects]
            removed_object: Index of object being removed
        
        Returns:
            affected_mask: [batch, num_objects] - 1 for affected, 0 for unaffected
        """
        batch_size, num_objects, _ = causal_graph.shape
        device = causal_graph.device
        
        affected = causal_graph[:, removed_object, :].clone()
        
        for _ in range(3):
            propagated = torch.bmm(affected.unsqueeze(1), causal_graph).squeeze(1)
            affected = (affected + propagated).clamp(0, 1)
        
        affected[:, removed_object] = 0
        
        return affected


class CausalObjectDropout(nn.Module):
    """
    Randomly drops objects during training to create counterfactual scenarios.
    
    For each batch:
    1. Select random object to remove
    2. Zero out that object's state
    3. Track which objects SHOULD be affected (via causal graph)
    4. Return both original and counterfactual states
    """
    
    def __init__(self, config: CausalTrainingConfig):
        super().__init__()
        self.config = config
        self.causal_builder = CausalGraphBuilder(config)
    
    def forward(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        force_dropout: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Apply causal object dropout.
        
        Args:
            states: [batch, seq_len, num_objects, state_dim]
            mask: [batch, num_objects]
            force_dropout: Always drop an object (for training)
        
        Returns:
            Dict with:
                - original_states: unchanged states
                - counterfactual_states: states with object removed
                - removed_objects: [batch] indices of removed objects
                - affected_mask: [batch, num_objects] which objects should change
                - unaffected_mask: [batch, num_objects] which should NOT change
        """
        batch_size, seq_len, num_objects, state_dim = states.shape
        device = states.device
        
        causal_graph = self.causal_builder.build_causal_graph(states, mask)
        
        removed_objects = torch.zeros(batch_size, dtype=torch.long, device=device)
        counterfactual_states = states.clone()
        affected_masks = torch.zeros(batch_size, num_objects, device=device)
        
        if mask.dim() == 2:
            obj_mask = mask[:, :num_objects]
        elif mask.dim() == 3:
            obj_mask = mask[:, 0, :num_objects]
        else:
            obj_mask = mask.view(batch_size, -1)[:, :num_objects]
        
        dropout_rand = torch.rand(batch_size, device=device)
        if force_dropout:
            should_dropout = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            should_dropout = dropout_rand < self.config.dropout_prob
        
        active_count = (obj_mask > 0.5).sum(dim=1)
        should_dropout = should_dropout & (active_count >= 2)
        
        obj_rand = torch.rand(batch_size, num_objects, device=device)
        obj_rand = obj_rand * (obj_mask > 0.5).float()
        obj_rand[~should_dropout] = -1
        
        removed_objects = obj_rand.argmax(dim=1)
        removed_objects[~should_dropout] = -1
        
        remove_mask = torch.zeros(batch_size, num_objects, device=device)
        valid_removes = removed_objects >= 0
        if valid_removes.any():
            remove_mask[valid_removes, removed_objects[valid_removes]] = 1.0
        
        counterfactual_states = states * (1 - remove_mask.view(batch_size, 1, num_objects, 1))
        
        affected_masks = (causal_graph * remove_mask.unsqueeze(1)).sum(dim=2)
        affected_masks = (affected_masks > 0).float()
        
        unaffected_masks = (1 - affected_masks) * obj_mask
        unaffected_masks = unaffected_masks * (1 - remove_mask)
        
        return {
            'original_states': states,
            'counterfactual_states': counterfactual_states,
            'removed_objects': removed_objects,
            'affected_mask': affected_masks,
            'unaffected_mask': unaffected_masks,
            'causal_graph': causal_graph
        }


class CausalInterventionLoss(nn.Module):
    """
    Loss function that enforces causal consistency.
    
    Key insight: When we remove an object, only causally connected objects
    should have different predictions. Unconnected objects should predict
    the SAME trajectory regardless of the intervention.
    
    Loss components:
    1. Prediction loss: Standard next-state prediction
    2. Intervention consistency: Unaffected objects should be unchanged
    3. Causal sensitivity: Affected objects SHOULD change
    """
    
    def __init__(self, config: CausalTrainingConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        original_predictions: torch.Tensor,
        counterfactual_predictions: torch.Tensor,
        original_targets: torch.Tensor,
        affected_mask: torch.Tensor,
        unaffected_mask: torch.Tensor,
        object_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute causal intervention loss.
        
        Args:
            original_predictions: [batch, seq_len, num_objects, state_dim]
            counterfactual_predictions: [batch, seq_len, num_objects, state_dim]
            original_targets: [batch, seq_len, num_objects, state_dim]
            affected_mask: [batch, num_objects] - objects that should change
            unaffected_mask: [batch, num_objects] - objects that should NOT change
            object_mask: [batch, num_objects] - valid objects
        
        Returns:
            Dict with loss components
        """
        batch_size = original_predictions.shape[0]
        pred_seq_len = original_predictions.shape[1]
        pred_num_objects = original_predictions.shape[2]
        
        # Handle 2D or 3D object_mask
        if object_mask.dim() == 3:
            # [batch, seq_len, num_objects] -> [batch, num_objects] (use first timestep)
            object_mask = object_mask[:, 0, :]
        
        mask_num_objects = object_mask.shape[1]
        num_objects = min(pred_num_objects, mask_num_objects)
        
        target_seq_len = original_targets.shape[1]
        min_seq_len = min(pred_seq_len, target_seq_len)
        
        orig_pred = original_predictions[:, :min_seq_len, :num_objects]
        cf_pred = counterfactual_predictions[:, :min_seq_len, :num_objects]
        targets = original_targets[:, :min_seq_len, :num_objects]
        
        obj_mask = object_mask[:, :num_objects]
        aff_mask = affected_mask[:, :num_objects]
        unaff_mask = unaffected_mask[:, :num_objects]
        
        prediction_loss = F.huber_loss(
            orig_pred * obj_mask.unsqueeze(1).unsqueeze(-1),
            targets * obj_mask.unsqueeze(1).unsqueeze(-1),
            delta=1.0
        )
        
        pred_diff = orig_pred - cf_pred
        
        unaffected_expanded = unaff_mask.unsqueeze(1).unsqueeze(-1)
        intervention_loss = (pred_diff.abs() * unaffected_expanded).mean()
        
        affected_expanded = aff_mask.unsqueeze(1).unsqueeze(-1)
        affected_diff = (pred_diff.abs() * affected_expanded).sum(dim=-1).mean(dim=1)
        
        causal_sensitivity_loss = F.relu(
            self.config.causal_margin - affected_diff
        ).mean()
        
        total_loss = (
            prediction_loss +
            self.config.intervention_weight * intervention_loss +
            self.config.contrastive_weight * causal_sensitivity_loss
        )
        
        return {
            'total': total_loss,
            'prediction': prediction_loss,
            'intervention': intervention_loss,
            'causal_sensitivity': causal_sensitivity_loss
        }


class CausalPhysicsTrainer:
    """
    Trainer that adds causal intervention training to PhysicsFormer.
    
    Training loop:
    1. Get batch of physics states
    2. Apply causal object dropout
    3. Run model on both original and counterfactual
    4. Compute causal intervention loss
    5. Backprop and update
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[CausalTrainingConfig] = None
    ):
        self.model = model
        self.config = config or CausalTrainingConfig()
        self.dropout = CausalObjectDropout(self.config)
        self.loss_fn = CausalInterventionLoss(self.config)
    
    def train_step(
        self,
        states: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Single causal training step.
        
        Args:
            states: [batch, seq_len, num_objects, state_dim]
            mask: [batch, num_objects]
        
        Returns:
            Dict with loss components
        """
        dropout_result = self.dropout(states, mask)
        
        original_pred, _, _ = self.model.forward_physics(
            dropout_result['original_states'], mask
        )
        
        batch_size = states.shape[0]
        num_objects = states.shape[2]
        
        if mask.dim() == 2:
            cf_mask = mask[:, :num_objects].clone()
        elif mask.dim() == 3:
            cf_mask = mask[:, 0, :num_objects].clone()
        else:
            cf_mask = mask.view(batch_size, -1)[:, :num_objects].clone()
        
        removed = dropout_result['removed_objects']
        valid_removes = removed >= 0
        if valid_removes.any():
            cf_mask[valid_removes, removed[valid_removes]] = 0
        
        counterfactual_pred, _, _ = self.model.forward_physics(
            dropout_result['counterfactual_states'], cf_mask
        )
        
        targets = states[:, 1:]
        
        losses = self.loss_fn(
            original_predictions=original_pred,
            counterfactual_predictions=counterfactual_pred,
            original_targets=targets,
            affected_mask=dropout_result['affected_mask'],
            unaffected_mask=dropout_result['unaffected_mask'],
            object_mask=mask
        )
        
        return losses


def add_causal_training_to_pipeline(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    mask: torch.Tensor,
    config: Optional[CausalTrainingConfig] = None
) -> Dict[str, float]:
    """
    Add causal training step to existing training pipeline.
    
    Can be called alongside regular physics training to add
    causal intervention capability.
    
    Args:
        model: PhysicsFormer model
        optimizer: Optimizer
        states: [batch, seq_len, num_objects, state_dim]
        mask: [batch, num_objects]
        config: Causal training config
    
    Returns:
        Dict with loss values
    """
    trainer = CausalPhysicsTrainer(model, config)
    
    optimizer.zero_grad()
    losses = trainer.train_step(states, mask)
    losses['total'].backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return {k: v.item() for k, v in losses.items()}


if __name__ == '__main__':
    print("="*70)
    print("CAUSAL TRAINING MODULE")
    print("="*70)
    print()
    print("This module adds causal intervention training to PhysicsFormer.")
    print()
    print("Key Components:")
    print("  - CausalGraphBuilder: Determines which objects affect which")
    print("  - CausalObjectDropout: Creates counterfactual scenarios")
    print("  - CausalInterventionLoss: Enforces causal consistency")
    print("  - CausalPhysicsTrainer: Training loop with interventions")
    print()
    print("Training Strategy:")
    print("  1. Randomly remove objects during training")
    print("  2. Build causal graph to know which objects SHOULD be affected")
    print("  3. Penalize when unaffected objects change (spurious correlation)")
    print("  4. Reward when affected objects DO change (causal sensitivity)")
    print()
    print("This addresses the CLEVRER-Humans gap by teaching the model")
    print("to distinguish correlation from causation.")
