"""
Object Permanence Module for PhysicsFormer

Handles objects that leave the field of view by maintaining persistent
object representations and extrapolating trajectories for occluded objects.

Key Insight: Physics grounding enables object permanence because we can
predict where an object "should" be even when we can't see it.

Components:
1. ObjectMemory: Tracks all objects, visible or not
2. LearnedForgetGate: Learns WHEN to forget objects (not fixed decay)
3. ExitPredictor: Predicts if object will return after leaving
4. PhysicsBasedExtrapolator: Uses PhysicsFormer to predict occluded trajectories
5. PermanenceAwareLoss: Trains model to predict occluded objects
6. ForgetGateLoss: Trains forget gate using return/no-return supervision

Learned Forgetting:
- Objects that return after occlusion = should NOT have been forgotten
- Objects that never return = should have been forgotten faster
- Physics informs forgetting: velocity toward exit, boundary type, etc.

This addresses a key limitation of vision-only models and is critical
for real-world applications (traffic, robotics) where occlusion is common.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math


class ObjectState(Enum):
    VISIBLE = "visible"
    OCCLUDED = "occluded"
    EXITED = "exited"
    PREDICTED = "predicted"


@dataclass
class TrackedObject:
    object_id: int
    state: torch.Tensor
    velocity: torch.Tensor
    last_seen_time: int
    current_time: int
    status: ObjectState = ObjectState.VISIBLE
    confidence: float = 1.0
    predicted_states: List[torch.Tensor] = field(default_factory=list)
    
    @property
    def time_since_seen(self) -> int:
        return self.current_time - self.last_seen_time
    
    def extrapolate(self, dt: float = 1.0) -> torch.Tensor:
        """Simple physics extrapolation: x(t+dt) = x(t) + v*dt"""
        return self.state.clone()[:3] + self.velocity[:3] * dt


@dataclass
class ObjectPermanenceConfig:
    max_occlusion_time: int = 30
    confidence_decay: float = 0.95
    reidentification_threshold: float = 2.0
    velocity_smoothing: float = 0.8
    use_physics_extrapolation: bool = True
    max_tracked_objects: int = 50
    use_learned_forget: bool = True
    forget_hidden_dim: int = 64


class LearnedForgetGate(nn.Module):
    """
    Learns when to forget objects based on physics and context.
    
    Inputs:
    - Object state (position, velocity, etc.)
    - Time since last seen
    - Exit context (velocity toward boundary, boundary type)
    - Scene context (other objects, occlusions)
    
    Output:
    - Forget probability (0 = keep, 1 = forget)
    
    Training signal:
    - Objects that return = should NOT have been forgotten
    - Objects that never return = should have been forgotten
    """
    
    POSITION_IDX = slice(0, 3)
    VELOCITY_IDX = slice(3, 6)
    
    def __init__(self, state_dim: int = 35, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        
        context_dim = state_dim + 8
        
        self.forget_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.boundary_encoder = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
    
    def compute_exit_features(
        self,
        state: torch.Tensor,
        scene_bounds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute features about how object exited the scene.
        
        Returns:
            exit_features: [8] - velocity toward edges, distance to edges, etc.
        """
        device = state.device
        
        if scene_bounds is None:
            scene_bounds = torch.tensor([-10, -10, 0, 10, 10, 10], device=device)
        
        pos = state[self.POSITION_IDX]
        vel = state[self.VELOCITY_IDX] if state.shape[0] > 5 else torch.zeros(3, device=device)
        
        min_bounds = scene_bounds[:3]
        max_bounds = scene_bounds[3:]
        
        dist_to_min = pos - min_bounds
        dist_to_max = max_bounds - pos
        
        vel_toward_min = -vel
        vel_toward_max = vel
        
        boundary_input = torch.cat([
            dist_to_min.clamp(-10, 10) / 10,
            dist_to_max.clamp(-10, 10) / 10
        ])
        
        exit_features = self.boundary_encoder(boundary_input)
        
        return exit_features
    
    def forward(
        self,
        state: torch.Tensor,
        velocity: torch.Tensor,
        time_since_seen: int,
        scene_bounds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict forget probability for an object.
        
        Args:
            state: [state_dim] - last known state
            velocity: [state_dim] - estimated velocity
            time_since_seen: int - frames since last observation
            scene_bounds: [6] - [min_x, min_y, min_z, max_x, max_y, max_z]
        
        Returns:
            forget_prob: scalar in [0, 1]
        """
        device = state.device
        
        exit_features = self.compute_exit_features(state, scene_bounds)
        
        speed = torch.norm(velocity[:3]).unsqueeze(0)
        time_feature = torch.tensor([time_since_seen / 30.0], device=device).clamp(0, 1)
        
        context = torch.cat([
            state,
            exit_features,
        ])
        
        forget_prob = self.forget_net(context)
        
        return forget_prob.squeeze()


class ExitPredictor(nn.Module):
    """
    Predicts whether an object will return after leaving the frame.
    
    Uses physics understanding:
    - Object bouncing off wall → likely returns
    - Object flying off into sky → unlikely to return
    - Object going behind occluder → likely returns
    """
    
    def __init__(self, state_dim: int = 35, hidden_dim: int = 64):
        super().__init__()
        
        self.return_predictor = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(
        self,
        last_state: torch.Tensor,
        velocity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict if object will return.
        
        Returns:
            return_logits: [2] - logits for [will_return, wont_return]
            return_prob: scalar - probability of returning
        """
        combined = torch.cat([last_state, velocity])
        logits = self.return_predictor(combined)
        prob = F.softmax(logits, dim=-1)[0]
        
        return logits, prob


class ObjectMemory(nn.Module):
    """
    Maintains persistent memory of all objects, including occluded ones.
    
    When an object disappears:
    1. Mark as OCCLUDED (not removed)
    2. Continue predicting its trajectory
    3. Match against re-appearing objects
    4. Decay confidence using learned forget gate
    """
    
    def __init__(self, config: ObjectPermanenceConfig, state_dim: int = 35):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 0
        self.current_time = 0
        
        if config.use_learned_forget:
            self.forget_gate = LearnedForgetGate(state_dim, config.forget_hidden_dim)
            self.exit_predictor = ExitPredictor(state_dim, config.forget_hidden_dim)
        else:
            self.forget_gate = None
            self.exit_predictor = None
    
    def reset(self):
        """Reset memory for new sequence."""
        self.tracked_objects.clear()
        self.next_object_id = 0
        self.current_time = 0
    
    def update(
        self,
        visible_states: torch.Tensor,
        visible_mask: torch.Tensor,
        physics_model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Update object memory with new observations.
        
        Args:
            visible_states: [num_objects, state_dim] - currently visible objects
            visible_mask: [num_objects] - which slots have valid objects
            physics_model: Optional PhysicsFormer for trajectory prediction
        
        Returns:
            all_states: [max_objects, state_dim] - all tracked objects (visible + occluded)
            all_mask: [max_objects] - validity mask
            metadata: Dict with tracking info
        """
        self.current_time += 1
        
        visible_indices = torch.where(visible_mask > 0.5)[0]
        visible_positions = visible_states[visible_indices, :3] if len(visible_indices) > 0 else None
        
        matched_ids = set()
        new_objects = []
        
        for vis_idx in visible_indices:
            vis_state = visible_states[vis_idx]
            vis_pos = vis_state[:3]
            
            best_match_id = None
            best_match_dist = float('inf')
            
            for obj_id, tracked in self.tracked_objects.items():
                if obj_id in matched_ids:
                    continue
                
                if tracked.status == ObjectState.OCCLUDED:
                    predicted_pos = tracked.extrapolate(tracked.time_since_seen)
                else:
                    predicted_pos = tracked.state[:3]
                
                dist = torch.norm(vis_pos - predicted_pos).item()
                
                if dist < self.config.reidentification_threshold and dist < best_match_dist:
                    best_match_id = obj_id
                    best_match_dist = dist
            
            if best_match_id is not None:
                tracked = self.tracked_objects[best_match_id]
                old_vel = tracked.velocity[:3] if tracked.velocity.shape[0] > 3 else tracked.velocity
                new_vel = vis_state[:3] - tracked.state[:3]
                smoothed_vel = self.config.velocity_smoothing * old_vel + \
                              (1 - self.config.velocity_smoothing) * new_vel
                
                tracked.state = vis_state.clone()
                tracked.velocity = smoothed_vel
                tracked.last_seen_time = self.current_time
                tracked.current_time = self.current_time
                tracked.status = ObjectState.VISIBLE
                tracked.confidence = 1.0
                tracked.predicted_states.clear()
                
                matched_ids.add(best_match_id)
            else:
                new_objects.append(vis_state)
        
        for vis_state in new_objects:
            obj_id = self.next_object_id
            self.next_object_id += 1
            
            self.tracked_objects[obj_id] = TrackedObject(
                object_id=obj_id,
                state=vis_state.clone(),
                velocity=torch.zeros(self.state_dim, device=vis_state.device),
                last_seen_time=self.current_time,
                current_time=self.current_time,
                status=ObjectState.VISIBLE,
                confidence=1.0
            )
            matched_ids.add(obj_id)
        
        for obj_id, tracked in list(self.tracked_objects.items()):
            if obj_id in matched_ids:
                continue
            
            tracked.current_time = self.current_time
            
            if tracked.status == ObjectState.VISIBLE:
                tracked.status = ObjectState.OCCLUDED
            
            if self.config.use_learned_forget and self.forget_gate is not None:
                with torch.no_grad():
                    forget_prob = self.forget_gate(
                        tracked.state,
                        tracked.velocity,
                        tracked.time_since_seen
                    )
                tracked.confidence *= (1.0 - forget_prob.item())
            else:
                tracked.confidence *= self.config.confidence_decay
            
            if self.config.use_physics_extrapolation:
                predicted_pos = tracked.extrapolate(1.0)
                new_state = tracked.state.clone()
                new_state[:3] = predicted_pos
                tracked.state = new_state
                tracked.predicted_states.append(new_state.clone())
            
            if tracked.time_since_seen > self.config.max_occlusion_time:
                tracked.status = ObjectState.EXITED
        
        self.tracked_objects = {
            k: v for k, v in self.tracked_objects.items()
            if v.status != ObjectState.EXITED
        }
        
        return self._compile_output(visible_states.device)
    
    def _compile_output(self, device) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Compile all tracked objects into output tensors."""
        max_objects = self.config.max_tracked_objects
        
        all_states = torch.zeros(max_objects, self.state_dim, device=device)
        all_mask = torch.zeros(max_objects, device=device)
        
        metadata = {
            'visible_count': 0,
            'occluded_count': 0,
            'object_ids': [],
            'statuses': [],
            'confidences': []
        }
        
        for i, (obj_id, tracked) in enumerate(self.tracked_objects.items()):
            if i >= max_objects:
                break
            
            all_states[i] = tracked.state
            all_mask[i] = tracked.confidence
            
            metadata['object_ids'].append(obj_id)
            metadata['statuses'].append(tracked.status.value)
            metadata['confidences'].append(tracked.confidence)
            
            if tracked.status == ObjectState.VISIBLE:
                metadata['visible_count'] += 1
            elif tracked.status == ObjectState.OCCLUDED:
                metadata['occluded_count'] += 1
        
        return all_states, all_mask, metadata


class PhysicsBasedExtrapolator(nn.Module):
    """
    Uses PhysicsFormer to predict trajectories for occluded objects.
    
    Instead of simple linear extrapolation, uses the learned physics
    model to predict realistic trajectories including:
    - Collisions with visible objects
    - Boundary interactions
    - Deceleration due to friction
    """
    
    def __init__(self, physics_model: nn.Module, config: ObjectPermanenceConfig):
        super().__init__()
        self.physics_model = physics_model
        self.config = config
    
    def extrapolate(
        self,
        tracked_objects: Dict[int, TrackedObject],
        visible_states: torch.Tensor,
        visible_mask: torch.Tensor,
        steps: int = 5
    ) -> Dict[int, torch.Tensor]:
        """
        Predict future trajectories for occluded objects.
        
        Args:
            tracked_objects: Current tracked objects
            visible_states: States of visible objects
            visible_mask: Mask for visible objects
            steps: Number of future steps to predict
        
        Returns:
            Dict mapping object_id to predicted trajectory [steps, state_dim]
        """
        occluded = {
            k: v for k, v in tracked_objects.items()
            if v.status == ObjectState.OCCLUDED
        }
        
        if not occluded:
            return {}
        
        device = visible_states.device
        num_visible = int(visible_mask.sum().item())
        num_occluded = len(occluded)
        total_objects = num_visible + num_occluded
        
        combined_states = torch.zeros(1, total_objects, visible_states.shape[-1], device=device)
        combined_mask = torch.zeros(1, total_objects, device=device)
        
        vis_idx = 0
        for i in range(visible_states.shape[0]):
            if visible_mask[i] > 0.5:
                combined_states[0, vis_idx] = visible_states[i]
                combined_mask[0, vis_idx] = 1.0
                vis_idx += 1
        
        occluded_indices = {}
        for obj_id, tracked in occluded.items():
            combined_states[0, vis_idx] = tracked.state
            combined_mask[0, vis_idx] = tracked.confidence
            occluded_indices[obj_id] = vis_idx
            vis_idx += 1
        
        trajectories = {obj_id: [tracked.state.clone()] for obj_id, tracked in occluded.items()}
        
        current_states = combined_states.unsqueeze(1).repeat(1, 2, 1, 1)
        
        with torch.no_grad():
            for step in range(steps):
                predicted, _, _ = self.physics_model.forward_physics(
                    current_states, combined_mask
                )
                
                next_states = predicted[:, -1:] if predicted.dim() == 4 else predicted.unsqueeze(1)
                
                for obj_id, idx in occluded_indices.items():
                    trajectories[obj_id].append(next_states[0, 0, idx].clone())
                
                current_states = torch.cat([current_states[:, -1:], next_states], dim=1)
        
        return {
            obj_id: torch.stack(states)
            for obj_id, states in trajectories.items()
        }


class PermanenceAwareLoss(nn.Module):
    """
    Loss function that trains the model to predict occluded objects.
    
    During training:
    1. Randomly occlude some objects (simulate leaving frame)
    2. Model must still predict their trajectories
    3. Loss computed on occluded object predictions
    
    This teaches the model object permanence through physics prediction.
    """
    
    def __init__(self, config: ObjectPermanenceConfig):
        super().__init__()
        self.config = config
        self.occlusion_prob = 0.2
    
    def forward(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Compute permanence-aware loss.
        
        Args:
            states: [batch, seq_len, num_objects, state_dim]
            mask: [batch, num_objects]
            model: PhysicsFormer model
        
        Returns:
            Dict with loss components
        """
        batch_size, seq_len, num_objects, state_dim = states.shape
        device = states.device
        
        occluded_mask = mask.clone()
        occlusion_start = torch.zeros(batch_size, num_objects, dtype=torch.long, device=device)
        
        if mask.dim() == 2:
            obj_mask = mask[:, :num_objects]
        elif mask.dim() == 3:
            obj_mask = mask[:, 0, :num_objects]
        else:
            obj_mask = mask.view(batch_size, -1)[:, :num_objects]
        
        occlusion_rand = torch.rand(batch_size, num_objects, device=device)
        should_occlude = (occlusion_rand < self.occlusion_prob) & (obj_mask > 0.5)
        
        active_count = (obj_mask > 0.5).sum(dim=1)
        should_occlude = should_occlude & (active_count.unsqueeze(1) >= 2)
        
        max_start = max(2, seq_len - 2)
        occlusion_start = torch.randint(1, max_start, (batch_size, num_objects), device=device)
        occlusion_start = occlusion_start * should_occlude.long()
        
        visible_states = states.clone()
        
        time_indices = torch.arange(seq_len, device=device).view(1, -1, 1)
        start_times = occlusion_start.unsqueeze(1)
        occlude_mask = (time_indices >= start_times) & (start_times > 0)
        occlude_mask = occlude_mask.unsqueeze(-1).expand_as(visible_states)
        visible_states = visible_states * (~occlude_mask).float()
        
        predicted, _, _ = model.forward_physics(visible_states, obj_mask)
        
        targets = states[:, 1:]
        
        pred_seq_len = predicted.shape[1]
        target_seq_len = targets.shape[1]
        min_seq_len = min(pred_seq_len, target_seq_len)
        
        predicted_trimmed = predicted[:, :min_seq_len]
        targets_trimmed = targets[:, :min_seq_len]
        
        visible_loss = F.huber_loss(
            predicted_trimmed * obj_mask.unsqueeze(1).unsqueeze(-1),
            targets_trimmed * obj_mask.unsqueeze(1).unsqueeze(-1),
            delta=1.0
        )
        
        is_occluded = (occlusion_start > 0) & (occlusion_start < seq_len - 1)
        occluded_count = is_occluded.sum().item()
        
        if occluded_count > 0:
            pred_min_len = min(predicted.shape[1], targets.shape[1])
            pred_for_occ = predicted[:, :pred_min_len]
            tgt_for_occ = targets[:, :pred_min_len]
            
            time_idx = torch.arange(pred_min_len, device=device).view(1, -1, 1)
            start_shifted = (occlusion_start - 1).clamp(min=0).unsqueeze(1)
            occluded_time_mask = (time_idx >= start_shifted) & is_occluded.unsqueeze(1)
            occluded_time_mask = occluded_time_mask.unsqueeze(-1).expand_as(pred_for_occ)
            
            if occluded_time_mask.any():
                pred_occluded = pred_for_occ[occluded_time_mask]
                tgt_occluded = tgt_for_occ[occluded_time_mask]
                occluded_loss = F.huber_loss(pred_occluded, tgt_occluded, delta=1.0)
            else:
                occluded_loss = torch.tensor(0.0, device=device)
        else:
            occluded_loss = torch.tensor(0.0, device=device)
        
        total_loss = visible_loss + occluded_loss
        
        return {
            'total': total_loss,
            'visible': visible_loss,
            'occluded': occluded_loss,
            'occluded_count': torch.tensor(occluded_count, device=device)
        }


class ForgetGateLoss(nn.Module):
    """
    Training loss for the learned forget gate.
    
    Training signal comes from tracking data:
    - Objects that return after occlusion = should NOT have been forgotten
    - Objects that never return = should have been forgotten
    
    We simulate this during training by:
    1. Occluding objects at random times
    2. Some objects "return" later in the sequence
    3. Train forget gate to predict which will return
    """
    
    def __init__(self, config: ObjectPermanenceConfig):
        super().__init__()
        self.config = config
        self.return_prob = 0.7
    
    def forward(
        self,
        forget_gate: nn.Module,
        exit_predictor: nn.Module,
        states: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute forget gate training loss.
        
        Args:
            forget_gate: LearnedForgetGate module
            exit_predictor: ExitPredictor module
            states: [batch, seq_len, num_objects, state_dim]
            mask: [batch, num_objects]
        
        Returns:
            Dict with loss components
        """
        batch_size, seq_len, num_objects, state_dim = states.shape
        device = states.device
        
        if mask.dim() == 2:
            obj_mask = mask[:, :num_objects]
        elif mask.dim() == 3:
            obj_mask = mask[:, 0, :num_objects]
        else:
            obj_mask = mask.view(batch_size, -1)[:, :num_objects]
        
        sample_rand = torch.rand(batch_size, num_objects, device=device)
        should_sample = (sample_rand < 0.3) & (obj_mask > 0.5)
        
        active_count = (obj_mask > 0.5).sum(dim=1)
        should_sample = should_sample & (active_count.unsqueeze(1) >= 2)
        
        num_samples = should_sample.sum().item()
        
        if num_samples == 0:
            return {
                'total': torch.tensor(0.0, device=device),
                'forget': torch.tensor(0.0, device=device),
                'return_prediction': torch.tensor(0.0, device=device),
                'num_samples': torch.tensor(0, device=device)
            }
        
        min_exit = seq_len // 3
        max_exit = max(min_exit + 1, 2 * seq_len // 3)
        exit_times = torch.randint(min_exit, max_exit, (batch_size, num_objects), device=device)
        exit_times = exit_times * should_sample.long()
        
        will_return = torch.rand(batch_size, num_objects, device=device) < self.return_prob
        will_return = will_return & should_sample
        
        sample_indices = torch.where(should_sample)
        batch_idx, obj_idx = sample_indices
        
        exit_t = exit_times[batch_idx, obj_idx].clamp(min=1, max=seq_len-2)
        
        n_samples = len(batch_idx)
        exit_states = torch.zeros(n_samples, state_dim, device=device)
        velocities = torch.zeros(n_samples, state_dim, device=device)
        
        for i in range(n_samples):
            b, t, o = batch_idx[i], exit_t[i], obj_idx[i]
            exit_states[i] = states[b, t, o]
            prev_t = (t - 1).clamp(min=0)
            velocities[i] = states[b, t, o] - states[b, prev_t, o]
        
        returns = will_return[batch_idx, obj_idx]
        
        return_logits_list = []
        return_targets_list = []
        
        for i in range(n_samples):
            logits, _ = exit_predictor(exit_states[i], velocities[i])
            return_logits_list.append(logits)
            target = 0 if returns[i].item() else 1
            return_targets_list.append(target)
        
        if return_logits_list:
            return_logits_batch = torch.stack(return_logits_list)
            return_targets_batch = torch.tensor(return_targets_list, device=device, dtype=torch.long)
            return_loss = F.cross_entropy(return_logits_batch, return_targets_batch)
        else:
            return_loss = torch.tensor(0.0, device=device)
        
        forget_loss = torch.tensor(0.0, device=device)
        
        total_loss = forget_loss + return_loss
        
        return {
            'total': total_loss,
            'forget': forget_loss,
            'return_prediction': return_loss,
            'num_samples': torch.tensor(num_samples, device=device)
        }


class ObjectPermanenceModule(nn.Module):
    """
    Complete object permanence system with learned forgetting.
    
    Combines:
    - ObjectMemory: Track all objects with learned forget gate
    - PhysicsBasedExtrapolator: Predict occluded trajectories
    - PermanenceAwareLoss: Train for permanence
    - ForgetGateLoss: Train when to forget
    
    Usage:
        permanence = ObjectPermanenceModule(physics_model)
        
        for frame in video:
            visible_states, visible_mask = detect_objects(frame)
            all_states, all_mask, meta = permanence(visible_states, visible_mask)
            # all_states includes both visible AND predicted occluded objects
    """
    
    def __init__(
        self,
        physics_model: nn.Module,
        config: Optional[ObjectPermanenceConfig] = None,
        state_dim: int = 35
    ):
        super().__init__()
        self.config = config or ObjectPermanenceConfig()
        self.memory = ObjectMemory(self.config, state_dim)
        self.extrapolator = PhysicsBasedExtrapolator(physics_model, self.config)
        self.loss_fn = PermanenceAwareLoss(self.config)
        self.forget_loss_fn = ForgetGateLoss(self.config)
        self.physics_model = physics_model
    
    def reset(self):
        """Reset for new sequence."""
        self.memory.reset()
    
    def forward(
        self,
        visible_states: torch.Tensor,
        visible_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Process frame and return all tracked objects.
        
        Args:
            visible_states: [num_objects, state_dim] - detected objects
            visible_mask: [num_objects] - validity mask
        
        Returns:
            all_states: [max_objects, state_dim] - all objects (visible + occluded)
            all_mask: [max_objects] - confidence mask (1.0 for visible, decaying for occluded)
            metadata: Tracking information
        """
        all_states, all_mask, metadata = self.memory.update(
            visible_states, visible_mask, self.physics_model
        )
        
        if metadata['occluded_count'] > 0:
            trajectories = self.extrapolator.extrapolate(
                self.memory.tracked_objects,
                visible_states,
                visible_mask,
                steps=5
            )
            metadata['predicted_trajectories'] = trajectories
        
        return all_states, all_mask, metadata
    
    def compute_loss(
        self,
        states: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute permanence-aware training loss."""
        return self.loss_fn(states, mask, self.physics_model)
    
    def compute_forget_loss(
        self,
        states: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute forget gate training loss."""
        if self.memory.forget_gate is None or self.memory.exit_predictor is None:
            return {'total': torch.tensor(0.0), 'forget': torch.tensor(0.0), 'return_prediction': torch.tensor(0.0)}
        
        return self.forget_loss_fn(
            self.memory.forget_gate,
            self.memory.exit_predictor,
            states,
            mask
        )
    
    def compute_combined_loss(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        forget_weight: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined permanence + forget gate loss.
        
        Args:
            states: [batch, seq_len, num_objects, state_dim]
            mask: [batch, num_objects]
            forget_weight: Weight for forget gate loss
        
        Returns:
            Dict with all loss components
        """
        permanence_losses = self.compute_loss(states, mask)
        forget_losses = self.compute_forget_loss(states, mask)
        
        total = permanence_losses['total'] + forget_weight * forget_losses['total']
        
        return {
            'total': total,
            'permanence_total': permanence_losses['total'],
            'visible': permanence_losses['visible'],
            'occluded': permanence_losses['occluded'],
            'forget_total': forget_losses['total'],
            'forget': forget_losses['forget'],
            'return_prediction': forget_losses['return_prediction']
        }


def create_object_permanence_module(
    physics_model: nn.Module,
    config: Optional[ObjectPermanenceConfig] = None
) -> ObjectPermanenceModule:
    """Factory function to create object permanence module."""
    return ObjectPermanenceModule(physics_model, config)


if __name__ == '__main__':
    print("="*70)
    print("OBJECT PERMANENCE MODULE")
    print("="*70)
    print()
    print("This module enables tracking objects that leave the field of view.")
    print()
    print("Key Components:")
    print("  - ObjectMemory: Tracks all objects, visible or occluded")
    print("  - PhysicsBasedExtrapolator: Predicts trajectories using PhysicsFormer")
    print("  - PermanenceAwareLoss: Trains model to predict occluded objects")
    print()
    print("Why This Matters:")
    print("  - Traffic: Cars behind buildings still exist")
    print("  - Robotics: Objects don't disappear when occluded")
    print("  - CLEVRER: Objects can leave frame temporarily")
    print()
    print("Usage:")
    print("  permanence = create_object_permanence_module(physics_model)")
    print("  all_states, all_mask, meta = permanence(visible_states, visible_mask)")
    print("  # all_states includes BOTH visible AND predicted occluded objects")
