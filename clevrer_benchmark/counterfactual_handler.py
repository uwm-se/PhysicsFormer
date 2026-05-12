"""
Counterfactual Question Handler for CLEVRER

Handles counterfactual questions like "What would happen if X were removed?"
by modifying state tensors and running forward predictions.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any


class CounterfactualHandler:
    """Handles counterfactual reasoning for Physics-LLM on CLEVRER."""
    
    def __init__(self, physics_model: Any, device: str = 'cuda'):
        """
        Initialize counterfactual handler.
        
        Args:
            physics_model: Physics-LLM model for forward prediction
            device: Device to run model on
        """
        self.model = physics_model
        self.device = device
    
    def remove_object(
        self,
        states: np.ndarray,
        masks: np.ndarray,
        object_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove an object from the scene by zeroing its state and mask.
        
        Args:
            states: [T, N, D] state tensor
            masks: [T, N] validity masks
            object_idx: Index of object to remove
            
        Returns:
            Modified states and masks with object removed
        """
        modified_states = states.copy()
        modified_masks = masks.copy()
        
        modified_states[:, object_idx, :] = 0.0
        modified_masks[:, object_idx] = 0.0
        
        return modified_states, modified_masks
    
    def predict_future_states(
        self,
        states: np.ndarray,
        masks: np.ndarray,
        num_future_frames: int = 50
    ) -> np.ndarray:
        """
        Run forward prediction to get future states.
        
        Args:
            states: [T, N, D] current state tensor
            masks: [T, N] validity masks
            num_future_frames: Number of frames to predict
            
        Returns:
            [num_future_frames, N, D] predicted future states
        """
        states_tensor = torch.from_numpy(states).float().unsqueeze(0).to(self.device)
        masks_tensor = torch.from_numpy(masks).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model.predict_trajectory(
                states=states_tensor,
                masks=masks_tensor,
                num_steps=num_future_frames
            )
        
        return predictions.squeeze(0).cpu().numpy()
    
    def detect_collisions(
        self,
        states: np.ndarray,
        collision_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Detect collision events in state sequence.
        
        Args:
            states: [T, N, D] state tensor
            collision_threshold: Distance threshold for collision detection
            
        Returns:
            List of collision events with frame, objects, and positions
        """
        num_frames, num_objects, _ = states.shape
        collisions = []
        
        for t in range(num_frames):
            for i in range(num_objects):
                for j in range(i + 1, num_objects):
                    pos_i = states[t, i, 0:3]
                    pos_j = states[t, j, 0:3]
                    
                    if np.all(pos_i == 0) or np.all(pos_j == 0):
                        continue
                    
                    distance = np.linalg.norm(pos_i - pos_j)
                    
                    bbox_i = states[t, i, 13:16]
                    bbox_j = states[t, j, 13:16]
                    combined_radius = (np.mean(bbox_i) + np.mean(bbox_j)) / 2
                    
                    if distance < combined_radius + collision_threshold:
                        collisions.append({
                            'frame': t,
                            'objects': [i, j],
                            'distance': distance,
                            'positions': [pos_i.tolist(), pos_j.tolist()]
                        })
        
        return collisions
    
    def answer_counterfactual(
        self,
        states: np.ndarray,
        masks: np.ndarray,
        question_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Answer a counterfactual question.
        
        Args:
            states: [T, N, D] original state tensor
            masks: [T, N] original validity masks
            question_data: Processed question data
            metadata: Scene metadata
            
        Returns:
            Answer to the counterfactual question
        """
        target_objects = question_data.get('target_objects', [])
        question_text = question_data.get('question_text', '').lower()
        
        if not target_objects:
            return "unable to identify target object"
        
        removed_obj_idx = target_objects[0]
        
        original_collisions = self.detect_collisions(states)
        
        modified_states, modified_masks = self.remove_object(states, masks, removed_obj_idx)
        
        try:
            predicted_states = self.predict_future_states(modified_states, modified_masks)
            counterfactual_collisions = self.detect_collisions(predicted_states)
        except Exception:
            counterfactual_collisions = self.detect_collisions(modified_states)
        
        original_collision_pairs = set()
        for c in original_collisions:
            pair = tuple(sorted(c['objects']))
            if removed_obj_idx not in pair:
                original_collision_pairs.add(pair)
        
        counterfactual_collision_pairs = set()
        for c in counterfactual_collisions:
            pair = tuple(sorted(c['objects']))
            counterfactual_collision_pairs.add(pair)
        
        prevented_collisions = original_collision_pairs - counterfactual_collision_pairs
        new_collisions = counterfactual_collision_pairs - original_collision_pairs
        
        if 'collision' in question_text or 'collide' in question_text:
            if prevented_collisions:
                return f"yes, {len(prevented_collisions)} collision(s) would be prevented"
            elif new_collisions:
                return f"no, but {len(new_collisions)} new collision(s) would occur"
            else:
                return "no significant change in collisions"
        
        if 'happen' in question_text or 'change' in question_text:
            changes = []
            if prevented_collisions:
                changes.append(f"{len(prevented_collisions)} collision(s) prevented")
            if new_collisions:
                changes.append(f"{len(new_collisions)} new collision(s)")
            
            if changes:
                return "; ".join(changes)
            return "no significant changes"
        
        return "counterfactual scenario analyzed"
    
    def compare_trajectories(
        self,
        original_states: np.ndarray,
        counterfactual_states: np.ndarray,
        object_idx: int
    ) -> Dict[str, Any]:
        """
        Compare trajectories between original and counterfactual scenarios.
        
        Args:
            original_states: Original state tensor
            counterfactual_states: Counterfactual state tensor
            object_idx: Object to compare
            
        Returns:
            Dictionary with trajectory comparison metrics
        """
        orig_pos = original_states[:, object_idx, 0:3]
        cf_pos = counterfactual_states[:, object_idx, 0:3]
        
        position_diff = np.linalg.norm(orig_pos - cf_pos, axis=1)
        
        orig_vel = original_states[:, object_idx, 3:6]
        cf_vel = counterfactual_states[:, object_idx, 3:6]
        velocity_diff = np.linalg.norm(orig_vel - cf_vel, axis=1)
        
        return {
            'max_position_deviation': float(np.max(position_diff)),
            'mean_position_deviation': float(np.mean(position_diff)),
            'max_velocity_deviation': float(np.max(velocity_diff)),
            'mean_velocity_deviation': float(np.mean(velocity_diff)),
            'trajectory_diverged': bool(np.max(position_diff) > 0.5)
        }
