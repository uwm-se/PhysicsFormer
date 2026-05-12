"""
Batch Processor

Handles batch preparation and data movement.
"""

import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes and prepares batches for training."""
    
    def __init__(self, device, task_specific_prep: Dict[str, callable] = None):
        self.device = device
        self.task_specific_prep = task_specific_prep or {}
    
    def prepare_batch(self, batch: Dict, task_name: str) -> Dict:
        """
        Prepare batch for training.
        
        Args:
            batch: Raw batch from dataloader
            task_name: Task name for task-specific preparation
            
        Returns:
            Prepared batch ready for model
        """
        # Move to device
        batch = self.move_to_device(batch)
        
        # Task-specific preparation
        if task_name in self.task_specific_prep:
            batch = self.task_specific_prep[task_name](batch)
        
        return batch
    
    def move_to_device(self, batch: Dict) -> Dict:
        """Move all tensors in batch to device."""
        return {
            k: self._move_value(v)
            for k, v in batch.items()
        }
    
    def _move_value(self, value: Any) -> Any:
        """Move a single value to device if it's a tensor."""
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        elif isinstance(value, dict):
            return {k: self._move_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return type(value)(self._move_value(v) for v in value)
        else:
            return value
    
    def prepare_physics_batch(self, batch: Dict) -> Dict:
        """Physics-specific batch preparation."""
        # Extract next states for prediction target
        if 'object_states' in batch and batch['object_states'].dim() == 4:
            # Sequence format: [batch, seq, objects, features]
            # Target is next timestep
            batch['target_states'] = batch['object_states'][:, 1:, :, :]
            batch['current_states'] = batch['object_states'][:, :-1, :, :]
        
        return batch
    
    def prepare_counting_batch(self, batch: Dict) -> Dict:
        """Counting-specific batch preparation."""
        # Ensure counts are long tensors for cross-entropy
        if 'counts' in batch:
            batch['counts'] = batch['counts'].long()
        
        return batch
    
    def prepare_arithmetic_batch(self, batch: Dict) -> Dict:
        """Arithmetic-specific batch preparation."""
        # Convert results to appropriate format
        if 'results' in batch and not isinstance(batch['results'], torch.Tensor):
            batch['results'] = torch.tensor(batch['results'], device=self.device)
        
        return batch
    
    def prepare_symbolic_batch(self, batch: Dict) -> Dict:
        """Symbolic-specific batch preparation."""
        # Similar to arithmetic
        return self.prepare_arithmetic_batch(batch)


# Default task-specific preparation functions
DEFAULT_TASK_PREP = {
    'physics': lambda bp, batch: bp.prepare_physics_batch(batch),
    'counting': lambda bp, batch: bp.prepare_counting_batch(batch),
    'arithmetic': lambda bp, batch: bp.prepare_arithmetic_batch(batch),
    'symbolic': lambda bp, batch: bp.prepare_symbolic_batch(batch)
}
