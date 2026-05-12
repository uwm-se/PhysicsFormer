"""
Standardized input preparation for PhysicsFormer model.

Provides consistent input formatting across training, validation, evaluation, and inference.
Ensures all inputs match expected tensor shapes and formats.
"""

import torch
import numpy as np
from typing import Dict, List, Union, Any, Optional


class ModelInputPreparator:
    """
    Standardizes input preparation for PhysicsFormer model.
    
    Handles conversion from various input formats (dicts, numpy arrays, lists)
    to the standardized tensor format expected by the model.
    """
    
    def __init__(self, config):
        """
        Initialize input preparator with model configuration.
        
        Args:
            config: Training configuration with model parameters
        """
        self.max_objects = config.max_objects
        self.state_dim = config.state_dim
        self.max_seq_length = config.max_seq_length
        self.device = getattr(config, 'device', 'cuda')
    
    def prepare_physics_input(
        self,
        raw_states: Union[List[Dict], np.ndarray, torch.Tensor],
        device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert raw physics states to model-ready tensors.
        
        Args:
            raw_states: Physics states in various formats:
                - List of dicts: [{'position': [x,y,z], 'velocity': [vx,vy,vz], 'mass': m}, ...]
                - Numpy array: [num_objects, state_dim] or [seq_len, num_objects, state_dim]
                - Torch tensor: [seq_len, num_objects, state_dim]
            device: Target device (default: from config)
            
        Returns:
            dict: {
                'object_states': [batch, seq, objects, state_dim],
                'object_mask': [batch, seq, objects]
            }
        """
        device = device or self.device
        
        # Convert to tensor
        if isinstance(raw_states, list) and len(raw_states) > 0 and isinstance(raw_states[0], dict):
            # List of dicts format
            states_tensor = self._convert_dict_list_to_tensor(raw_states)
        elif isinstance(raw_states, np.ndarray):
            states_tensor = torch.from_numpy(raw_states).float()
        elif isinstance(raw_states, torch.Tensor):
            states_tensor = raw_states.float()
        else:
            raise ValueError(
                f"Unsupported input format: {type(raw_states)}. "
                f"Expected list of dicts, numpy array, or torch tensor."
            )
        
        # Ensure correct shape: [batch, seq, objects, state_dim]
        if states_tensor.dim() == 2:
            # [objects, state_dim] -> [1, 1, objects, state_dim]
            states_tensor = states_tensor.unsqueeze(0).unsqueeze(0)
        elif states_tensor.dim() == 3:
            # [seq, objects, state_dim] -> [1, seq, objects, state_dim]
            states_tensor = states_tensor.unsqueeze(0)
        elif states_tensor.dim() != 4:
            raise ValueError(
                f"Invalid state tensor shape: {states_tensor.shape}. "
                f"Expected 2D, 3D, or 4D tensor."
            )
        
        batch_size, seq_len, num_objects, state_dim = states_tensor.shape
        
        # Validate dimensions
        if state_dim != self.state_dim:
            raise ValueError(
                f"State dimension mismatch: got {state_dim}, expected {self.state_dim}"
            )
        
        # Pad or truncate to max_objects
        if num_objects > self.max_objects:
            states_tensor = states_tensor[:, :, :self.max_objects, :]
            num_objects = self.max_objects
        elif num_objects < self.max_objects:
            padding = torch.zeros(
                batch_size, seq_len, self.max_objects - num_objects, state_dim
            )
            states_tensor = torch.cat([states_tensor, padding], dim=2)
        
        # Pad or truncate to max_seq_length
        if seq_len > self.max_seq_length:
            states_tensor = states_tensor[:, :self.max_seq_length, :, :]
            seq_len = self.max_seq_length
        elif seq_len < self.max_seq_length:
            padding = torch.zeros(
                batch_size, self.max_seq_length - seq_len, self.max_objects, state_dim
            )
            states_tensor = torch.cat([states_tensor, padding], dim=1)
        
        # Create object mask (1 for real objects, 0 for padding)
        object_mask = torch.ones(batch_size, self.max_seq_length, self.max_objects)
        if num_objects < self.max_objects:
            object_mask[:, :, num_objects:] = 0
        if seq_len < self.max_seq_length:
            object_mask[:, seq_len:, :] = 0
        
        # Move to device
        return {
            'object_states': states_tensor.to(device),
            'object_mask': object_mask.to(device)
        }
    
    def prepare_counting_input(
        self,
        object_states: torch.Tensor,
        device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare counting task input from physics states.
        
        Args:
            object_states: [batch, seq, objects, state_dim]
            device: Target device
            
        Returns:
            dict: Model input for counting task
        """
        device = device or self.device
        
        if object_states.dim() != 4:
            raise ValueError(
                f"Invalid object_states shape: {object_states.shape}. "
                f"Expected [batch, seq, objects, state_dim]"
            )
        
        batch_size, seq_len, num_objects, state_dim = object_states.shape
        
        # Create mask based on non-zero states
        object_mask = (object_states.abs().sum(dim=-1) > 0).float()
        
        return {
            'object_states': object_states.to(device),
            'object_mask': object_mask.to(device)
        }
    
    def prepare_arithmetic_input(
        self,
        numbers: List[int],
        operations: List[str],
        device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare arithmetic task input.
        
        Args:
            numbers: List of numbers [num1, num2, ...]
            operations: List of operations ['add', 'subtract', ...]
            device: Target device
            
        Returns:
            dict: Model input for arithmetic task
        """
        device = device or self.device
        
        # Convert numbers to tensor
        numbers_tensor = torch.tensor(numbers, dtype=torch.long).unsqueeze(0)  # [1, num_count]
        
        return {
            'numbers': numbers_tensor.to(device),
            'operations': operations  # Keep as list of strings
        }
    
    def validate_input_shape(
        self,
        batch: Dict[str, Any],
        expected_mode: str
    ) -> None:
        """
        Validate that input shapes match model expectations.
        
        Args:
            batch: Input batch dict
            expected_mode: Expected task mode ('physics', 'counting', 'arithmetic', 'symbolic')
            
        Raises:
            ValueError: If shapes don't match expectations
        """
        if expected_mode in ['physics', 'counting']:
            if 'object_states' not in batch:
                raise ValueError(f"Missing 'object_states' for {expected_mode} task")
            
            states = batch['object_states']
            if states.dim() != 4:
                raise ValueError(
                    f"Invalid object_states shape: {states.shape}. "
                    f"Expected [batch, seq, objects, state_dim]"
                )
            
            batch_size, seq_len, num_objects, state_dim = states.shape
            
            if state_dim != self.state_dim:
                raise ValueError(
                    f"State dimension mismatch: got {state_dim}, expected {self.state_dim}"
                )
            
            if num_objects != self.max_objects:
                raise ValueError(
                    f"Object count mismatch: got {num_objects}, expected {self.max_objects}"
                )
            
            if seq_len > self.max_seq_length:
                raise ValueError(
                    f"Sequence length exceeds maximum: got {seq_len}, max {self.max_seq_length}"
                )
            
            if 'object_mask' in batch:
                mask = batch['object_mask']
                if mask.shape != (batch_size, seq_len, num_objects):
                    raise ValueError(
                        f"Mask shape mismatch: got {mask.shape}, "
                        f"expected [{batch_size}, {seq_len}, {num_objects}]"
                    )
        
        elif expected_mode in ['arithmetic', 'symbolic']:
            if 'numbers' not in batch:
                raise ValueError(f"Missing 'numbers' for {expected_mode} task")
            
            if 'operations' not in batch:
                raise ValueError(f"Missing 'operations' for {expected_mode} task")
    
    def _convert_dict_list_to_tensor(self, dict_list: List[Dict]) -> torch.Tensor:
        """
        Convert list of state dicts to tensor.
        
        Args:
            dict_list: [{'position': [x,y,z], 'velocity': [vx,vy,vz], ...}, ...]
            
        Returns:
            torch.Tensor: [num_objects, state_dim]
        """
        states = []
        
        for obj_dict in dict_list:
            # Extract state components
            state_vector = []
            
            # Position (3D)
            if 'position' in obj_dict:
                pos = obj_dict['position']
                if isinstance(pos, (list, tuple)):
                    state_vector.extend(pos[:3])
                else:
                    state_vector.extend([pos, 0, 0])
            else:
                state_vector.extend([0, 0, 0])
            
            # Velocity (3D)
            if 'velocity' in obj_dict:
                vel = obj_dict['velocity']
                if isinstance(vel, (list, tuple)):
                    state_vector.extend(vel[:3])
                else:
                    state_vector.extend([vel, 0, 0])
            else:
                state_vector.extend([0, 0, 0])
            
            # Mass (1D)
            if 'mass' in obj_dict:
                state_vector.append(obj_dict['mass'])
            else:
                state_vector.append(1.0)
            
            # Pad to state_dim if needed
            while len(state_vector) < self.state_dim:
                state_vector.append(0.0)
            
            # Truncate if too long
            state_vector = state_vector[:self.state_dim]
            
            states.append(state_vector)
        
        return torch.tensor(states, dtype=torch.float32)
    
    def create_minimal_test_batch(
        self,
        mode: str,
        batch_size: int = 1,
        device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Create minimal valid test batch for a given mode.
        
        Useful for testing and debugging.
        
        Args:
            mode: Task mode ('physics', 'counting', 'arithmetic', 'symbolic')
            batch_size: Batch size
            device: Target device
            
        Returns:
            dict: Minimal valid input batch
        """
        device = device or self.device
        
        if mode in ['physics', 'counting']:
            return {
                'object_states': torch.randn(
                    batch_size, self.max_seq_length, self.max_objects, self.state_dim
                ).to(device),
                'object_mask': torch.ones(
                    batch_size, self.max_seq_length, self.max_objects
                ).to(device)
            }
        
        elif mode in ['arithmetic', 'symbolic']:
            return {
                'numbers': torch.randint(1, 100, (batch_size, 2)).to(device),
                'operations': ['add'] * batch_size
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
