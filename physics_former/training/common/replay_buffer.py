"""Replay buffer for preventing catastrophic forgetting in curriculum learning."""
import random
from typing import List, Dict, Any
from collections import deque
import torch


class ReplayBuffer:
    """
    Stores past training examples to prevent catastrophic forgetting.
    Samples from buffer are mixed with current batch during training.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        replay_ratio: float = 0.2,
        sampling_strategy: str = 'uniform'
    ):
        """
        Args:
            max_size: Maximum number of samples to store
            replay_ratio: Fraction of batch to replace with replay samples (0.0-1.0)
            sampling_strategy: How to sample from buffer
                - 'uniform': Random uniform sampling
                - 'recent': Bias toward recent samples
                - 'diverse': Sample to maximize schema diversity
        """
        self.max_size = max_size
        self.replay_ratio = replay_ratio
        self.sampling_strategy = sampling_strategy
        
        self.buffer = deque(maxlen=max_size)
        self.schema_counts = {}  # Track schema distribution in buffer
        
    def add(self, batch: Dict[str, torch.Tensor], schema_ids: List[int] = None):
        """
        Add batch to replay buffer.
        
        Args:
            batch: Dictionary of tensors (states, masks, counts, etc.)
            schema_ids: Optional schema IDs for diversity tracking
        """
        batch_size = batch[list(batch.keys())[0]].size(0)
        
        # Convert batch to list of individual samples
        for i in range(batch_size):
            sample = {key: tensor[i].cpu() for key, tensor in batch.items()}
            
            # Add schema tracking
            if schema_ids is not None and i < len(schema_ids):
                sample['_schema_id'] = schema_ids[i]
                
                # Update schema counts
                schema_id = schema_ids[i]
                self.schema_counts[schema_id] = self.schema_counts.get(schema_id, 0) + 1
            
            self.buffer.append(sample)
        
        # If buffer is full, update schema counts
        if len(self.buffer) >= self.max_size:
            self._recount_schemas()
    
    def _recount_schemas(self):
        """Recount schema distribution in buffer (called when buffer is full)."""
        self.schema_counts = {}
        for sample in self.buffer:
            if '_schema_id' in sample:
                schema_id = sample['_schema_id']
                self.schema_counts[schema_id] = self.schema_counts.get(schema_id, 0) + 1
    
    def sample(self, n: int, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """
        Sample n examples from replay buffer.
        
        Args:
            n: Number of samples to draw
            device: Device to move tensors to
        
        Returns:
            Batch dictionary with n samples
        """
        if len(self.buffer) == 0:
            return None
        
        # Sample indices
        n = min(n, len(self.buffer))
        
        if self.sampling_strategy == 'uniform':
            indices = random.sample(range(len(self.buffer)), n)
        elif self.sampling_strategy == 'recent':
            # Bias toward recent samples (last 20% of buffer)
            recent_start = max(0, len(self.buffer) - len(self.buffer) // 5)
            indices = random.choices(range(recent_start, len(self.buffer)), k=n)
        elif self.sampling_strategy == 'diverse':
            # Sample to maximize schema diversity
            indices = self._sample_diverse(n)
        else:
            indices = random.sample(range(len(self.buffer)), n)
        
        # Gather samples
        samples = [self.buffer[i] for i in indices]
        
        # Collate into batch
        batch = {}
        for key in samples[0].keys():
            if key == '_schema_id':
                continue  # Skip metadata
            
            tensors = [sample[key] for sample in samples]
            batch[key] = torch.stack(tensors).to(device)
        
        return batch
    
    def _sample_diverse(self, n: int) -> List[int]:
        """Sample indices to maximize schema diversity."""
        if not self.schema_counts:
            return random.sample(range(len(self.buffer)), min(n, len(self.buffer)))
        
        # Calculate target samples per schema
        num_schemas = len(self.schema_counts)
        samples_per_schema = max(1, n // num_schemas)
        
        indices = []
        schema_samples = {schema_id: [] for schema_id in self.schema_counts.keys()}
        
        # Group buffer indices by schema
        for i, sample in enumerate(self.buffer):
            if '_schema_id' in sample:
                schema_id = sample['_schema_id']
                if schema_id in schema_samples:
                    schema_samples[schema_id].append(i)
        
        # Sample from each schema
        for schema_id, schema_indices in schema_samples.items():
            if len(schema_indices) > 0:
                k = min(samples_per_schema, len(schema_indices))
                indices.extend(random.sample(schema_indices, k))
        
        # Fill remaining with random samples
        while len(indices) < n and len(indices) < len(self.buffer):
            idx = random.randint(0, len(self.buffer) - 1)
            if idx not in indices:
                indices.append(idx)
        
        return indices[:n]
    
    def merge_with_batch(
        self,
        current_batch: Dict[str, torch.Tensor],
        device: str = 'cuda'
    ) -> Dict[str, torch.Tensor]:
        """
        Merge replay samples with current batch.
        
        Args:
            current_batch: Current training batch
            device: Device for tensors
        
        Returns:
            Merged batch with replay samples replacing some current samples
        """
        if len(self.buffer) == 0 or self.replay_ratio == 0:
            return current_batch
        
        batch_size = current_batch[list(current_batch.keys())[0]].size(0)
        n_replay = int(batch_size * self.replay_ratio)
        
        if n_replay == 0:
            return current_batch
        
        # Sample from replay buffer
        replay_batch = self.sample(n_replay, device=device)
        
        if replay_batch is None:
            return current_batch
        
        # Merge: keep (1 - replay_ratio) of current batch + replay samples
        n_current = batch_size - n_replay
        
        merged_batch = {}
        for key in current_batch.keys():
            if key in replay_batch:
                # Concatenate current[:n_current] + replay
                merged_batch[key] = torch.cat([
                    current_batch[key][:n_current],
                    replay_batch[key]
                ], dim=0)
            else:
                # Key not in replay (shouldn't happen, but handle gracefully)
                merged_batch[key] = current_batch[key]
        
        return merged_batch
    
    def __len__(self):
        return len(self.buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'utilization': len(self.buffer) / self.max_size,
            'num_schemas': len(self.schema_counts),
            'schema_distribution': dict(self.schema_counts)
        }
    
    def clear(self):
        """Clear the replay buffer."""
        self.buffer.clear()
        self.schema_counts.clear()


class AdaptiveReplayBuffer(ReplayBuffer):
    """
    Replay buffer with adaptive replay ratio based on forgetting detection.
    Increases replay ratio when forgetting is detected.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        initial_replay_ratio: float = 0.2,
        max_replay_ratio: float = 0.5,
        forgetting_threshold: float = 0.1,
        **kwargs
    ):
        """
        Args:
            max_size: Maximum buffer size
            initial_replay_ratio: Starting replay ratio
            max_replay_ratio: Maximum replay ratio
            forgetting_threshold: Loss increase threshold to trigger adaptation
        """
        super().__init__(max_size, initial_replay_ratio, **kwargs)
        self.initial_replay_ratio = initial_replay_ratio
        self.max_replay_ratio = max_replay_ratio
        self.forgetting_threshold = forgetting_threshold
        
        self.schema_losses = {}  # Track loss per schema
        self.loss_history = deque(maxlen=100)
    
    def update_loss(self, schema_id: int, loss: float):
        """Update loss tracking for a schema."""
        if schema_id not in self.schema_losses:
            self.schema_losses[schema_id] = deque(maxlen=10)
        
        self.schema_losses[schema_id].append(loss)
        self.loss_history.append(loss)
        
        # Check for forgetting
        self._check_forgetting()
    
    def _check_forgetting(self):
        """Check if forgetting is occurring and adapt replay ratio."""
        if len(self.loss_history) < 20:
            return
        
        # Compare recent loss to earlier loss
        recent_loss = sum(list(self.loss_history)[-10:]) / 10
        earlier_loss = sum(list(self.loss_history)[-20:-10]) / 10
        
        # If loss increased significantly, increase replay ratio
        if recent_loss > earlier_loss * (1 + self.forgetting_threshold):
            self.replay_ratio = min(
                self.replay_ratio * 1.2,
                self.max_replay_ratio
            )
            print(f"[REPLAY] Forgetting detected! Increased replay ratio to {self.replay_ratio:.2f}")
        
        # If loss is stable/decreasing, gradually reduce replay ratio
        elif recent_loss < earlier_loss * 0.95:
            self.replay_ratio = max(
                self.replay_ratio * 0.95,
                self.initial_replay_ratio
            )
