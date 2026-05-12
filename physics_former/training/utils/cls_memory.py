"""
Complementary Learning Systems (CLS) Memory Components

Implements hippocampal-like episodic memory buffers for experience replay,
based on McClelland, McNaughton & O'Reilly (1995) CLS theory.
"""

import torch
import random
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any


class EpisodicBuffer:
    """
    Hippocampal-like episodic memory buffer.
    
    Stores recent experiences for replay during consolidation.
    Implements reservoir sampling for efficient memory management.
    
    Based on:
    - McClelland et al. (1995): Complementary Learning Systems
    - Wilson & McNaughton (1994): Reactivation of hippocampal ensemble memories
    """
    
    def __init__(self, capacity: int = 10000, device: str = 'cpu'):
        """
        Initialize episodic buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            device: Device for tensor storage
        """
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
        self.full = False
    
    def add(self, experience: Dict[str, torch.Tensor]):
        """
        Add experience to buffer (hippocampal encoding).
        
        Uses reservoir sampling to maintain representative sample.
        
        Args:
            experience: Dictionary of tensors (batch)
        """
        # Detach and move to CPU for storage (like long-term memory)
        stored_experience = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in experience.items()
        }
        
        # SAFEGUARD: Estimate memory size and warn if too large
        # This prevents memory explosion from storing huge batches
        total_elements = sum(
            v.numel() if isinstance(v, torch.Tensor) else 0
            for v in stored_experience.values()
        )
        memory_mb = (total_elements * 2) / (1024 ** 2)  # Assume FP16 (2 bytes per element)
        
        if memory_mb > 100:  # Warn if single experience > 100MB
            print(f"WARNING: Large experience ({memory_mb:.1f}MB) being stored in CLS memory")
            print(f"  Keys: {list(stored_experience.keys())}")
            print(f"  Consider filtering to essential keys only")
        
        if len(self.buffer) < self.capacity:
            # Buffer not full yet
            self.buffer.append(stored_experience)
        else:
            # Reservoir sampling: randomly replace old experience
            # This maintains representative distribution
            if not self.full:
                self.full = True
            
            # Random replacement (prevents recency bias)
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = stored_experience
    
    def sample(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """
        Sample experiences for replay (hippocampal reactivation).
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Batch of experiences moved to device
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Random sampling (like sleep replay)
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Combine into batch
        batch = {}
        for key in samples[0].keys():
            values = [s[key] for s in samples]
            if isinstance(values[0], torch.Tensor):
                # Stack tensors and move to device
                batch[key] = torch.cat(values, dim=0).to(self.device)
            else:
                batch[key] = values
        
        return batch
    
    def __len__(self):
        return len(self.buffer)
    
    def is_full(self):
        return self.full


class HippocampalSystem:
    """
    Multi-task hippocampal memory system.
    
    Manages separate episodic buffers for different task types,
    enabling task-specific consolidation during replay.
    """
    
    def __init__(
        self,
        tasks: List[str] = ['physics', 'counting', 'arithmetic', 'symbolic'],
        capacity_per_task: int = 10000,
        device: str = 'cpu'
    ):
        """
        Initialize hippocampal system.
        
        Args:
            tasks: List of task names
            capacity_per_task: Buffer capacity for each task
            device: Device for tensor storage
        """
        self.tasks = tasks
        self.buffers = {
            task: EpisodicBuffer(capacity=capacity_per_task, device=device)
            for task in tasks
        }
        self.device = device
    
    def encode(self, experience: Dict[str, torch.Tensor], task: str):
        """
        Encode experience into task-specific buffer.
        
        Args:
            experience: Experience dictionary
            task: Task name
        """
        if task not in self.buffers:
            raise ValueError(f"Unknown task: {task}")
        
        self.buffers[task].add(experience)
    
    def replay(
        self,
        task: str,
        batch_size: int = 1
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Replay experiences from task buffer.
        
        Args:
            task: Task to replay
            batch_size: Number of experiences
        
        Returns:
            Batch of experiences or None if buffer empty
        """
        if task not in self.buffers:
            raise ValueError(f"Unknown task: {task}")
        
        if len(self.buffers[task]) == 0:
            return None
        
        return self.buffers[task].sample(batch_size)
    
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        return {
            task: len(buffer)
            for task, buffer in self.buffers.items()
        }


class ConsolidationScheduler:
    """
    Manages consolidation schedule (sleep-like replay).
    
    Implements spacing and interleaving based on cognitive science:
    - Ebbinghaus (1885): Spacing effect
    - Rohrer & Taylor (2007): Interleaved practice
    """
    
    def __init__(
        self,
        consolidation_frequency: float = 0.2,
        task_weights: Optional[Dict[str, float]] = None,
        adaptive: bool = True
    ):
        """
        Initialize consolidation scheduler.
        
        Args:
            consolidation_frequency: Probability of consolidation per batch (0.2 = 20%)
            task_weights: Relative importance of each task for sampling
            adaptive: Adjust weights based on forgetting
        """
        self.consolidation_frequency = consolidation_frequency
        self.task_weights = task_weights or {}
        self.adaptive = adaptive
        self.consolidation_count = {}
        self.task_performance = {}
    
    def should_consolidate(self) -> bool:
        """
        Decide whether to consolidate (like REM sleep cycles).
        
        Returns:
            True if should consolidate this step
        """
        return random.random() < self.consolidation_frequency
    
    def select_task(self, available_tasks: List[str]) -> str:
        """
        Select which task to consolidate.
        
        Uses weighted sampling, prioritizing:
        1. Tasks with lower performance (adaptive)
        2. Tasks with less recent consolidation
        3. User-specified weights
        
        Args:
            available_tasks: Tasks with available replay data
        
        Returns:
            Selected task name
        """
        if not available_tasks:
            raise ValueError("No tasks available for consolidation")
        
        # Get weights
        weights = []
        for task in available_tasks:
            weight = self.task_weights.get(task, 1.0)
            
            # Adaptive: prioritize tasks with lower performance
            if self.adaptive and task in self.task_performance:
                # Lower performance -> higher weight
                performance = self.task_performance[task]
                weight *= (1.0 - performance + 0.1)  # +0.1 to avoid zero
            
            weights.append(weight)
        
        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Sample
        selected = random.choices(available_tasks, weights=weights, k=1)[0]
        
        # Track consolidation
        self.consolidation_count[selected] = self.consolidation_count.get(selected, 0) + 1
        
        return selected
    
    def update_performance(self, task: str, accuracy: float):
        """
        Update task performance for adaptive scheduling.
        
        Args:
            task: Task name
            accuracy: Current accuracy (0-1)
        """
        self.task_performance[task] = accuracy
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        return {
            'consolidation_count': self.consolidation_count.copy(),
            'task_performance': self.task_performance.copy()
        }


class CLSMemorySystem:
    """
    Complete Complementary Learning Systems memory implementation.
    
    Combines:
    - Hippocampal episodic buffers
    - Consolidation scheduling
    - Multi-task replay
    
    Based on McClelland et al. (1995) CLS theory.
    """
    
    def __init__(
        self,
        tasks: List[str] = ['physics', 'counting', 'arithmetic', 'symbolic'],
        capacity_per_task: int = 10000,
        consolidation_frequency: float = 0.2,
        device: str = 'cpu',
        adaptive: bool = True
    ):
        """
        Initialize CLS memory system.
        
        Args:
            tasks: List of task names
            capacity_per_task: Episodic buffer capacity per task
            consolidation_frequency: Replay frequency (0.2 = 20%)
            device: Device for tensors
            adaptive: Use adaptive consolidation
        """
        self.hippocampus = HippocampalSystem(
            tasks=tasks,
            capacity_per_task=capacity_per_task,
            device=device
        )
        
        self.scheduler = ConsolidationScheduler(
            consolidation_frequency=consolidation_frequency,
            adaptive=adaptive
        )
        
        self.device = device
        self.tasks = tasks
        
        # Store for easy access
        self.consolidation_frequency = consolidation_frequency
        self.adaptive = adaptive
    
    def encode_experience(self, experience: Dict[str, torch.Tensor], task: str):
        """
        Encode experience into hippocampal buffer.
        
        Args:
            experience: Experience dictionary
            task: Task name
        """
        self.hippocampus.encode(experience, task)
    
    def consolidate(
        self,
        exclude_tasks: Optional[List[str]] = None,
        batch_size: int = 1
    ) -> Optional[Tuple[str, Dict[str, torch.Tensor]]]:
        """
        Perform consolidation step (replay).
        
        Args:
            exclude_tasks: Tasks to exclude from consolidation
            batch_size: Replay batch size
        
        Returns:
            (task_name, replay_batch) or None if no consolidation
        """
        # Check if should consolidate
        if not self.scheduler.should_consolidate():
            return None
        
        # Get available tasks
        exclude_tasks = exclude_tasks or []
        available_tasks = [
            task for task in self.tasks
            if task not in exclude_tasks and len(self.hippocampus.buffers[task]) > 0
        ]
        
        if not available_tasks:
            return None
        
        # Select task to consolidate
        task = self.scheduler.select_task(available_tasks)
        
        # Replay experiences
        replay_batch = self.hippocampus.replay(task, batch_size)
        
        return task, replay_batch
    
    def update_performance(self, task: str, accuracy: float):
        """
        Update task performance for adaptive consolidation.
        
        Args:
            task: Task name
            accuracy: Current accuracy (0-1)
        """
        self.scheduler.update_performance(task, accuracy)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'hippocampus': self.hippocampus.get_stats(),
            'scheduler': self.scheduler.get_stats()
        }
    
    def get_memory_sizes(self) -> Dict[str, int]:
        """Get memory buffer sizes for each task."""
        return self.hippocampus.get_stats()
    
    def save(self, path: str):
        """Save memory system state using atomic write to prevent corruption."""
        from pathlib import Path
        import tempfile
        import shutil
        
        state = {
            'buffers': {
                task: buffer.buffer
                for task, buffer in self.hippocampus.buffers.items()
            },
            'scheduler_stats': self.scheduler.get_stats()
        }
        
        # Use atomic write pattern: write to temp file, then rename
        # This prevents corruption if process is killed during save
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write to temporary file in same directory (ensures same filesystem)
            with tempfile.NamedTemporaryFile(
                mode='wb',
                dir=path_obj.parent,
                delete=False,
                suffix='.tmp'
            ) as tmp_file:
                tmp_path = tmp_file.name
                torch.save(state, tmp_file)
            
            # Atomic rename (overwrites existing file on POSIX, may fail on Windows if file exists)
            try:
                Path(tmp_path).replace(path_obj)
            except OSError:
                # Windows: remove target first if it exists
                if path_obj.exists():
                    path_obj.unlink()
                Path(tmp_path).replace(path_obj)
                
        except MemoryError as e:
            # If we hit memory error, reduce buffer sizes and raise
            print(f"ERROR: Memory error saving CLS memory. Buffer sizes:")
            for task, buffer in self.hippocampus.buffers.items():
                print(f"  {task}: {len(buffer.buffer)} experiences")
            print(f"Consider reducing capacity_per_task in config.")
            raise
        except Exception as e:
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and Path(tmp_path).exists():
                Path(tmp_path).unlink()
            raise
    
    def load(self, path: str):
        """Load memory system state."""
        state = torch.load(path, map_location='cpu')
        
        # Restore buffers
        for task, buffer_data in state['buffers'].items():
            self.hippocampus.buffers[task].buffer = buffer_data
        
        # Restore scheduler stats
        scheduler_stats = state['scheduler_stats']
        self.scheduler.consolidation_count = scheduler_stats.get('consolidation_count', {})
        self.scheduler.task_performance = scheduler_stats.get('task_performance', {})


__all__ = [
    'EpisodicBuffer',
    'HippocampalSystem',
    'ConsolidationScheduler',
    'CLSMemorySystem',
]
