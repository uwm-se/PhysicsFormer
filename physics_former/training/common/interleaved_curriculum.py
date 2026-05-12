"""Interleaved curriculum training to prevent catastrophic forgetting."""
import random
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader

from ..constants import *


class InterleavedCurriculumDataset(Dataset):
    """
    Dataset that samples from multiple schemas simultaneously with configurable weights.
    Prevents catastrophic forgetting by always including samples from all difficulty levels.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        schema_difficulty_map: Dict[str, str],
        difficulty_weights: Dict[str, float] = None,
        samples_per_epoch: int = 10000,
        seed: int = 42
    ):
        """
        Args:
            base_dataset: Underlying dataset (e.g., CountingDataset)
            schema_difficulty_map: Map schema name → difficulty level ('easy', 'medium', 'hard')
            difficulty_weights: Sampling weights for each difficulty
                Default: {'easy': 0.2, 'medium': 0.3, 'hard': 0.5}
            samples_per_epoch: Total samples per epoch
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.schema_difficulty_map = schema_difficulty_map
        self.samples_per_epoch = samples_per_epoch
        
        if difficulty_weights is None:
            # Default: emphasize hard schemas
            difficulty_weights = {'easy': 0.2, 'medium': 0.3, 'hard': 0.5}
        self.difficulty_weights = difficulty_weights
        
        # Group indices by difficulty
        self.difficulty_indices = {'easy': [], 'medium': [], 'hard': []}
        self._build_difficulty_groups()
        
        # Pre-sample indices for this epoch
        random.seed(seed)
        self.epoch_indices = self._sample_epoch_indices()
        
    def _build_difficulty_groups(self):
        """Group dataset indices by schema difficulty."""
        if not hasattr(self.base_dataset, 'valid_indices'):
            # If base dataset doesn't have valid_indices, use all indices
            valid_indices = list(range(len(self.base_dataset)))
        else:
            valid_indices = self.base_dataset.valid_indices
        
        # Get schema names
        if hasattr(self.base_dataset, 'physics_dataset'):
            schema_names = sorted(list(self.base_dataset.physics_dataset.schema_names))
        else:
            # Fallback: assume indices map to schemas sequentially
            schema_names = list(self.schema_difficulty_map.keys())
        
        # Group indices by difficulty
        for idx in valid_indices:
            # Determine schema from index
            file_idx = idx // 50000  # Assuming 50K episodes per schema
            if file_idx < len(schema_names):
                schema = schema_names[file_idx]
                difficulty = self.schema_difficulty_map.get(schema, 'medium')
                self.difficulty_indices[difficulty].append(idx)
        
        print(f"[INTERLEAVED] Difficulty distribution:")
        for diff, indices in self.difficulty_indices.items():
            print(f"  {diff.capitalize()}: {len(indices):,} samples")
    
    def _sample_epoch_indices(self) -> List[int]:
        """Sample indices for one epoch according to difficulty weights."""
        epoch_indices = []
        
        for difficulty, weight in self.difficulty_weights.items():
            n_samples = int(self.samples_per_epoch * weight)
            indices = self.difficulty_indices[difficulty]
            
            if len(indices) == 0:
                continue
            
            # Sample with replacement if needed
            sampled = random.choices(indices, k=n_samples)
            epoch_indices.extend(sampled)
        
        # Shuffle to mix difficulties
        random.shuffle(epoch_indices)
        return epoch_indices
    
    def resample_epoch(self, seed: int = None):
        """Resample indices for a new epoch."""
        if seed is not None:
            random.seed(seed)
        self.epoch_indices = self._sample_epoch_indices()
    
    def __len__(self):
        return len(self.epoch_indices)
    
    def __getitem__(self, idx):
        """Get item from base dataset using pre-sampled index."""
        base_idx = self.epoch_indices[idx]
        
        # Map back to base dataset index
        if hasattr(self.base_dataset, 'valid_indices'):
            # Find position in valid_indices
            try:
                dataset_idx = self.base_dataset.valid_indices.index(base_idx)
            except ValueError:
                # Fallback: use modulo
                dataset_idx = idx % len(self.base_dataset)
        else:
            dataset_idx = base_idx % len(self.base_dataset)
        
        return self.base_dataset[dataset_idx]


# Schema difficulty classification
SCHEMA_DIFFICULTY_MAP = {
    # Easy (Level 0-2): Basic physics, few objects
    'projectile_motion': 'easy',
    'pendulum_simple': 'easy',
    'collision_elastic': 'easy',
    'collision_inelastic': 'easy',
    'friction_sliding': 'easy',
    'containment': 'easy',
    'force_motion': 'easy',
    'blockage': 'easy',
    'support': 'easy',
    'path': 'easy',
    'balance': 'easy',
    'numerical': 'easy',
    
    # Medium (Level 3-5): Moderate complexity, more objects
    'spring_oscillation': 'medium',
    'orbit_circular': 'medium',
    'orbit_elliptical': 'medium',
    'damped_oscillation': 'medium',
    'driven_oscillation': 'medium',
    'resonance': 'medium',
    'coupled_oscillators': 'medium',
    'wave_propagation': 'medium',
    'standing_wave': 'medium',
    'interference_constructive': 'medium',
    'interference_destructive': 'medium',
    'diffraction': 'medium',
    'reflection': 'medium',
    'refraction': 'medium',
    'doppler_effect': 'medium',
    'conservation_momentum': 'medium',
    'conservation_energy': 'medium',
    'conservation_angular_momentum': 'medium',
    
    # Hard (Level 6-8): Complex dynamics, many objects, chaos
    'chaos_double_pendulum': 'hard',
    'chaos_three_body': 'hard',
    'multi_scale_interaction': 'hard',
    'emergent_pattern': 'hard',
    'phase_transition': 'hard',
    'critical_point': 'hard',
    'symmetry_breaking': 'hard',
    'hysteresis': 'hard',
    'bifurcation': 'hard',
    'attractor': 'hard',
    'strange_attractor': 'hard',
    'fractal_dimension': 'hard',
    'self_organization': 'hard',
    'adaptation': 'hard',
    'learning': 'hard',
    'memory': 'hard',
    'anticipation': 'hard',
    'goal_directed': 'hard',
    'tool_use': 'hard',
    'cooperation': 'hard',
    'competition': 'hard',
    'predator_prey': 'hard',
    'exponential_growth': 'hard',
}


def create_interleaved_dataloader(
    base_dataset: Dataset,
    batch_size: int,
    difficulty_weights: Dict[str, float] = None,
    samples_per_epoch: int = 10000,
    num_workers: int = 0,
    seed: int = 42
) -> DataLoader:
    """
    Create a DataLoader with interleaved curriculum sampling.
    
    Args:
        base_dataset: Base dataset (e.g., CountingDataset)
        batch_size: Batch size
        difficulty_weights: Sampling weights {'easy': 0.2, 'medium': 0.3, 'hard': 0.5}
        samples_per_epoch: Total samples per epoch
        num_workers: Number of data loading workers
        seed: Random seed
    
    Returns:
        DataLoader with interleaved sampling
    
    Example:
        >>> counting_dataset = CountingDataset(...)
        >>> train_loader = create_interleaved_dataloader(
        ...     counting_dataset,
        ...     batch_size=48,
        ...     difficulty_weights={'easy': 0.2, 'medium': 0.3, 'hard': 0.5}
        ... )
    """
    interleaved_dataset = InterleavedCurriculumDataset(
        base_dataset=base_dataset,
        schema_difficulty_map=SCHEMA_DIFFICULTY_MAP,
        difficulty_weights=difficulty_weights,
        samples_per_epoch=samples_per_epoch,
        seed=seed
    )
    
    return DataLoader(
        interleaved_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
