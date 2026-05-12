"""
Training Configuration for PhysicsFormer - A100 80GB

Optimized for NVIDIA A100 80GB GPU
Target Training Time: ~6-12 hours (vs 3-5 days on 12GB GPU)

Key Optimizations (A100 80GB):
- hidden_dim: 768 - Larger model capacity
- num_layers: 8 - Deeper network
- num_heads: 24 - More attention heads (head_dim=32)
- batch_size: 64 - 8x larger than 12GB config
- max_seq_length: 512 - Full sequences, no chunking needed
- num_workers: 8 - Full multiprocessing on Linux
- No chunking needed - A100 can handle full pairwise matrices

GPU Memory Breakdown (A100 80GB, FP16 mixed precision):
- Model (FP16): ~400 MB (150M parameters)
- Optimizer (FP32): ~1.6 GB (AdamW states)
- Gradients (FP16): ~400 MB
- Pairwise embeddings: ~13 GB (batch=64 * seq=512 * 20² * 768 * 2 bytes)
- Activations & workspace: ~20-30 GB
- **Peak GPU: ~40-50 GB** (safe for 80GB with 30GB margin)

Training Time Estimate (A100 80GB):
- Physics: 30K episodes × 50 epochs ÷ batch_size=64 → ~4-6 hours
- Total with all stages: ~6-12 hours
"""

import os
from pathlib import Path
import torch


class A100Config:
    """Configuration for A100 80GB - Maximum performance."""
    
    # Model Architecture (Larger for A100)
    state_dim = 35            # 35D comprehensive physics state
    hidden_dim = 768          # LARGER: More capacity (was 512)
    num_layers = 8            # DEEPER: More layers (was 6)
    num_heads = 24            # MORE HEADS: Better attention (was 16)
    ff_dim = 3072             # LARGER: 4x hidden_dim
    max_objects = 20          # Keep at 20 (quadratic scaling)
    num_schema_classes = 37   # 37 Isaac Sim physics schemas
    vocab_size = 100          # For symbolic math
    dropout = 0.1             # REDUCED: Less regularization with more data throughput
    label_smoothing = 0.1     # Smooth labels
    weight_decay = 0.01       # L2 regularization
    
    # Schema Curriculum
    schema_curriculum_level = 1
    counting_schema_level = 8
    
    # Training (MAXIMIZED for A100 80GB)
    batch_size = 64               # 8x larger than 12GB config
    batch_size_physics = 64       # Physics Stage 1
    batch_size_physics_long = 48  # Longer sequences
    batch_size_consolidation = 128  # Consolidation replay
    accumulation_steps = 1        # No accumulation needed
    effective_batch_size = 64
    
    # Optimization (Tuned for stability)
    learning_rate = 1e-4          # REDUCED: Prevent NaN explosion in deep layers
    weight_decay = 0.01           # Standard weight decay
    max_grad_norm = 0.5           # TIGHTER: Prevent gradient explosion
    max_grad_value = 0.5          # TIGHTER: Per-param clipping
    warmup_steps = 1000           # Shorter warmup with large batch
    lr_scheduler = 'cosine'       # Cosine annealing
    min_lr = 1e-6                 # Higher minimum LR
    weight_init_scale = 0.02      # Initialization scale
    
    # Loss Weights (FIXED for stable training)
    prediction_weight = 1.0
    threshold_weight = 0.05       # REDUCED: Less pressure on threshold
    energy_weight = 0.001         # MUCH LOWER: Prevent gradient explosion
    momentum_weight = 0.001       # MUCH LOWER: Prevent gradient explosion
    counting_weight = 1.0
    arithmetic_weight = 0.5
    symbolic_weight = 1.0
    
    # Sequence Processing (NO CHUNKING - A100 can handle it)
    max_seq_length = 512          # Full sequences
    encoder_chunk_size = 512      # No chunking needed
    seq_chunk_size = 512          # No chunking needed
    
    # Data Loading (FULL MULTIPROCESSING on Linux)
    num_workers = 8               # Full parallel loading
    prefetch_factor = 4           # Prefetch batches
    pin_memory = True             # Faster GPU transfer
    persistent_workers = True     # Keep workers alive
    
    # Mixed Precision (BF16 preferred on A100)
    mixed_precision = True
    use_bf16 = True               # A100 has native BF16 support
    
    # Memory Optimization
    compile_model = True          # torch.compile for speed
    cache_dataset_to_ram = True   # Cache full dataset in RAM
    
    # Epochs (INCREASED to compensate for fewer gradient updates with large batch)
    # With batch=64 vs batch=8, we get 8x fewer updates per epoch
    # So we need more epochs to reach similar total gradient updates
    physics_epochs = 400          # COMPLETE: Full curriculum with patience (13 × 30 epochs)
    
    # Validation
    val_frequency = 1             # Validate every epoch
    val_split = 0.1               # 10% validation
    
    # Checkpointing
    checkpoint_frequency = 5      # Save every 5 epochs
    keep_last_n_checkpoints = 3   # Keep last 3
    
    # Early Stopping
    use_early_stopping = True     # Enable early stopping
    early_stopping_patience = 10  # More patience with large batch
    early_stopping_min_delta = 1e-4
    patience = 20                 # General patience for training
    min_delta = 1e-5              # Minimum improvement threshold
    save_every = 5                # Save checkpoint every 5 epochs
    
    # Data Paths (Colab/Cloud)
    data_dir = '/content/drive/MyDrive/physics_hdf5'
    checkpoint_dir = '/content/drive/MyDrive/physics_checkpoints'
    
    # Max episodes per file for RAM caching
    max_episodes_per_file = 1000  # Load all episodes
    
    # Device
    device = 'cuda'
    
    # Debug
    debug = False                 # Disable for speed
    
    # Modern Improvements (ALL ENABLED)
    use_rmsnorm = True
    use_flash_attention = True
    use_rope = True
    use_swiglu = True
    use_ema = True
    ema_decay = 0.999
    use_cosine_schedule = True
    warmup_ratio = 0.05
    min_lr_ratio = 0.1
    use_data_augmentation = True
    augment_permute_prob = 0.3
    augment_noise_std = 0.01
    augment_time_reverse_prob = 0.1
    use_multi_scale_loss = True
    temporal_horizons = [1, 5, 10]
    temporal_weights = [1.0, 0.5, 0.25]
    use_contrastive_loss = False
    contrastive_temperature = 0.07
    contrastive_weight = 0.1
    use_label_smoothing = True
    physics_label_smoothing = 0.1
    
    # Curriculum Learning
    curriculum_min_epochs = 5     # Faster progression
    curriculum_max_epochs = 30    # HARD LIMIT: 400 epochs / 13 schemas ≈ 30 epochs per schema
    curriculum_patience = 3       # Less patience
    curriculum_improvement_threshold = 0.02
    curriculum_min_accuracy = 0.1  # RELAXED: Allow progression with 10% accuracy
    
    # Threshold-aware loss parameters (RELAXED for learning)
    accuracy_threshold = 0.5      # RELAXED: Much easier threshold
    threshold_margin = 0.1        # RELAXED: Softer margin
    
    # Uncertainty Quantification
    predict_uncertainty = True
    uncertainty_loss_weight = 0.1
    
    # Zero-Shot Generalization
    zero_shot_mode = False
    train_schemas = list(range(1, 11))
    test_schemas = list(range(11, 14))
    train_max_objects = 5
    test_max_objects = 20
    
    # Replay buffer and curriculum
    use_replay_buffer = True
    replay_buffer_size = 10000
    replay_ratio = 0.2
    use_interleaved_curriculum = True
    interleaved_difficulty_weights = {'easy': 0.2, 'medium': 0.3, 'hard': 0.5}
    
    # Counting-specific (not used for physics-only but needed for compatibility)
    oversample_high_counts = True
    high_count_threshold = 12
    high_count_multiplier = 8
    very_high_count_threshold = 16
    very_high_count_multiplier = 16
    use_focal_loss = True
    focal_loss_gamma = 2.0
    use_count_weighting = True
    count_weight_min = 1.0
    count_weight_max = 2.5
    use_ordinal_loss = True
    ordinal_loss_weight = 0.5
    use_positional_counting = True
    use_recursive_counting = True
    counting_head_type = 'recursive'
    recursive_group_size = 5
    
    # Validation
    use_separate_validation = False
    validation_split = 0.1
    forgetting_threshold = 0.1
    
    # Additional sequence lengths
    max_seq_length_long = 512
    max_seq_length_arithmetic = 64
    max_seq_length_symbolic = 64
    
    # Dataset limits
    max_physics_episodes = 1000
    excluded_schemas = []
    
    # Gradient checkpointing
    gradient_checkpointing = False  # Not needed on A100
    cudnn_benchmark = True
    
    # Backward compatibility
    @property
    def physics_dir(self):
        return str(Path(self.data_dir))
    
    @property
    def physics_train_dir(self):
        return str(Path(self.data_dir) / 'train')
    
    @property
    def physics_val_dir(self):
        return str(Path(self.data_dir) / 'val')


# Create singleton instance
A100_CONFIG = A100Config()


def get_a100_config():
    """Get A100 configuration."""
    return A100_CONFIG


if __name__ == '__main__':
    config = get_a100_config()
    print("=" * 70)
    print("A100 80GB CONFIGURATION")
    print("=" * 70)
    print(f"\nModel Architecture:")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  ff_dim: {config.ff_dim}")
    print(f"\nTraining:")
    print(f"  batch_size: {config.batch_size}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  physics_epochs: {config.physics_epochs}")
    print(f"\nData Loading:")
    print(f"  num_workers: {config.num_workers}")
    print(f"  cache_dataset_to_ram: {config.cache_dataset_to_ram}")
    print(f"\nEstimated Training Time: 6-12 hours")
    print("=" * 70)
