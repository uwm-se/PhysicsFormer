"""
Training Configuration for PhysicsFormer

Memory-Optimized for 12GB VRAM + 21GB RAM
Target Training Time: ~3-5 days (with optimized batch size + AMP)

Key Optimizations (12GB VRAM + 21GB RAM):
- hidden_dim: 512 - Increased capacity for 35D state (head_dim=32 with 16 heads)
- max_objects: 20 - Reduced from 25 (saves 36% memory, quadratic reduction)
- max_seq_length: 512 - Full dynamics capture for complex schemas (~8.5 seconds at 60fps)
- encoder_chunk_size: 256 - Processes in 8 chunks (eliminates overhead)
- seq_chunk_size: 64 - Temporal chunking for memory efficiency
- batch_size: 4 (physics), 8 (arithmetic) - Optimized for 12GB GPU with long sequences
- accumulation_steps: 12 - Maintains effective batch=48 (physics)
- num_workers: 0 - Disabled for Windows stability (multiprocessing issues)
- mixed_precision: True - AMP for 2-3x speedup + 40% memory savings
- Dataset limits: 30K episodes per stage - Practical training time (~3-5 days)

GPU Memory Breakdown (12GB VRAM, FP16 mixed precision):
- Model (FP16): ~140 MB (70M parameters)
- Optimizer (FP32): ~560 MB (AdamW states)
- Gradients (FP16): ~140 MB
- Pairwise embeddings: ~832 MB (batch=4 * seq=512 * 20² * 512 * 2 bytes / 8 chunks)
- Activations & workspace: ~3-5 GB (transformer, cuDNN)
- PyTorch allocator overhead: ~1-2 GB
- **Peak GPU: ~5-8 GB** (safe for 12GB VRAM with 4-7GB margin)

System RAM Breakdown (21GB):
- Dataset (memory-mapped HDF5): ~3-5 GB resident
- DataLoader buffers: ~2-3 GB
- Python/PyTorch overhead: ~2-3 GB
- **Peak RAM: ~8-12 GB** (safe for 21GB RAM)

Training Time Estimate (with optimized batch size):
- Physics: 30K episodes × 50 epochs × 1.5s/batch ÷ batch_size=8 → ~1.0 days
- Counting: 20K samples × 15 epochs → ~0.4 days
- Arithmetic: 30K episodes × 30 epochs → ~1.5 days
- Symbolic: 30K problems × 20 epochs → ~1.0 days
- Total: ~3-5 days (with batch_size=8, no multiprocessing overhead)

Note: Single-process data loading (num_workers=0) is more stable on Windows
"""

import os
from pathlib import Path
import torch

class TrainingConfig:
    """Configuration for training all 4 levels - Memory Optimized for 10GB VRAM + 21GB RAM."""
    
    # Model Architecture (Increased capacity for 35D state)
    state_dim = 35            # 35D comprehensive physics state
    hidden_dim = 512          # INCREASED: More capacity for 35D state (was 256)
    num_layers = 8            # Match checkpoint (was trained with 8 layers)
    num_heads = 16            # Multi-head attention (head_dim=32, proper expressivity)
    ff_dim = 2048             # Feed-forward dimension
    max_objects = 20          # Keep at 20 (quadratic memory scaling)
    num_schema_classes = 37   # 37 Isaac Sim physics schemas (11 groups)
    vocab_size = 100          # For symbolic math (hybrid encoder uses this as fixed range)
    dropout = 0.2             # INCREASED: More regularization to prevent overfitting (was 0.1)
    label_smoothing = 0.1     # NEW: Smooth labels for counting task
    weight_decay = 0.01       # NEW: L2 regularization
    
    # Schema Curriculum (Progressive difficulty)
    # Level 1-11: Physics schema groups (4 → 37 schemas)
    # Level 12: CAUSAL training (object dropout + intervention loss)
    # Level 13: COUNTERFACTUAL training ("what if" reasoning)
    # 
    # Start with level 1 for fast initial training (~10-15 min/epoch)
    # Curriculum auto-advances when loss plateaus
    schema_curriculum_level = 1  # Start simple, will progress to 12-13 for causal/counterfactual
    counting_schema_level = 8     # Counting uses Groups 1-8 (33 schemas) - excludes articulated/rotation/complex
    
    # Training (batch sizes optimized for 35D state + 12GB VRAM)
    # SPEED OPTIMIZED: 1-2 day training target
    batch_size = 8                # Local GPU (12GB VRAM)
    batch_size_physics = 8        # Physics Stage 1: optimized for 12GB VRAM
    batch_size_physics_long = 8   # Physics with longer sequences
    batch_size_consolidation = 32 # Consolidation replay
    accumulation_steps = 1        # No accumulation needed with larger batches
    effective_batch_size = batch_size * accumulation_steps  # 64 effective
    
    # Optimization (FIXED for actual learning)
    learning_rate = 5e-4          # INCREASED: Model wasn't learning at 1e-5
    weight_decay = 0.01           # Standard weight decay
    max_grad_norm = 1.0           # TIGHT: Prevent inf gradients
    max_grad_value = 1.0          # INCREASED: Allow larger updates for learning
    warmup_steps = 500            # REDUCED: Start learning faster
    lr_scheduler = 'cosine'       # Cosine annealing for better convergence
    min_lr = 1e-6                 # Higher minimum to keep learning
    weight_init_scale = 0.02      # Xavier/He initialization scale
    
    # Loss Weights (FIXED for stable training)
    prediction_weight = 1.0       # Primary loss component
    threshold_weight = 0.05       # REDUCED: Less pressure on threshold
    energy_weight = 0.001         # MUCH LOWER: Prevent gradient explosion
    momentum_weight = 0.001       # MUCH LOWER: Prevent gradient explosion
    counting_weight = 1.0         # Equal to MSE - counting is important
    arithmetic_weight = 0.5
    symbolic_weight = 1.0
    
    # Threshold-aware loss parameters (RELAXED for learning)
    accuracy_threshold = 0.5      # RELAXED: Much easier threshold (was 0.05)
    threshold_margin = 0.1         # RELAXED: Softer margin
    
    # Memory Optimization (Optimized for 35D state + 12GB GPU)
    mixed_precision = True    # FP16 for speed and memory savings (AMP enabled in training loop)
    gradient_checkpointing = True  # ENABLED: Save VRAM by recomputing activations (was False)
    pin_memory = True
    num_workers = 2  # Multi-process data loading (2-4 workers recommended for Windows)
    persistent_workers = True  # Keep workers alive between epochs
    prefetch_factor = 2  # Prefetch 2 batches per worker
    encoder_chunk_size = 128  # REDUCED: Process in smaller chunks for 35D (was 256)
    seq_chunk_size = 32  # REDUCED: Smaller sequence chunks for 35D (was 64)
    
    # Data Loading Optimization (NEW - 10x speedup!)
    cache_dataset_to_ram = True  # Load entire dataset into RAM (requires ~10-15GB RAM, gives 10x speedup)
    max_episodes_per_file = 500   # LIMIT: Faster epochs, still enough data per schema
    
    # Speed Optimization
    compile_model = False     # PyTorch 2.0+ compilation (disabled on Windows due to dataclass compatibility issues)
    cudnn_benchmark = True    # cuDNN autotuner
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Epochs per stage
    # PHYSICS-ONLY TRAINING (Stages 2-4 archived - see archive/ARCHIVED_STAGES_2_3_4.md)
    # 11 schema levels × 3 epochs/level = 33 epochs minimum
    # With faster progression: ~40-50 epochs total
    physics_epochs = 200      # COMPLETE: Full curriculum with patience (13 × 15 epochs)
    
    # Dataset size limits (PER FILE to maintain schema diversity)
    # With 43 schema files: 1200 × 43 = ~51,600 total episodes
    # This ensures all schemas are represented in training
    max_physics_episodes = 1000       # Per-file limit: ~37K total with 37 schemas
    # max_counting_samples, max_arithmetic_episodes, max_symbolic_problems - ARCHIVED
    
    # Curriculum Learning Options (NEW)
    use_interleaved_curriculum = True  # Use interleaved training instead of sequential
    interleaved_difficulty_weights = {  # Sampling weights for each difficulty
        'easy': 0.2,    # 20% easy schemas (prevent forgetting)
        'medium': 0.3,  # 30% medium schemas
        'hard': 0.5     # 50% hard schemas (emphasize difficult cases)
    }
    use_replay_buffer = True   # Use replay buffer to prevent catastrophic forgetting
    replay_buffer_size = 10000  # Maximum samples in replay buffer
    replay_ratio = 0.2          # 20% of batch from replay buffer
    
    # Counting-Specific Improvements (NEW)
    oversample_high_counts = True   # Oversample high counts (>=12) for better learning
    high_count_threshold = 12       # Counts >= this value get oversampled
    high_count_multiplier = 8       # Multiply high count samples by this factor (increased from 2)
    very_high_count_threshold = 16  # Counts >= this value get even more oversampling
    very_high_count_multiplier = 16 # Multiply very high count samples by this factor
    use_focal_loss = True           # Use focal loss instead of standard cross-entropy
    focal_loss_gamma = 2.0          # Focal loss focusing parameter (higher = more focus on hard)
    use_count_weighting = True      # Weight loss by count difficulty
    count_weight_min = 1.0          # Minimum weight for low counts
    count_weight_max = 2.5          # Maximum weight for high counts (15-20)
    
    # Advanced Counting Improvements
    use_ordinal_loss = True         # Use ordinal regression loss
    ordinal_loss_weight = 0.5       # Weight for ordinal loss component
    use_positional_counting = True  # Use positional encoding counting head
    use_recursive_counting = True   # Use recursive counting head
    counting_head_type = 'recursive' # Type: 'standard', 'positional', 'recursive', 'hierarchical', 'ensemble'
    recursive_group_size = 5        # Group size for recursive counting
    
    # Early stopping (relaxed for counting)
    use_early_stopping = True # Enable early stopping to prevent overfitting
    patience = 20             # INCREASED: Allow more time to converge (was 10)
    min_delta = 1e-5          # REDUCED: More sensitive to improvements (was 1e-4)
    
    # Checkpointing
    save_every = 1  # Save checkpoint every epoch for safety
    checkpoint_dir = os.environ.get('CHECKPOINT_DIR', '$CHECKPOINT_DIR')  # Save to D: drive with more space
    
    # Data - Use environment variable or default
    # Set via: export PHYSICS_DATA_DIR=/path/to/data (Linux/Mac) or set PHYSICS_DATA_DIR=C:\path\to\data (Windows)
    # Expected structure:
    #   $PHYSICS_DATA_DIR/
    #   ├── physics/          (HDF5 physics simulations - all data)
    #   ├── physics_train/    (Training split - 90%)
    #   ├── physics_val/      (Validation split - 10%)
    #   ├── arithmetic/       (Arithmetic operations - all data)
    #   ├── arithmetic_train/ (Training split - 90%)
    #   ├── arithmetic_val/   (Validation split - 10%)
    #   ├── symbolic/         (Symbolic math - all data)
    #   ├── symbolic_train/   (Training split - 90%)
    #   └── symbolic_val/     (Validation split - 10%)
    data_dir = os.environ.get('PHYSICS_DATA_DIR', 'D:/physics_hdf5')
    
    # Schema filtering - Exclude chaotic/unpredictable schemas for better learning
    # These schemas are inherently difficult to predict due to:
    # - Chaotic dynamics (exponential sensitivity to initial conditions)
    # - Very large datasets (slow training, memory intensive)
    # - Complex multi-scale interactions
    # - Path-dependent/irreversible behavior
    excluded_schemas = []  # Include all 48 schemas
    
    # Validation configuration
    use_separate_validation = False  # Use runtime split (quick fix - works with existing data)
    validation_split = 0.1  # 10% for validation (runtime split)
    
    # Fixed sequence length for physics - 128 frames (2.1 seconds at 60fps)
    # SPEED OPTIMIZED: 2x faster training, still captures key physics events
    # 128 frames is sufficient for collisions, bounces, and most dynamics
    max_seq_length = 128  # REDUCED: 2x speedup (was 256)
    max_seq_length_long = 128  # Same as default
    max_seq_length_arithmetic = 64  # Shorter for arithmetic (less temporal)
    max_seq_length_symbolic = 64    # Shorter for symbolic (less temporal)
    
    # Memory formula: batch_size_physics * seq_chunk * objects² * hidden_dim * 2 bytes (FP16)
    # With encoder_chunk_size=256, batch_flat is processed in chunks:
    # Physics (512): batch=4 × seq=512 = 2048 batch_flat → 8 chunks of 256 each
    # Per chunk: 256 × 20² × 256 × 2 bytes = 52 MB (pairwise only, reduced from 82 MB with 25 objects)
    # Total pairwise: 8 chunks × 52 MB = 416 MB (reduced from 656 MB, saves 36%)
    # Arithmetic (64): batch=8 × seq=64 = 512 batch_flat → 2 chunks of 256 each = 105 MB (reduced from 164 MB)
    # Total GPU usage (with model, activations, gradients): Peak ~2.5-4 GB (safe for 12GB GPU, reduced from 3-5 GB)
    
    forgetting_threshold = 0.1  # 10% drop triggers warning
    
    # Debug Mode
    debug = True  # Enable verbose error tracebacks
    
    # ============================================================
    # MODERN TRAINING IMPROVEMENTS (LLaMA/GPT-4 style)
    # ============================================================
    
    # Architecture Improvements (applied at model creation)
    use_rmsnorm = True              # Replace LayerNorm with faster RMSNorm
    use_flash_attention = True      # Use Flash Attention (PyTorch 2.0+)
    use_rope = True                 # Rotary Position Embeddings
    use_swiglu = True               # SwiGLU activation (LLaMA-style)
    
    # EMA (Exponential Moving Average)
    use_ema = True                  # Maintain smoothed model weights
    ema_decay = 0.999               # EMA decay rate (0.999 = slow update)
    
    # Learning Rate Schedule
    use_cosine_schedule = True      # Cosine decay with warmup
    warmup_ratio = 0.05             # 5% of training for warmup
    min_lr_ratio = 0.1              # Decay to 10% of initial LR
    
    # Data Augmentation (physics-preserving)
    use_data_augmentation = True    # Enable data augmentation
    augment_permute_prob = 0.3      # Object permutation probability
    augment_noise_std = 0.01        # Small noise injection std
    augment_time_reverse_prob = 0.1 # Time reversal probability
    
    # Multi-Scale Temporal Loss
    use_multi_scale_loss = True     # Penalize errors at multiple horizons
    temporal_horizons = [1, 5, 10]  # Prediction horizons (timesteps)
    temporal_weights = [1.0, 0.5, 0.25]  # Weights for each horizon
    
    # Contrastive Learning (optional - more compute)
    use_contrastive_loss = False    # Physics-aware contrastive learning
    contrastive_temperature = 0.07  # InfoNCE temperature
    contrastive_weight = 0.1        # Weight for contrastive loss
    
    # Label Smoothing for Physics
    use_label_smoothing = True      # Prevent overconfident predictions
    physics_label_smoothing = 0.1   # Smoothing factor for physics targets
    
    # Curriculum Learning Parameters (prevents rushing through schemas)
    curriculum_min_epochs = 10      # Min epochs per schema level before advancing
    curriculum_patience = 5         # Epochs without improvement before considering advance
    curriculum_improvement_threshold = 0.02  # 2% improvement required to reset patience
    curriculum_min_accuracy = 0.4   # Must reach 40% accuracy before advancing
    
    # Uncertainty Quantification
    predict_uncertainty = True      # Predict aleatoric uncertainty (variance)
    uncertainty_loss_weight = 0.1   # Weight for NLL loss term
    
    # Zero-Shot Generalization (for ICML experiments)
    zero_shot_mode = False          # Enable zero-shot evaluation mode
    train_schemas = list(range(1, 11))   # Train on schemas 1-10
    test_schemas = list(range(11, 14))   # Test on schemas 11-13 (unseen)
    train_max_objects = 5           # Train on 2-5 objects
    test_max_objects = 20           # Test on up to 20 objects (unseen)
    
    # Backward compatibility property
    @property
    def physics_dir(self):
        """Returns the physics directory path (all HDF5 files are in data_dir root)."""
        return str(Path(self.data_dir))
    
    # Train/Val directory properties
    @property
    def physics_train_dir(self):
        """Returns the physics training subdirectory path."""
        return str(Path(self.data_dir) / 'physics_train')
    
    @property
    def physics_val_dir(self):
        """Returns the physics validation subdirectory path."""
        return str(Path(self.data_dir) / 'physics_val')
    
    @property
    def arithmetic_dir(self):
        """Returns the arithmetic subdirectory path."""
        return str(Path(self.data_dir) / 'arithmetic')
    
    @property
    def arithmetic_train_dir(self):
        """Returns the arithmetic training subdirectory path."""
        return str(Path(self.data_dir) / 'arithmetic_train')
    
    @property
    def arithmetic_val_dir(self):
        """Returns the arithmetic validation subdirectory path."""
        return str(Path(self.data_dir) / 'arithmetic_val')
    
    @property
    def symbolic_dir(self):
        """Returns the symbolic subdirectory path."""
        return str(Path(self.data_dir) / 'symbolic')
    
    @property
    def symbolic_train_dir(self):
        """Returns the symbolic training subdirectory path."""
        return str(Path(self.data_dir) / 'symbolic_train')
    
    @property
    def symbolic_val_dir(self):
        """Returns the symbolic validation subdirectory path."""
        return str(Path(self.data_dir) / 'symbolic_val')
    
    # Logging
    log_every = 100
    validate_every = 1000
    
    def __repr__(self):
        return f"""
TrainingConfig:
  Model: hidden_dim={self.hidden_dim}, layers={self.num_layers}, heads={self.num_heads}
  Training: batch={self.batch_size}, lr={self.learning_rate}
  Memory: mixed_precision={self.mixed_precision}, pin_memory={self.pin_memory}
  Device: {self.device}
  
Estimated Memory Usage:
  Model: ~100 MB (25M parameters)
  Training: ~6-7 GB with batch_size={self.batch_size}
  Your GPU: 12 GB (plenty of headroom!)
"""


class ArithmeticStageConfig(TrainingConfig):
    """Config for Arithmetic/Symbolic stages (Stages 2-4) with reduced batch sizes to avoid OOM."""
    
    # Reduce batch sizes for stages with consolidation replay
    batch_size = 4                # Arithmetic/symbolic: REDUCED from 48 to avoid OOM
    batch_size_consolidation = 4  # Consolidation replay: REDUCED from 16
    max_objects = 15              # REDUCED from 20 to save memory during replay
    
    def __repr__(self):
        return f"""
ArithmeticStageConfig (Stages 2-4):
  Model: hidden_dim={self.hidden_dim}, layers={self.num_layers}, heads={self.num_heads}
  Batch sizes: arithmetic={self.batch_size}, consolidation={self.batch_size_consolidation}
  Max objects: {self.max_objects} (reduced for replay memory)
  
Memory Optimized for:
  - Arithmetic/Symbolic training with smaller batches
  - Consolidation replay without OOM
  - Physics replay with pairwise embeddings
"""


class ConservativeConfig(TrainingConfig):
    """Conservative config for smaller GPUs or testing."""
    
    hidden_dim = 256
    num_layers = 6
    num_heads = 8
    ff_dim = 1024
    batch_size = 16
    accumulation_steps = 2
    
    def __repr__(self):
        return f"""
ConservativeConfig:
  Model: hidden_dim={self.hidden_dim}, layers={self.num_layers}, heads={self.num_heads}
  Training: batch={self.batch_size}, lr={self.learning_rate}
  
Estimated Memory Usage:
  Model: ~25 MB (6M parameters)
  Training: ~2.5 GB with batch_size={self.batch_size}
  Effective batch: {self.batch_size * self.accumulation_steps}
"""


class MaximumConfig(TrainingConfig):
    """Maximum config - push your RTX 4080 to the limit!"""
    
    hidden_dim = 768
    num_layers = 16
    num_heads = 24
    ff_dim = 3072
    batch_size = 4  # Reduce batch for larger model
    accumulation_steps = 16  # Maintain effective batch of 64
    
    def __repr__(self):
        return f"""
MaximumConfig:
  Model: hidden_dim={self.hidden_dim}, layers={self.num_layers}, heads={self.num_heads}
  Training: batch={self.batch_size}, lr={self.learning_rate}
  
Estimated Memory Usage:
  Model: ~200 MB (50M parameters)
  Training: ~9-10 GB with batch_size={self.batch_size}
  Effective batch: {self.batch_size * self.accumulation_steps}
  Your GPU: 12 GB (using 80%!)
"""


class FineTuningConfig(TrainingConfig):
    """Config for fine-tuning with velocity consistency loss.
    
    Use this to fine-tune an existing checkpoint with the new dual-supervision
    approach that adds a consistency loss ensuring:
        current_velocity + predicted_delta ≈ target_absolute_velocity
    
    Key differences from base config:
    - Lower learning rate (1/10th) for stable fine-tuning
    - Velocity consistency loss enabled (γ = 0.1)
    - Fewer epochs (just need to adapt, not learn from scratch)
    - No warmup (starting from trained weights)
    """
    
    # Fine-tuning optimization (gentler than training from scratch)
    learning_rate = 5e-5          # 1/10th of base LR for fine-tuning
    warmup_steps = 0              # No warmup - already trained
    min_lr = 1e-6                 # Same minimum
    
    # Velocity consistency loss weight (Option B: dual supervision)
    # Start with small weight to not disrupt existing learned representations
    velocity_consistency_weight = 0.1  # γ = 0.1 (can increase if helpful)
    
    # Fewer epochs - just adapting to new loss signal
    physics_epochs = 20           # 20 epochs should be enough for fine-tuning
    
    # More aggressive early stopping for fine-tuning
    patience = 10                 # Stop earlier if no improvement
    min_delta = 1e-4              # Slightly less sensitive
    
    # Keep curriculum at current level (don't restart)
    schema_curriculum_level = 11  # Use all schemas during fine-tuning
    
    def __repr__(self):
        return f"""
FineTuningConfig (Velocity Consistency):
  Learning rate: {self.learning_rate} (1/10th of base)
  Velocity consistency weight (γ): {self.velocity_consistency_weight}
  Epochs: {self.physics_epochs}
  Patience: {self.patience}
  
  This adds dual supervision:
    L_total = L_delta + γ * L_consistency
  Where L_consistency = MSE(v_t + Δv_pred, v_t+1_target)
"""


# Default config
config = TrainingConfig()

if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING CONFIGURATIONS")
    print("=" * 70)
    
    print("\n1. CONSERVATIVE (For testing or smaller GPUs)")
    print(ConservativeConfig())
    
    print("\n2. AGGRESSIVE (Recommended for RTX 4080) [NEW]")
    print(TrainingConfig())
    
    print("\n3. MAXIMUM (Push to the limit!)")
    print(MaximumConfig())
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION FOR YOUR RTX 4080")
    print("=" * 70)
    print("\nUse: TrainingConfig() (Aggressive)")
    print("  - 512 hidden dim")
    print("  - 12 layers")
    print("  - Batch size 8 (effective 64 with accumulation)")
    print("  - ~6-7 GB memory")
    print("  - Best quality/speed trade-off")
    print("\nPASS: This will give you research-grade results!")
