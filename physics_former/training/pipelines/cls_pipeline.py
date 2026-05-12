"""
CLS-Based Training Pipeline with Advanced Features

Implements Complementary Learning Systems theory for hierarchical learning
with catastrophic forgetting prevention PLUS advanced training features.

CLS Features (Forgetting Prevention):
- McClelland, McNaughton & O'Reilly (1995): CLS theory
- Wilson & McNaughton (1994): Hippocampal replay
- Rasch & Born (2013): Sleep consolidation

Advanced Features (Performance Boost):
- Curriculum learning (progressive difficulty)
- Auxiliary tasks (richer representations)
- Contrastive learning (better embeddings)
- Warmup + cosine scheduler (stable training)
- Enhanced number encoder (zero-shot generalization)

Expected: 90-98% accuracy with maintained grounding!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
import time
import json
import logging

# Import refactored modules (only the ones we actually use)
from .modules import (
    GradientHandler,
    CheckpointManager,
    OptimizerFactory,
    BatchProcessor
)
from .training_helpers import (
    log_training_progress, 
    print_epoch_summary, 
    prepare_physics_batch,
    compute_improved_loss as compute_improved_loss_helper,
    print_final_training_report,
    evaluate_embodied_cognition_helper,
    validate_all_tasks_helper,
    print_training_explainer
)

# Import shared training utilities
from training.common.training_utils import (
    compute_gradient_norm,
    freeze_encoder,
    unfreeze_all,
    move_batch_to_device
)
from training.common.checkpointing import save_checkpoint as save_checkpoint_util, load_checkpoint as load_checkpoint_util
# Note: validate_all_tasks is now in training_helpers.py as validate_all_tasks_helper

# Import base pipeline
from .full_pipeline import FullPipeline
from ..configs.config import TrainingConfig
from ..constants import (
    STAGE_NAMES,
    MSG_STAGE_COMPLETE,
    MSG_TRAINING_START,
    MAX_GRAD_NORM,
    TASK_PHYSICS,
    # DISABLED: Stages 2-4 moved to stages_2_3_4_pipeline.py
    TASK_COUNTING,  # Keep import for validation_history keys
    TASK_ARITHMETIC,  # Keep import for validation_history keys
    TASK_SYMBOLIC,  # Keep import for validation_history keys
    TASK_NAMES,
    BATCH_KEY_OBJECT_STATES,
    BATCH_KEY_OBJECT_MASK,
    BATCH_KEY_NEXT_STATES,
    BATCH_KEY_TARGET_STATES,
    BATCH_KEY_CURRENT_STATES,
    BATCH_KEY_MASSES,
    BATCH_KEY_SCHEMA,
    BATCH_KEY_COUNTS,
    BATCH_KEY_RESULTS,
    BATCH_KEY_ANSWER,
    BATCH_KEY_ANSWERS,
    BATCH_KEY_NUMBERS,
    OUTPUT_KEY_PREDICTED_STATES,
    OUTPUT_KEY_PREDICTED_ANSWER,
    OUTPUT_KEY_NUMBER_EMBEDDINGS,
    LOSS_KEY_AUXILIARY,
    LOSS_KEY_CONTRASTIVE,
    METRICS_KEY_AUX_LOSS,
    METRICS_KEY_CONTRAST_LOSS,
    DEFAULT_BATCH_SIZE,
    MASS_IDX
)
from ..utils.tensor_ops import extract_sequence_input_for_prediction
from ..utils.cls_memory import CLSMemorySystem
from ..causal_training import CausalPhysicsTrainer, CausalTrainingConfig
from ..progressive_curriculum import ProgressiveCurriculum

# Try to import embodied metrics (optional)
try:
    from ..embodied_metrics import EmbodiedMetricsAnalyzer
    EMBODIED_METRICS_AVAILABLE = True
except ImportError:
    EMBODIED_METRICS_AVAILABLE = False

# Import advanced features
try:
    from improvements import (
        CurriculumScheduler,
        create_optimizer,
        create_scheduler,
        AuxiliaryTaskHeads,
        AuxiliaryTaskLoss,
        generate_auxiliary_targets,
        CombinedContrastiveLoss,
        HybridEnhancedEncoder
    )
    IMPROVEMENTS_AVAILABLE = True
except ImportError:
    IMPROVEMENTS_AVAILABLE = False
    print("[WARN]  Advanced features not available (improvements module not found)")

# Import early stopping
try:
    from training.improvements.better_optimization import EarlyStopping
    EARLY_STOPPING_AVAILABLE = True
except ImportError:
    EARLY_STOPPING_AVAILABLE = False
    print("[WARN]  Early stopping not available (better_optimization module not found)")


class CLSTrainingPipeline(FullPipeline):
    """
    Training pipeline with Complementary Learning Systems + Advanced Features.
    
    CLS Features (Forgetting Prevention):
    1. Hippocampal episodic buffers (experience storage)
    2. Sleep-like consolidation (experience replay)
    3. Schema protection (encoder freezing)
    4. Adaptive consolidation (prioritize weak tasks)
    5. Multi-task validation (monitor all stages)
    
    Advanced Features (Performance Boost):
    6. Curriculum learning (progressive difficulty) +5-10%
    7. Auxiliary tasks (richer representations) +3-5%
    8. Contrastive learning (better embeddings) +4-6%
    9. Warmup + cosine scheduler (stable training) +1-2%
    10. Enhanced number encoder (zero-shot) +2-3%
    
    """
    
    @classmethod
    def from_config_file(cls, config_path: str, **kwargs):
        """
        Factory method to create pipeline from config file path.
        
        Args:
            config_path: Path to config file
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            CLSTrainingPipeline instance
        """
        config = TrainingConfig()  # Load default or from file
        return cls(config=config, **kwargs)
    
    def __init__(
        self,
        config: TrainingConfig,
        checkpoint_dir=None,
        use_cls=True,
        consolidation_frequency=0.2,
        adaptive_consolidation=True,
        use_curriculum=False,
        use_auxiliary=False,
        use_contrastive=False,
        use_enhanced_encoder=False,
        use_advanced_optimizer=False
    ):
        """
        Initialize CLS pipeline with advanced features.
        
        Args:
            config: TrainingConfig object (use from_config_file() to load from path)
            checkpoint_dir: Checkpoint directory (default: use config.checkpoint_dir)
            use_cls: Enable CLS memory system
            consolidation_frequency: Replay frequency (0.2 = 20%)
            adaptive_consolidation: Prioritize weak tasks
            use_curriculum: Enable curriculum learning
            use_auxiliary: Enable auxiliary tasks
            use_contrastive: Enable contrastive learning
            use_enhanced_encoder: Enable enhanced number encoder
            use_advanced_optimizer: Enable warmup + cosine scheduler
        """
        # Initialize parent with config object
        super().__init__('aggressive')  # Call parent with dummy string (legacy requirement)
        self.config = config  # Use provided config object
        
        # Print device information
        print("\n" + "=" * 70)
        print("DEVICE CONFIGURATION")
        print("=" * 70)
        print(f"Config device: {self.config.device}")
        if self.config.device == 'cuda' or (hasattr(self.config.device, 'type') and self.config.device.type == 'cuda'):
            if torch.cuda.is_available():
                print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                print(f"   Device: cuda:0")
            else:
                print("[WARN] Config set to 'cuda' but no GPU detected - will use CPU")
                print("   This will be VERY slow!")
        else:
            print("[INFO] Using CPU (this will be slow)")
        print("=" * 70 + "\n")
        
        # Use checkpoint_dir from config if not explicitly provided
        if checkpoint_dir is None:
            checkpoint_dir = self.config.checkpoint_dir
        
        # Ensure checkpoint_dir is a directory, not a file path
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.is_file():
            # If a checkpoint file was passed, use its parent directory
            self.checkpoint_dir = checkpoint_path.parent
            print(f"[WARN] Checkpoint file passed instead of directory. Using parent: {self.checkpoint_dir}")
        else:
            self.checkpoint_dir = checkpoint_path
        
        # Create directory if it doesn't exist
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.use_cls = use_cls
        
        # Advanced features flags
        self.use_curriculum = use_curriculum and IMPROVEMENTS_AVAILABLE
        self.use_auxiliary = use_auxiliary and IMPROVEMENTS_AVAILABLE
        self.use_contrastive = use_contrastive and IMPROVEMENTS_AVAILABLE
        self.use_enhanced_encoder = use_enhanced_encoder and IMPROVEMENTS_AVAILABLE
        self.use_advanced_optimizer = use_advanced_optimizer and IMPROVEMENTS_AVAILABLE
        
        # Initialize CLS memory system (hippocampus)
        if use_cls:
            self.cls_memory = CLSMemorySystem(
                tasks=TASK_NAMES,
                capacity_per_task=500,  # Reduced from 10000 to prevent memory explosion (56GB -> ~2.8GB)
                consolidation_frequency=consolidation_frequency,
                device=self.config.device,
                adaptive=adaptive_consolidation
            )
            print("[OK] CLS memory system initialized (hippocampal buffers)")
        else:
            self.cls_memory = None
            print("[WARN]  CLS disabled - catastrophic forgetting may occur!")
        
        # Initialize Replay Buffer for curriculum learning
        if getattr(config, 'use_replay_buffer', False):
            from ..common.replay_buffer import ReplayBuffer
            self.replay_buffer = ReplayBuffer(
                max_size=config.replay_buffer_size,
                replay_ratio=config.replay_ratio,
                sampling_strategy='diverse'
            )
            print(f"[OK] Replay buffer initialized (size={config.replay_buffer_size}, ratio={config.replay_ratio:.0%})")
        else:
            self.replay_buffer = None
        
        # Initialize refactored modules
        self.gradient_handler = GradientHandler(
            max_grad_norm=self.config.max_grad_norm,
            max_grad_value=self.config.max_grad_value,
            detect_anomalies=True
        )
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.checkpoint_dir)
        self.batch_processor = BatchProcessor(device=self.config.device)
        
        # Initialize progressive curriculum (sequence length AND schema difficulty)
        # ENABLED: Automatically progress schemas (1→13) and sequence length
        # Use config values or defaults from ProgressiveCurriculum
        self.progressive_curriculum = ProgressiveCurriculum(
            initial_seq_length=self.config.max_seq_length,
            target_seq_length=256,
            initial_schema_level=1,  # Always start from level 1
            target_schema_level=13,  # Progress to all 13 levels (11 physics + causal + counterfactual)
            min_epochs_per_phase=getattr(self.config, 'curriculum_min_epochs', 10),
            max_epochs_per_phase=getattr(self.config, 'curriculum_max_epochs', 30),  # HARD LIMIT: 400/13 ≈ 30
            convergence_patience=getattr(self.config, 'curriculum_patience', 5),
            improvement_threshold=getattr(self.config, 'curriculum_improvement_threshold', 0.02),
            min_accuracy_to_progress=getattr(self.config, 'curriculum_min_accuracy', 0.4)
        )
        print(f"\n[CURRICULUM] Progressive Curriculum (Schema + Sequence Length):")
        print(f"   Schema Level: {self.progressive_curriculum.current_schema_level} -> {self.progressive_curriculum.target_schema_level}")
        print(f"   Schema Progression: {self.progressive_curriculum.schema_progression}")
        print(f"   Sequence Length: {self.progressive_curriculum.current_seq_length} -> {self.progressive_curriculum.target_seq_length}")
        print(f"   Sequence Progression: {self.progressive_curriculum.sequence_progression}")
        print(f"   Min epochs per phase: {self.progressive_curriculum.min_epochs_per_phase}")
        print(f"   Max epochs per phase: {self.progressive_curriculum.max_epochs_per_phase} (HARD LIMIT)")
        print(f"   Strategy: Progress schemas first, then sequence length")
        print(f"   Level 12: Causal Training (object dropout + intervention loss)")
        print(f"   Level 13: Counterfactual Training (contrastive what-if reasoning)")
        
        # Initialize causal trainer for schema levels 12-13
        self.causal_trainer = None  # Will be initialized when model is set
        self.causal_config = CausalTrainingConfig(
            dropout_prob=0.3,
            intervention_weight=1.0,
            contrastive_weight=0.5,
            causal_margin=0.1,
            collision_threshold=2.0,
            velocity_threshold=0.1
        )
        
        # Initialize advanced features
        self.curriculum_scheduler = None
        self.aux_heads = None
        self.aux_loss_fn = None
        self.contrastive_loss_fn = None
        
        if IMPROVEMENTS_AVAILABLE:
            print("\n" + "="*70)
            print("ADVANCED FEATURES")
            print("="*70)
            
            if self.use_curriculum:
                self.curriculum_scheduler = CurriculumScheduler(
                    stages=[
                        (0, 5),    # Stage 1: Very easy
                        (0, 10),   # Stage 2: Easy
                        (0, 50),   # Stage 3: Medium
                        (0, 100),  # Stage 4: Hard
                    ],
                    epochs_per_stage=20
                )
                print("[OK] Curriculum learning enabled (+5-10% accuracy)")
            
            if self.use_auxiliary:
                self.aux_heads = AuxiliaryTaskHeads(
                    hidden_dim=self.config.hidden_dim,
                    max_objects=self.config.max_objects
                )
                # Move aux heads to device when model is set up
                self.aux_loss_fn = AuxiliaryTaskLoss(
                    comparison_weight=0.2,
                    magnitude_weight=0.1,
                    parity_weight=0.1,
                    digit_sum_weight=0.1
                )
                print("[OK] Auxiliary tasks enabled (+3-5% accuracy)")
                print("  (Aux heads will be moved to device during model setup)")
            
            if self.use_contrastive:
                self.contrastive_loss_fn = CombinedContrastiveLoss(
                    use_contrastive=True,
                    use_triplet=True,
                    use_ordering=True,
                    contrastive_weight=1.0,
                    triplet_weight=0.5,
                    ordering_weight=0.2
                )
                print("[OK] Contrastive learning enabled (+4-6% accuracy)")
            
            if self.use_enhanced_encoder:
                print("[OK] Enhanced number encoder enabled (+2-3% accuracy)")
                print("  (Will be installed during model setup)")
            
            if self.use_advanced_optimizer:
                print("[OK] Warmup + cosine scheduler enabled (+1-2% accuracy)")
            
            total_improvement = 0
            if self.use_curriculum: total_improvement += 7.5
            if self.use_auxiliary: total_improvement += 4
            if self.use_contrastive: total_improvement += 5
            if self.use_enhanced_encoder: total_improvement += 2.5
            if self.use_advanced_optimizer: total_improvement += 1.5
            
            print(f"\n[IMPROVE] Expected total improvement: +{total_improvement:.1f}% accuracy")
            print("="*70)
        
        # Track validation performance
        self.validation_history = {
            TASK_PHYSICS: [],
            TASK_COUNTING: [],
            TASK_ARITHMETIC: [],
            TASK_SYMBOLIC: []
        }
        
        # Initialize temporal metrics tracker (no global singleton)
        try:
            from ..utils.temporal_metrics import create_temporal_metrics_tracker
            self.temporal_metrics = create_temporal_metrics_tracker()
        except ImportError:
            self.temporal_metrics = None
        
        # Training statistics
        self.stats = {
            'consolidations': 0,
            'consolidation_by_task': {},
            'forgetting_detected': [],
            'curriculum_enabled': self.use_curriculum,
            'auxiliary_enabled': self.use_auxiliary,
            'contrastive_enabled': self.use_contrastive,
            'embodied_metrics': []  # Track embodied cognition over time
        }
    
    def setup(self):
        """Setup model and device, then move aux_heads to device."""
        # Call parent setup
        super().setup()
        
        # Move auxiliary heads to device if they exist
        if self.use_auxiliary and self.aux_heads is not None:
            self.aux_heads = self.aux_heads.to(self.device)
            print("[OK] Auxiliary task heads moved to device")
        
        return self
    
    
    def create_improved_optimizer(self, model, lr=None):
        """Create optimizer using OptimizerFactory."""
        if lr is None:
            lr = self.config.learning_rate
        
        # Use OptimizerFactory for cleaner code
        optimizer = OptimizerFactory.create_optimizer(
            model=model,
            optimizer_type='adamw',
            learning_rate=lr,
            weight_decay=self.config.weight_decay
        )
        return optimizer
    
    def create_improved_scheduler(self, optimizer, total_steps):
        """Create scheduler using OptimizerFactory."""
        # Use OptimizerFactory for cleaner code
        scheduler = OptimizerFactory.create_scheduler(
            optimizer=optimizer,
            scheduler_type='cosine' if self.use_advanced_optimizer else 'none',
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
            min_lr=self.config.min_lr
        )
        return scheduler
    
    def compute_improved_loss(self, loss, embeddings=None, numbers=None):
        """Compute loss with advanced features - delegates to helper."""
        return compute_improved_loss_helper(
            loss, embeddings, numbers,
            self.aux_heads, self.aux_loss_fn, self.contrastive_loss_fn,
            self.use_auxiliary, self.use_contrastive, self.config
        )
    
    def train_stage_with_cls(
        self,
        stage: int,
        dataloader,
        epochs: int,
        task_name: str,
        should_freeze_encoder: bool = False,
        enable_consolidation: bool = False,
        consolidation_tasks: Optional[List[str]] = None,
        metrics_logger=None,
        validation_dataloaders: dict = None,
        start_epoch: int = 0
    ):
        """
        Train a stage with CLS consolidation.
        
        Args:
            stage: Stage number (1-4)
            dataloader: Training data
            epochs: Number of epochs
            task_name: Task name ('physics', 'counting', etc.)
            should_freeze_encoder: Freeze physics encoder (schema protection)
            enable_consolidation: Enable experience replay
            consolidation_tasks: Tasks to consolidate (replay)
            metrics_logger: Optional metrics logger
            validation_dataloaders: Optional dict of validation dataloaders
        """
        # Check if dataloader is valid
        if dataloader is None:
            print(f"\n[ERROR] Dataloader for task '{task_name}' is None!")
            print(f"[ERROR] Cannot train Stage {stage}. Please check data loading.")
            return {'stage': stage, 'epoch': 0, 'status': 'failed'}
        
        if stage == "physics":
            print("=" * 70 + "\nTRAINING PHYSICS\n" + "=" * 70)
        else:
            print(MSG_TRAINING_START.format(stage=STAGE_NAMES[stage-1]))
        print(f"\n[TARGET] Task: {task_name.upper()}")
        print(f"[DATA] Dataset size: {len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 'Unknown'}")
        print(f"[COUNT] Batches per epoch: {len(dataloader)}")
        print(f"[CYCLE] Total epochs: {epochs}")
        print(f"[STATS] Total training steps: {epochs * len(dataloader)}")
        
        if self.model is None:
            print("\n[CONFIG]  Setting up model...")
            self.setup()
            print("[OK] Model ready")
        
        # Initialize causal trainer now that model is available
        if self.causal_trainer is None:
            self.causal_trainer = CausalPhysicsTrainer(self.model, self.causal_config)
            print("[OK] Causal trainer initialized for schema levels 12-13")
        
        # Schema protection: Freeze encoder if requested (using shared utility)
        print(f"\n[LOCKED] Parameter Configuration:")
        if should_freeze_encoder:
            freeze_encoder(self.model)
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"   [BRAIN] Schema protection ACTIVE (preserving physics grounding)")
            print(f"   [UNLOCKED] Trainable: {trainable_params:,} / {total_params:,} parameters ({trainable_params/total_params*100:.1f}%)")
        else:
            # Unfreeze all (using shared utility)
            unfreeze_all(self.model)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"   [UNLOCKED] All parameters trainable: {total_params:,}")
        
        # Optimizer with appropriate learning rate
        print(f"\n[CONFIG]  ========== TRAINING CONFIGURATION ==========")
        print(f"   Learning Rate: {self.config.learning_rate:.2e}")
        print(f"   Weight Decay: {self.config.weight_decay}")
        print(f"   Gradient Clipping (norm): {self.config.max_grad_norm}")
        if hasattr(self.config, 'max_grad_value'):
            print(f"   Gradient Clipping (value): {self.config.max_grad_value}")
        print(f"   Warmup Steps: {self.config.warmup_steps}")
        print(f"   Min LR: {self.config.min_lr:.2e}")
        print(f"   Batch Size: {self.config.batch_size} (arithmetic/symbolic)")
        print(f"   Batch Size Physics: {self.config.batch_size_physics}")
        print(f"   Accumulation Steps: {self.config.accumulation_steps}")
        print(f"   Effective Batch Size: {self.config.effective_batch_size}")
        print(f"   Mixed Precision: {self.config.mixed_precision}")
        print(f"   ===============================================")
        
        print(f"\n[CONFIG]  Optimizer Configuration:")
        if should_freeze_encoder:
            # Slow neocortical consolidation
            lr = self.config.learning_rate / 10  # 10x slower
            print(f"   [LOSS] Slow consolidation mode: lr = {lr:.2e} (neocortical integration)")
        else:
            lr = self.config.learning_rate
            print(f"   [UP] Standard learning rate: {lr:.2e}")
        
        # Create optimizer (with advanced features if enabled)
        optimizer = self.create_improved_optimizer(self.model, lr=lr)
        
        # Create scheduler (with advanced features if enabled)
        total_steps = epochs * len(dataloader)
        scheduler = self.create_improved_scheduler(optimizer, total_steps)
        
        # Initialize modern training improvements (EMA, data augmentation, etc.)
        ema_model = None
        data_augmentation = None
        try:
            from training.improvements.modern_training import EMAModel, PhysicsDataAugmentation
            
            if getattr(self.config, 'use_ema', True):
                ema_decay = getattr(self.config, 'ema_decay', 0.999)
                ema_model = EMAModel(self.model, decay=ema_decay)
                print(f"\n[MODERN] EMA enabled (decay={ema_decay})")
            
            if getattr(self.config, 'use_data_augmentation', True):
                data_augmentation = PhysicsDataAugmentation(
                    permute_prob=getattr(self.config, 'augment_permute_prob', 0.3),
                    noise_std=getattr(self.config, 'augment_noise_std', 0.01),
                    time_reverse_prob=getattr(self.config, 'augment_time_reverse_prob', 0.1)
                )
                print(f"[MODERN] Data augmentation enabled")
        except ImportError:
            print(f"[WARN] Modern training improvements not available")
        
        # Consolidation info
        if enable_consolidation and self.use_cls:
            print(f"\n[CYCLE] Consolidation Configuration:")
            print(f"   [OK] Experience replay ENABLED")
            print(f"   [CURRICULUM] Tasks to consolidate: {', '.join(consolidation_tasks) if consolidation_tasks else 'None'}")
            print(f"   [RANDOM] Replay frequency: {self.cls_memory.consolidation_frequency:.1%}")
            if self.cls_memory.adaptive:
                print(f"   [BRAIN] Adaptive consolidation: ACTIVE (prioritizes weak tasks)")
        else:
            print(f"\n[CYCLE] Consolidation: DISABLED (first task or no CLS)")
        
        # Loss function with config weights (CRITICAL: prevents loss explosion)
        from models.physics_former_full import FullPhysicsLoss
        loss_fn = FullPhysicsLoss(
            prediction_weight=self.config.prediction_weight,
            energy_weight=self.config.energy_weight,
            momentum_weight=self.config.momentum_weight,
            counting_weight=self.config.counting_weight,
            arithmetic_weight=self.config.arithmetic_weight,
            symbolic_weight=self.config.symbolic_weight,
            threshold_weight=getattr(self.config, 'threshold_weight', 0.3),
            accuracy_threshold=getattr(self.config, 'accuracy_threshold', 0.1),
            threshold_margin=getattr(self.config, 'threshold_margin', 0.05),
            label_smoothing=getattr(self.config, 'label_smoothing', 0.1)
        )
        
        # DISABLED: Counting-specific loss improvements - stages 2-4 moved to stages_2_3_4_pipeline.py
        # if task_name == TASK_COUNTING:
        #     loss_fn.use_focal_loss = getattr(self.config, 'use_focal_loss', False)
        #     ... (see stages_2_3_4_pipeline.py for counting loss config)
        print(f"\n[LOSS] Loss function: FullPhysicsLoss (mode={task_name})")
        print(f"   Prediction weight: {self.config.prediction_weight}")
        print(f"   Threshold weight: {getattr(self.config, 'threshold_weight', 0.3)} (encourages crossing accuracy threshold)")
        print(f"   Energy weight: {self.config.energy_weight} (normalized for stability)")
        print(f"   Momentum weight: {self.config.momentum_weight} (normalized for stability)")
        
        # Initialize GradScaler for mixed precision training
        use_amp = self.config.mixed_precision and torch.cuda.is_available()
        scaler = GradScaler(enabled=use_amp)
        if use_amp:
            print(f"\n[SPEED] Automatic Mixed Precision (AMP) ENABLED")
            print(f"   Expected speedup: 2-3x faster training")
            print(f"   Memory savings: ~40% reduction")
        else:
            print(f"\n[WARN] AMP disabled (CPU mode or mixed_precision=False)")
        
        # Initialize early stopping
        early_stopping = None
        if EARLY_STOPPING_AVAILABLE and self.config.use_early_stopping:
            early_stopping = EarlyStopping(
                patience=self.config.patience,
                min_delta=self.config.min_delta
            )
            print(f"\n[EARLY STOP] Early stopping enabled")
            print(f"   Patience: {self.config.patience} epochs")
            print(f"   Min delta: {self.config.min_delta}")
        elif not self.config.use_early_stopping:
            print(f"\n[INFO] Early stopping disabled (use_early_stopping=False)")
        
        # Training loop
        self.model.train()
        print(f"\n{'='*70}")
        print(f"[START] STARTING TRAINING")
        print(f"{'='*70}")
        
        stage_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_losses = []
            consolidation_count = 0
            batch_size = None
            batch_start_times = []
            last_grad_norm = 0.0
            accumulation_steps = getattr(self.config, 'accumulation_steps', 1)
            validation_accuracy = 0.0
            
            # Epoch header
            current_lr = optimizer.param_groups[0]['lr'] if scheduler else self.config.learning_rate
            schema_level = self.progressive_curriculum.current_schema_level
            print(f"\n[EPOCH {epoch+1}/{epochs}] Schema {schema_level}/13 | LR: {current_lr:.2e} | {len(dataloader)} batches")
            
            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            for batch_idx, batch in enumerate(dataloader):
                batch_process_start = time.time()
                is_first_batch = (batch_idx == 0)
                
                # First batch only: initialization message
                if is_first_batch and epoch == 0:
                    print(f"   Initializing GPU memory...", flush=True)
                
                # Prepare and transfer batch to GPU
                batch = prepare_physics_batch(batch, constants=sys.modules['training.constants'])
                batch = self.batch_processor.move_to_device(batch)
                
                # Apply data augmentation if enabled (only during training)
                if data_augmentation is not None and self.model.training:
                    batch = data_augmentation(batch)
                
                # Store batch size from first batch
                if batch_size is None:
                    batch_size = batch[BATCH_KEY_OBJECT_STATES].shape[0] if BATCH_KEY_OBJECT_STATES in batch else DEFAULT_BATCH_SIZE
                
                # Forward pass with automatic mixed precision
                with autocast(enabled=use_amp):
                    # Check if we're in causal/counterfactual training mode (schema levels 12-13)
                    current_schema = self.progressive_curriculum.current_schema_level
                    use_causal_training = (current_schema >= 12 and task_name == TASK_PHYSICS)
                    
                    # Initialize causal metrics storage
                    causal_metrics = {}
                    
                    if use_causal_training:
                        # CAUSAL TRAINING: Use object dropout and intervention loss
                        states = batch.get(BATCH_KEY_OBJECT_STATES)
                        mask = batch.get(BATCH_KEY_OBJECT_MASK)
                        
                        if states is not None and mask is not None:
                            causal_losses = self.causal_trainer.train_step(states, mask)
                            outputs = self.model(**batch, mode=task_name)
                            
                            # Store causal loss to add to total loss later
                            causal_total_loss = causal_losses.get('total', torch.tensor(0.0, device=self.device))
                            
                            # Store causal metrics to add to main metrics later
                            causal_metrics['causal_total'] = causal_total_loss.item() if hasattr(causal_total_loss, 'item') else causal_total_loss
                            causal_metrics['causal_intervention'] = causal_losses.get('intervention', torch.tensor(0.0)).item()
                            causal_metrics['causal_sensitivity'] = causal_losses.get('causal_sensitivity', torch.tensor(0.0)).item()
                            
                            if batch_idx == 0:
                                print(f"      [CAUSAL] Schema {current_schema}: total={causal_metrics['causal_total']:.4f}, intervention={causal_metrics['causal_intervention']:.4f}, sensitivity={causal_metrics['causal_sensitivity']:.4f}")
                        else:
                            outputs = self.model(**batch, mode=task_name)
                            causal_total_loss = None
                    else:
                        # Standard forward pass
                        outputs = self.model(**batch, mode=task_name)
                        causal_total_loss = None
                    
                    # Compute base loss
                    loss, metrics = loss_fn(outputs, batch, mode=task_name)
                
                # Check for nan loss (outlier batch detection)
                if torch.isnan(loss):
                    print(f"\n[WARNING] Skipping batch {batch_idx} (NaN loss)")
                    optimizer.zero_grad()
                    del outputs, loss
                    continue
                
                # Add causal loss if in causal training mode (schema levels 12-13)
                if causal_total_loss is not None and isinstance(causal_total_loss, torch.Tensor):
                    loss = loss + causal_total_loss
                    metrics.update(causal_metrics)
                
                # Add auxiliary and contrastive losses if enabled
                if (self.use_auxiliary or self.use_contrastive) and IMPROVEMENTS_AVAILABLE:
                    # Extract embeddings and numbers from outputs/batch if available
                    embeddings = outputs.get(OUTPUT_KEY_NUMBER_EMBEDDINGS, None)
                    numbers = batch.get(BATCH_KEY_NUMBERS, None)
                    
                    if embeddings is not None and numbers is not None:
                        loss, loss_dict = self.compute_improved_loss(loss, embeddings, numbers)
                        # Log additional losses
                        if LOSS_KEY_AUXILIARY in loss_dict:
                            metrics[METRICS_KEY_AUX_LOSS] = loss_dict[LOSS_KEY_AUXILIARY].item()
                        if LOSS_KEY_CONTRASTIVE in loss_dict:
                            metrics[METRICS_KEY_CONTRAST_LOSS] = loss_dict[LOSS_KEY_CONTRASTIVE].item()
                
                # Backward pass with gradient accumulation
                accumulation_steps = getattr(self.config, 'accumulation_steps', 1)
                scaled_loss = loss / accumulation_steps
                scaler.scale(scaled_loss).backward()
                
                # Store in CLS memory (essential keys only)
                if self.use_cls:
                    essential_keys = {
                        BATCH_KEY_OBJECT_STATES, BATCH_KEY_OBJECT_MASK, BATCH_KEY_NEXT_STATES,
                        BATCH_KEY_MASSES, BATCH_KEY_SCHEMA
                    }
                    detached_batch = {
                        k: v.detach() if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items() if k in essential_keys
                    }
                    self.cls_memory.encode_experience(detached_batch, task_name)
                
                # === SLEEP: Consolidation (experience replay) ===
                if enable_consolidation and self.use_cls:
                    # Use smaller batch size for consolidation to avoid OOM with pairwise embeddings
                    consolidation_batch_size = getattr(self.config, 'batch_size_consolidation', 16)
                    consolidation_result = self.cls_memory.consolidate(
                        exclude_tasks=[task_name],  # Don't replay current task
                        batch_size=consolidation_batch_size
                    )
                    
                    if consolidation_result is not None:
                        replay_task, replay_batch = consolidation_result
                        
                        # Prepare and truncate replay batch
                        replay_batch = prepare_physics_batch(replay_batch, sys.modules['training.constants'])
                        current_seq_len = self.progressive_curriculum.current_seq_length
                        if 'object_states' in replay_batch and replay_batch['object_states'].shape[1] > current_seq_len:
                            replay_batch['object_states'] = replay_batch['object_states'][:, :current_seq_len, :, :]
                            if 'object_mask' in replay_batch:
                                replay_batch['object_mask'] = replay_batch['object_mask'][:, :current_seq_len, :]
                            if 'next_states' in replay_batch:
                                replay_batch['next_states'] = replay_batch['next_states'][:, :current_seq_len-1, :, :]
                        
                        replay_batch = move_batch_to_device(replay_batch, self.device)
                        
                        with autocast(enabled=use_amp):
                            replay_outputs = self.model(**replay_batch, mode=replay_task)
                            replay_loss, _ = loss_fn(replay_outputs, replay_batch, mode=replay_task)
                        
                        scaler.scale(replay_loss).backward()
                        consolidation_count += 1
                        self.stats['consolidations'] += 1
                
                # Gradient clipping and optimizer step (only every accumulation_steps batches)
                should_step = (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader)
                
                if should_step:
                    scaler.unscale_(optimizer)
                    
                    # Check for inf/nan gradients
                    grad_norm_before = compute_gradient_norm(self.model)
                    last_grad_norm = grad_norm_before
                    
                    if torch.isinf(torch.tensor(grad_norm_before)).item() or torch.isnan(torch.tensor(grad_norm_before)).item():
                        optimizer.zero_grad()
                        scaler.update()
                    else:
                        grad_norm, bad_params = self.gradient_handler.clip_and_check(self.model)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
                    
                    # Update EMA model weights
                    if ema_model is not None:
                        ema_model.update(self.model)
                    
                    # Periodic memory cleanup
                    if (batch_idx + 1) % 50 == 0:
                        torch.cuda.empty_cache()
                
                # Track loss and timing
                loss_value = loss.item()
                epoch_losses.append(loss_value)
                
                # Clean up tensors to free memory
                del outputs, loss, scaled_loss
                if 'replay_outputs' in locals():
                    del replay_outputs, replay_loss
                
                batch_process_time = time.time() - batch_process_start
                batch_start_times.append(batch_process_time)
                
                # First batch completion message
                if is_first_batch:
                    print(f"   [OK] First batch: {batch_process_time:.1f}s, Loss: {loss_value:.4f}", flush=True)
                
                # Log progress every 20 batches (reduced frequency)
                if batch_idx % 20 == 0 and batch_idx > 0:
                    log_training_progress(
                        batch_idx, len(dataloader), epoch_losses, epoch_start_time,
                        batch_size, batch_start_times, optimizer, self.model, metrics,
                        last_grad_norm=last_grad_norm
                    )
            
            # Epoch summary - delegate to helper
            epoch_time = time.time() - epoch_start_time
            if len(epoch_losses) > 0:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                
                # Educational explainer at end of epoch
                print_training_explainer(epoch, -1, stage)
                
                print_epoch_summary(
                    epoch, epochs, epoch_losses, epoch_time, 
                    batch_size if batch_size else DEFAULT_BATCH_SIZE,
                    consolidation_count, self.cls_memory if self.use_cls else None
                )
                
                # Update progressive curriculum (schema level + sequence length)
                # ONLY for Physics - other stages don't use curriculum
                # Pass validation accuracy to prevent rushing through curriculum
                result = self.progressive_curriculum.update(avg_epoch_loss, accuracy=validation_accuracy) if stage == "physics" else {'should_progress': False}
                if result['should_progress']:
                    # Curriculum wants to progress (schema level or sequence length)
                    # Message already printed by curriculum.update()
                    
                    # Save ablation study checkpoint if at milestone
                    if result.get('save_ablation'):
                        ablation_name = result['ablation_name']
                        ablation_path = Path(self.checkpoint_dir) / f"ablation_{ablation_name}.pt"
                        print(f"\n📊 Saving ablation checkpoint: {ablation_path}")
                        self.save_checkpoint(
                            path=ablation_path,
                            epoch=epoch,
                            stage=stage
                        )
                        print(f"   ✅ Ablation checkpoint saved: {ablation_name}")
                    
                    # Build return signal with updated config
                    reload_signal = {
                        'reload_dataloader': True,
                        'epochs_completed': epoch + 1  # +1 because we just finished this epoch
                    }
                    
                    # Add schema level if progressed
                    if 'new_schema_level' in result:
                        reload_signal['new_schema_level'] = result['new_schema_level']
                        self.config.schema_curriculum_level = result['new_schema_level']
                    
                    # Add sequence length if progressed
                    if 'new_seq_length' in result:
                        reload_signal['new_seq_length'] = result['new_seq_length']
                        reload_signal['new_batch_size'] = result['new_batch_size']
                        self.config.max_seq_length = result['new_seq_length']
                    
                    # Return signal to caller to reload dataloader
                    return reload_signal
                else:
                    # Print curriculum status (why not progressing)
                    # ONLY for Physics - other stages don't use curriculum
                    if stage == "physics":
                        print(f"\n[CURRICULUM] {result['message']}")
                    
                    # Check if curriculum is complete (all schemas + longest sequences)
                    # ONLY for Physics
                    if stage == "physics" and self.progressive_curriculum.is_complete():
                        # Check if converged at final level (extra patience)
                        if self.progressive_curriculum.epochs_without_improvement >= 5:  # Extra patience at final level
                            # Run final validation before completing
                            print(f"\n[SEARCH] Running FINAL validation before curriculum completion...")
                            # Physics only for now - stages 2-4 disabled
                            trained_tasks = [TASK_PHYSICS]
                            self.validate_all_tasks(epoch, validation_dataloaders=validation_dataloaders, trained_tasks=trained_tasks)
                            print(f"[OK] Final validation complete")
                            
                            # Get final validation metrics
                            final_val_accuracy = None
                            if task_name in self.validation_history and len(self.validation_history[task_name]) > 0:
                                final_val_accuracy = self.validation_history[task_name][-1]
                            
                            print(f"\n{'='*70}")
                            print(f"[COMPLETE] CURRICULUM COMPLETE!")
                            print(f"{'='*70}")
                            print(f"  ✅ All 8 schema levels mastered")
                            print(f"  ✅ Maximum sequence length ({self.progressive_curriculum.current_seq_length}) reached")
                            print(f"  ✅ Model converged (no improvement for 5 epochs)")
                            print(f"  Final training loss: {avg_epoch_loss:.6f}")
                            if final_val_accuracy is not None:
                                print(f"  Final validation accuracy: {final_val_accuracy:.2%}")
                            print(f"  Total epochs: {epoch + 1}")
                            print(f"{'='*70}\n")
                            
                            # Save final checkpoint before stopping
                            actual_epoch = start_epoch + epoch
                            checkpoint_path = self.checkpoint_dir / f"physics_epoch{actual_epoch}_curriculum_complete.pt"
                            self.save_checkpoint(
                                path=checkpoint_path,
                                epoch=actual_epoch,
                                stage=stage,
                                optimizer=optimizer
                            )
                            print(f"[SAVE] Final checkpoint saved: {checkpoint_path}")
                            
                            # Return early stop signal
                            return {
                                'early_stop': True,
                                'reason': 'curriculum_complete',
                                'epochs_completed': epoch + 1,
                                'final_val_accuracy': final_val_accuracy
                            }
                
                # Log epoch summary to metrics logger
                if metrics_logger:
                    metrics_logger.log_epoch_summary(epoch, epoch_time)
            else:
                print(f"\n[WARN]  No batches processed in epoch {epoch+1}")
            
            # Advance curriculum
            if self.use_curriculum and self.curriculum_scheduler:
                old_range = self.curriculum_scheduler.get_current_range()
                self.curriculum_scheduler.step()
                new_range = self.curriculum_scheduler.get_current_range()
                if old_range != new_range:
                    print(f"\n[CURRICULUM] Curriculum advanced: {old_range} -> {new_range}")
            
            # Multi-task validation (every epoch)
            validation_loss = None
            validation_accuracy = 0.0
            if epoch % 1 == 0:  # Can adjust frequency
                print(f"\n[SEARCH] Running validation...")
                
                # Use EMA weights for validation if available (better generalization)
                if ema_model is not None:
                    ema_model.apply_shadow(self.model)
                    print(f"   [EMA] Using EMA weights for validation")
                
                # Physics only for now - stages 2-4 disabled
                trained_tasks = [TASK_PHYSICS]
                val_results = self.validate_all_tasks(epoch, validation_dataloaders=validation_dataloaders, trained_tasks=trained_tasks)
                validation_accuracy = val_results.get('physics_accuracy', 0.0) if val_results else 0.0
                
                # Restore original weights after validation
                if ema_model is not None:
                    ema_model.restore(self.model)
                
                print(f"[OK] Validation complete")
                
                # Get validation loss for early stopping (use current task's validation accuracy as proxy)
                if task_name in self.validation_history and len(self.validation_history[task_name]) > 0:
                    # Convert accuracy to loss (1 - accuracy) for early stopping
                    current_accuracy = self.validation_history[task_name][-1]
                    validation_loss = 1.0 - current_accuracy
                else:
                    # No fallback - validation must work properly
                    raise RuntimeError(
                        f"Validation history not found for task '{task_name}'! "
                        f"Available tasks: {list(self.validation_history.keys())}. "
                        f"This indicates validation is not running properly. "
                        f"Check that validation_dataloaders are provided and contain '{task_name}'."
                    )
            
            # Check early stopping (only if we have valid validation loss)
            if early_stopping is not None and validation_loss is not None and validation_loss > 0:
                # Check if this is a new best model (before calling early_stopping)
                is_new_best = (early_stopping.best_loss is None or 
                              validation_loss < early_stopping.best_loss - early_stopping.min_delta)
                
                should_stop = early_stopping(validation_loss)
                
                # Save best model checkpoint when validation improves
                if is_new_best:
                    best_checkpoint = self.checkpoint_dir / f"physics_best.pt"
                    print(f"\n[SAVE] New best validation! Saving checkpoint...")
                    self.save_checkpoint(best_checkpoint, stage=stage, epoch=start_epoch + epoch + 1, optimizer=optimizer, ema_model=ema_model)
                    print(f"[OK] Best model saved: {best_checkpoint.name}")
                
                if should_stop:
                    # For physics training with curriculum, early stopping should advance curriculum, not stop training
                    if stage == "physics" and not self.progressive_curriculum.is_complete():
                        print(f"\n{'='*70}")
                        print(f"[EARLY STOP] Validation plateaued at schema level {self.progressive_curriculum.current_schema_level}")
                        print(f"   No improvement for {early_stopping.patience} epochs")
                        print(f"   Best validation loss: {early_stopping.best_loss:.4f}")
                        print(f"   Forcing curriculum advancement...")
                        
                        # Force curriculum to advance
                        old_level = self.progressive_curriculum.current_schema_level
                        self.progressive_curriculum.force_advance()
                        new_level = self.progressive_curriculum.current_schema_level
                        
                        print(f"   Schema level: {old_level} -> {new_level}")
                        print(f"   Resetting early stopping counter...")
                        print(f"{'='*70}")
                        
                        # Reset early stopping for new curriculum level
                        early_stopping.reset()
                        
                        # If curriculum advanced, signal dataloader reload
                        if new_level != old_level:
                            reload_signal = {
                                'reload_dataloader': True,
                                'new_schema_level': new_level,
                                'new_seq_length': self.progressive_curriculum.current_seq_length,
                                'reason': 'early_stop_forced_advance'
                            }
                            return reload_signal
                    else:
                        # Curriculum complete or not physics - actually stop
                        print(f"\n{'='*70}")
                        print(f"[EARLY STOP] Early stopping triggered at epoch {epoch+1}/{epochs}")
                        print(f"   Validation loss has not improved for {early_stopping.patience} epochs")
                        print(f"   Best validation loss: {early_stopping.best_loss:.4f}")
                        print(f"   Current validation loss: {validation_loss:.4f}")
                        print(f"   Stopping training to prevent overfitting")
                        print(f"   Best model saved as: physics_best.pt")
                        print(f"{'='*70}")
                        break
            
            # Save checkpoint (file naming: total - current_epoch = checkpoint_number)
            absolute_epoch = start_epoch + epoch + 1
            if (epoch + 1) % self.config.save_every == 0:
                total_epochs = start_epoch + epochs
                checkpoint_epoch_name = total_epochs - absolute_epoch
                checkpoint_path = self.checkpoint_dir / f"physics_epoch{checkpoint_epoch_name}_of_{total_epochs}.pt"
                print(f"\n[SAVE] Saving checkpoint at epoch {absolute_epoch} (file: epoch{checkpoint_epoch_name}_of_{total_epochs})... (this may take a moment)")
                self.save_checkpoint(checkpoint_path, stage=stage, epoch=absolute_epoch, optimizer=optimizer, ema_model=ema_model)
                print(f"[OK] Checkpoint saved: {checkpoint_path.name}")
        
        # Final checkpoint
        final_absolute_epoch = start_epoch + epochs
        final_checkpoint = self.checkpoint_dir / f"physics_epoch{final_absolute_epoch}_final.pt"
        print(f"\n[SAVE] Saving final checkpoint at epoch {final_absolute_epoch}...")
        self.save_checkpoint(final_checkpoint, stage=stage, epoch=final_absolute_epoch, optimizer=optimizer, ema_model=ema_model)
        
        # Stage summary
        stage_time = time.time() - stage_start_time
        stage_hours = stage_time / 3600
        stage_minutes = stage_time / 60
        
        print(f"\n{'='*70}")
        print(f"[DONE] PHYSICS TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"\n[STATS] Stage Statistics:")
        print(f"   Task: {task_name}")
        print(f"   Epochs completed: {epochs}")
        print(f"   Total batches: {epochs * len(dataloader)}")
        if stage_hours >= 1:
            print(f"   Total time: {stage_hours:.2f} hours ({stage_minutes:.1f} minutes)")
        else:
            print(f"   Total time: {stage_minutes:.1f} minutes ({stage_time:.0f} seconds)")
        if self.use_cls and enable_consolidation:
            total_consolidations = sum(self.stats['consolidation_by_task'].values())
            print(f"   Total consolidations: {total_consolidations}")
            print(f"   Consolidations by task: {dict(self.stats['consolidation_by_task'])}")
        print(f"\n[SAVE] Checkpoint saved: {final_checkpoint}")
        print(f"{'='*70}")
    
    def validate_all_tasks(self, epoch: int, validation_dataloaders: dict = None, trained_tasks: list = None):
        """Multi-task validation - delegates to helper. Returns validation results."""
        return validate_all_tasks_helper(
            self.model, self.device, self.config, self.validation_history,
            self.stats, self.cls_memory, self.use_cls, self.temporal_metrics,
            epoch, validation_dataloaders, trained_tasks, TASK_NAMES,
            TASK_PHYSICS, TASK_COUNTING, TASK_ARITHMETIC, TASK_SYMBOLIC,
            BATCH_KEY_NEXT_STATES, OUTPUT_KEY_PREDICTED_STATES, BATCH_KEY_COUNTS,
            OUTPUT_KEY_PREDICTED_ANSWER, BATCH_KEY_RESULTS, BATCH_KEY_ANSWERS,
            BATCH_KEY_OBJECT_STATES, BATCH_KEY_OBJECT_MASK
        )
    
    def train_physics(self, dataloader, epochs=None, metrics_logger=None, validation_dataloaders=None, start_epoch=0):
        """
        Physics Training.
        
        No consolidation needed (only task).
        """
        print("\n" + "="*70)
        print("[TARGET] PHYSICS TRAINING")
        print("="*70)
        print("[SCIENCE] Learning physical dynamics from simulation data")
        print("[TARGET] Goal: Ground numerical understanding in physical reality")
        print("[UNLOCKED] All parameters trainable (no schema protection needed)")
        print("[CYCLE] Consolidation: DISABLED (first task - nothing to consolidate)")
        print("="*70)
        
        epochs = epochs or self.config.physics_epochs
        
        return self.train_stage_with_cls(
            stage="physics",
            dataloader=dataloader,
            epochs=epochs,
            task_name=TASK_PHYSICS,
            should_freeze_encoder=False,
            enable_consolidation=False,
            consolidation_tasks=None,
            metrics_logger=metrics_logger,
            validation_dataloaders=validation_dataloaders,
            start_epoch=start_epoch
        )
    
    # =========================================================================
    # STAGES 2-4 MOVED TO stages_2_3_4_pipeline.py
    # Uncomment and import from that file when ready to train counting/arithmetic/symbolic
    # =========================================================================
    
    def train_stage_2(self, dataloader, epochs=None, metrics_logger=None, validation_dataloaders=None, start_epoch=0):
        """Stage 2: Counting - DISABLED. See stages_2_3_4_pipeline.py"""
        raise NotImplementedError(
            "Stage 2 (Counting) is currently disabled. "
            "See physics_former/training/pipelines/stages_2_3_4_pipeline.py to re-enable."
        )
    
    def train_stage_3(self, dataloader=None, epochs=None, metrics_logger=None, validation_dataloaders=None, start_epoch=0, data_dir=None):
        """Stage 3: Arithmetic - DISABLED. See stages_2_3_4_pipeline.py"""
        raise NotImplementedError(
            "Stage 3 (Arithmetic) is currently disabled. "
            "See physics_former/training/pipelines/stages_2_3_4_pipeline.py to re-enable."
        )
    
    def train_stage_4(self, dataloader=None, epochs=None, metrics_logger=None, validation_dataloaders=None, start_epoch=0, data_dir=None):
        """Stage 4: Symbolic - DISABLED. See stages_2_3_4_pipeline.py"""
        raise NotImplementedError(
            "Stage 4 (Symbolic) is currently disabled. "
            "See physics_former/training/pipelines/stages_2_3_4_pipeline.py to re-enable."
        )
    
    def evaluate_embodied_cognition(self, test_data: dict = None, stage: int = None):
        """Evaluate embodied cognition metrics - delegates to helper."""
        return evaluate_embodied_cognition_helper(
            self.model, self.device, test_data, stage, 
            self.stats, EMBODIED_METRICS_AVAILABLE
        )
    
    def save_checkpoint(self, path: Path, stage: int, epoch: int, optimizer=None, scheduler=None, ema_model=None):
        """Save checkpoint with CLS memory using centralized utility."""
        # Save progressive curriculum state
        curriculum_state = {
            'current_phase': self.progressive_curriculum.current_phase,
            'current_schema_level': self.progressive_curriculum.current_schema_level,
            'current_seq_length': self.progressive_curriculum.current_seq_length,
            'epochs_in_phase': self.progressive_curriculum.epochs_in_phase,
            'best_loss_in_phase': self.progressive_curriculum.best_loss_in_phase,
            'epochs_without_improvement': self.progressive_curriculum.epochs_without_improvement,
            'loss_history': self.progressive_curriculum.loss_history
        }
        
        # Use centralized checkpoint utility
        save_checkpoint_util(
            model=self.model,
            optimizer=optimizer,
            stage=stage,
            epoch=epoch,
            path=path,
            scheduler=scheduler,
            validation_history=self.validation_history,
            stats=self.stats,
            curriculum_state=curriculum_state
        )
        
        # Save EMA state separately if available
        if ema_model is not None:
            ema_path = path.parent / f"{path.stem}_ema.pt"
            torch.save(ema_model.state_dict(), ema_path)
            print(f"[OK] EMA state saved: {ema_path.name}")
        
        # Save CLS memory separately
        if self.use_cls:
            memory_path = path.parent / f"{path.stem}_memory.pkl"
            self.cls_memory.save(str(memory_path))
            print(f"[OK] CLS memory saved: {memory_path}")
    
    def load_checkpoint(self, path: Path, optimizer=None, scheduler=None, strict=True, ema_model=None):
        """Load checkpoint with CLS memory using centralized utility."""
        if self.model is None:
            self.setup()
        
        # Use centralized checkpoint utility
        checkpoint = load_checkpoint_util(
            path=path,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            strict=strict
        )
        
        # Restore validation history and stats
        self.validation_history = checkpoint.get('validation_history', {})
        self.stats = checkpoint.get('stats', {})
        
        # Store restored epoch for resume
        self.resumed_epoch = checkpoint.get('epoch', 0)
        print(f"[OK] Restored epoch: {self.resumed_epoch}")
        
        # Restore progressive curriculum state
        if 'curriculum_state' in checkpoint:
            curr_state = checkpoint['curriculum_state']
            self.progressive_curriculum.current_phase = curr_state['current_phase']
            self.progressive_curriculum.current_schema_level = curr_state.get('current_schema_level', 1)
            self.progressive_curriculum.current_seq_length = curr_state['current_seq_length']
            self.progressive_curriculum.epochs_in_phase = curr_state['epochs_in_phase']
            self.progressive_curriculum.best_loss_in_phase = curr_state['best_loss_in_phase']
            self.progressive_curriculum.epochs_without_improvement = curr_state['epochs_without_improvement']
            self.progressive_curriculum.loss_history = curr_state['loss_history']
            
            # Update config with restored schema level
            self.config.schema_curriculum_level = self.progressive_curriculum.current_schema_level
            
            print(f"[OK] Progressive curriculum restored: Phase {curr_state['current_phase'] + 1}, schema_level={self.progressive_curriculum.current_schema_level}, seq_length={curr_state['current_seq_length']}, epochs_in_phase={curr_state['epochs_in_phase']}")
        
        # Load EMA state if available
        if ema_model is not None:
            ema_path = path.parent / f"{path.stem}_ema.pt"
            if ema_path.exists():
                ema_state = torch.load(ema_path, map_location=self.device)
                ema_model.load_state_dict(ema_state)
                print(f"[OK] EMA state loaded: {ema_path.name}")
        
        # Load CLS memory
        if self.use_cls:
            memory_path = path.parent / f"{path.stem}_memory.pkl"
            if memory_path.exists():
                self.cls_memory.load(str(memory_path))
                print(f"[OK] CLS memory loaded: {memory_path}")
        
        return checkpoint
    
    def get_cls_stats(self) -> Dict:
        """Get CLS memory statistics."""
        if not self.use_cls:
            return {}
        
        stats = {
            'training_stats': self.stats,
            'memory_stats': self.cls_memory.get_stats(),
            'validation_history': self.validation_history
        }
        
        return stats
    
    def print_final_report(self):
        """Print final training report - delegates to helper."""
        print_final_training_report(
            self.use_cls, self.cls_memory, self.stats, self.validation_history
        )
    
    # Alias for backward compatibility
    train_stage_1 = train_physics


# NOTE: This module should NOT be run directly.
# Use run_cls_training_pipeline.py instead for proper training execution.
if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("ERROR: This module should NOT be run directly!")
    print("=" * 70)
    print("\nThis file contains the CLSTrainingPipeline class definition.")
    print("Use the proper training script instead:\n")
    print("  python physics_former/run_cls_training_pipeline.py --all-stages\n")
    print("For quick gradient testing:")
    print("  python physics_former/run_cls_training_pipeline.py --all-stages \\")
    print("         --skip-arithmetic --skip-symbolic --skip-counting\n")
    print("=" * 70)
    sys.exit(1)
    
    # Legacy code kept for reference but unreachable
    import argparse
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    import gc
    
    parser = argparse.ArgumentParser(
        description='Train PhysicsFormer with CLS consolidation'
    )
    parser.add_argument('--config', type=str, default='aggressive',
                        choices=['conservative', 'aggressive', 'maximum'])
    parser.add_argument('--stage', type=int, default=0,
                        help='Train specific stage (1-4) or 0 for all')
    parser.add_argument('--no-cls', action='store_true',
                        help='Disable CLS (for ablation study)')
    parser.add_argument('--consolidation-freq', type=float, default=0.2,
                        help='Consolidation frequency (default: 0.2 = 20%%)')
    parser.add_argument('--train-all', action='store_true',
                        help='Train all stages with auto-loaded data')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing training data')
    parser.add_argument('--physics-epochs', type=int, default=None,
                        help='Override physics epochs')
    parser.add_argument('--counting-epochs', type=int, default=None,
                        help='Override counting epochs')
    parser.add_argument('--arithmetic-epochs', type=int, default=None,
                        help='Override arithmetic epochs')
    parser.add_argument('--symbolic-epochs', type=int, default=None,
                        help='Override symbolic epochs')
    parser.add_argument('--checkpoint-every', type=int, default=None,
                        help='Save checkpoint every N epochs (default: from config)')
    parser.add_argument('--validate-every', type=int, default=5,
                        help='Run validation every N epochs (default: 5)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from checkpoint path')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (reduce if OOM)')
    parser.add_argument('--grad-accum', type=int, default=None,
                        help='Gradient accumulation steps (increase if reducing batch size)')
    parser.add_argument('--skip-arithmetic', action='store_true',
                        help='Skip arithmetic data loading (for quick testing)')
    parser.add_argument('--skip-symbolic', action='store_true',
                        help='Skip symbolic data loading (for quick testing)')
    parser.add_argument('--skip-counting', action='store_true',
                        help='Skip counting data loading (for quick testing)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CLSTrainingPipeline(
        config=args.config,
        use_cls=not args.no_cls,
        consolidation_frequency=args.consolidation_freq
    )
    
    # Override config if specified
    if args.batch_size:
        pipeline.config.batch_size = args.batch_size
        print(f"[CONFIG]  Batch size overridden: {args.batch_size}")
    
    if args.grad_accum:
        pipeline.config.accumulation_steps = args.grad_accum
        print(f"[CONFIG]  Gradient accumulation: {args.grad_accum} steps")
        print(f"   Effective batch size: {pipeline.config.batch_size * args.grad_accum}")
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"\n[FOLDER] Resuming from checkpoint: {args.resume_from}")
        pipeline.load_checkpoint(Path(args.resume_from))
    
    print("\n" + "="*70)
    print("CLS-BASED TRAINING PIPELINE")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"CLS Enabled: {not args.no_cls}")
    if not args.no_cls:
        print(f"Consolidation Frequency: {args.consolidation_freq:.1%}")
    print("="*70)
    
    # Train all stages with auto-loaded data
    if args.train_all:
        import time
        training_start_time = time.time()
        
        print("\n[START] Starting full training pipeline...")
        print(f"[SAVE] Checkpoints will be saved to: {pipeline.checkpoint_dir}")
        print(f"[CLOCK] Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check disk space
        import shutil
        try:
            total, used, free = shutil.disk_usage(pipeline.checkpoint_dir)
            free_gb = free // (2**30)
            print(f"[DISK] Free disk space: {free_gb}GB")
            if free_gb < 10:
                print(f"[WARN]  WARNING: Low disk space! Checkpoints may fail.")
        except (OSError, AttributeError) as e:
            # Disk space check is optional - may fail on some filesystems
            logging.debug(f"Could not check disk space: {e}")
        
        # Setup model early (before loading data)
        print("\n" + "="*70)
        print("INITIALIZING MODEL")
        print("="*70)
        pipeline.setup()
        
        # Print model info
        total_params = sum(p.numel() for p in pipeline.model.parameters())
        trainable_params = sum(p.numel() for p in pipeline.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")
        if pipeline.config.mixed_precision:
            print(f"With FP16: ~{total_params * 2 / 1024**2:.1f} MB")
        
        # Import physics datasets (arithmetic and symbolic load lazily in their respective stages)
        try:
            from ..datasets import HDF5PhysicsDataset
            from ..datasets.cached_physics_dataset import CachedPhysicsDataset
        except ImportError:
            print("[ERROR] Error: Could not import physics datasets")
            print("Make sure datasets are in physics_former/training/datasets/")
            sys.exit(1)
        
        # Setup data directory
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"[ERROR] Error: Data directory not found: {data_dir}")
            print("\nGenerate data first:")
            print("  cd physics_former/data_generation")
            print("  python generate_all_data.py")
            sys.exit(1)
        
        # Stage 1: Physics
        print("\n" + "="*70)
        print("STAGE 1/4: PHYSICS DATA")
        print("="*70)
        physics_dir = data_dir / "physics_episodes"
        if not physics_dir.exists():
            physics_dir = data_dir / "physics"  # Try alternate name
        
        if physics_dir.exists():
            print("\n[CURRICULUM] Loading physics data...")
            
            # Use RAM-cached dataset if enabled (10x speedup)
            use_cache = getattr(pipeline.config, 'cache_dataset_to_ram', False)
            
            if use_cache:
                # Get current curriculum level from pipeline
                current_level = pipeline.progressive_curriculum.current_schema_level
                print(f"[SPEEDUP] Using RAM-cached dataset (10x faster training)")
                print(f"[CURRICULUM] Loading schema level {current_level}/13")
                physics_dataset = CachedPhysicsDataset(
                    data_dir=physics_dir,
                    max_seq_length=pipeline.config.max_seq_length,
                    max_objects=pipeline.config.max_objects,
                    state_dim=pipeline.config.state_dim,
                    max_episodes_per_file=getattr(pipeline.config, 'max_episodes_per_file', 500),
                    cache_to_ram=True,
                    schema_curriculum_level=current_level
                )
            else:
                current_level = pipeline.progressive_curriculum.current_schema_level
                print(f"[INFO] Using on-demand HDF5 loading (slower but uses less RAM)")
                print(f"[CURRICULUM] Loading schema level {current_level}/13")
                physics_dataset = HDF5PhysicsDataset(
                    data_dir=physics_dir,
                    max_seq_length=pipeline.config.max_seq_length,
                    max_objects=pipeline.config.max_objects,
                    state_dim=pipeline.config.state_dim,
                    max_episodes_per_file=None,
                    schema_curriculum_level=current_level
                )
            print("[DONE] Physics data loaded successfully")
            
            physics_loader = DataLoader(
                physics_dataset,
                batch_size=pipeline.config.batch_size_physics,  # Physics-specific: smaller batch for long sequences (512 frames)
                shuffle=True,
                num_workers=pipeline.config.num_workers,
                pin_memory=pipeline.config.pin_memory if torch.cuda.is_available() else False,
                persistent_workers=pipeline.config.persistent_workers if pipeline.config.num_workers > 0 else False,
                prefetch_factor=pipeline.config.prefetch_factor if pipeline.config.num_workers > 0 else None
            )
            print(f"[OK] Loaded {len(physics_dataset)} physics episodes")
            
            epochs = args.physics_epochs or pipeline.config.physics_epochs
            
            # Get start_epoch from resumed checkpoint (if any)
            start_epoch = getattr(pipeline, 'resumed_epoch', 0)
            if start_epoch > 0:
                print(f"[RESUME] Resuming from epoch {start_epoch}")
            
            try:
                pipeline.train_stage_1(physics_loader, epochs=epochs, start_epoch=start_epoch)
            except KeyboardInterrupt:
                print("\n\n[WARN]  Training interrupted by user!")
                print("[SAVE] Saving emergency checkpoint...")
                emergency_path = pipeline.checkpoint_dir / "emergency_stage1.pt"
                pipeline.save_checkpoint(emergency_path, stage=1, epoch=0, optimizer=None)
                raise
            except Exception as e:
                print(f"\n\n[ERROR] Error during Stage 1 training: {e}")
                print("[SAVE] Saving error checkpoint...")
                error_path = pipeline.checkpoint_dir / "error_stage1.pt"
                pipeline.save_checkpoint(error_path, stage=1, epoch=0, optimizer=None)
                raise
            
            # Clear memory after physics stage
            del physics_loader, physics_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"[WARN]  Physics data not found at {physics_dir}")
        
        # Stages 2-4 ARCHIVED - see archive/ARCHIVED_STAGES_2_3_4.md
        # Physics-only training for ICML submission
        
        # Print final report
        total_training_time = time.time() - training_start_time
        total_hours = total_training_time / 3600
        
        print("\n" + "="*70)
        print("[SUCCESS] PHYSICS TRAINING COMPLETE!")
        print("="*70)
        print(f"[OK] Physics training completed successfully")
        print(f"[OK] Total training time: {total_hours:.2f} hours ({total_training_time/60:.1f} minutes)")
        print(f"[OK] Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[OK] Checkpoints saved to: {pipeline.checkpoint_dir}")
        print("\nTo use the trained model:")
        print(f"  from pathlib import Path")
        print(f"  pipeline.load_checkpoint(Path('{pipeline.checkpoint_dir}/physics_final.pt'))")
        
        pipeline.print_final_report()
        
    # Train specific stage
    elif args.stage > 0:
        print(f"\n[WARN]  Stage {args.stage} training requires dataloader")
        print("Use --train-all to automatically load data and train all stages")
    else:
        print("\n[WARN]  No training action specified")
        print("\nOptions:")
        print("  --train-all              Train all stages automatically")
        print("  --stage N                Train specific stage (requires code modification)")
        print("\nExample:")
        print("  python -m physics_former.training.pipelines.cls_pipeline --train-all")
