"""
Full Pipeline: All 4 Levels for Lakoff Validation

Level 1: Physics Prediction
Level 2: Object Counting
Level 3: Arithmetic Operations
Level 4: Symbolic Math Transfer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.physics_former_full import FullPhysicsFormer, FullPhysicsLoss
from configs.config import TrainingConfig

__all__ = ['FullPipeline']


class FullPipeline:
    """
    Complete training pipeline for all 4 levels.
    
    Usage:
        pipeline = FullPipeline(config='aggressive')
        pipeline.train_stage_1()  # Physics
        pipeline.train_stage_2()  # Counting
        pipeline.train_stage_3()  # Arithmetic
        pipeline.train_stage_4()  # Symbolic
    """
    
    def __init__(self, config='aggressive'):
        """
        Initialize pipeline.
        
        Args:
            config: 'conservative', 'aggressive', or 'maximum'
        """
        from configs.config import TrainingConfig, ConservativeConfig, MaximumConfig
        
        configs = {
            'conservative': ConservativeConfig(),
            'aggressive': TrainingConfig(),
            'maximum': MaximumConfig()
        }
        
        self.config = configs.get(config, TrainingConfig())
        self.model = None
        self.device = None
        
        print("=" * 70)
        print("FULL PIPELINE INITIALIZED")
        print("=" * 70)
        print(f"\nConfiguration: {config}")
        print(f"  hidden_dim: {self.config.hidden_dim}")
        print(f"  num_layers: {self.config.num_layers}")
        print(f"  batch_size: {self.config.batch_size}")
        print(f"  mixed_precision: {self.config.mixed_precision}")
    
    def setup(self):
        """Setup model and device."""
        import torch
        
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"\nDevice: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Create model
        self.model = FullPhysicsFormer(
            state_dim=self.config.state_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            ff_dim=self.config.ff_dim,
            max_objects=self.config.max_objects,
            num_schema_classes=self.config.num_schema_classes,
            vocab_size=self.config.vocab_size,
            dropout=self.config.dropout,
            encoder_chunk_size=getattr(self.config, 'encoder_chunk_size', 6),
            seq_chunk_size=getattr(self.config, 'seq_chunk_size', 32)
        ).to(self.device)
        
        print(f"\nPASS: Model created: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Apply modern improvements (RMSNorm, Flash Attention, RoPE)
        try:
            from ..models.apply_modern_improvements import apply_modern_improvements
            self.model = apply_modern_improvements(self.model, self.config, verbose=True)
        except ImportError as e:
            print(f"[WARN] Could not apply modern improvements: {e}")
        
        # Compile if available (Windows compatibility: may fail with dataclass errors)
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                print("Compiling model...")
                self.model = torch.compile(self.model)
                print("PASS: Model compiled")
            except (TypeError, RuntimeError, ImportError) as e:
                print(f"WARNING: Model compilation failed: {e}")
                print("Continuing without compilation (training will be slower but functional)")
        
        return self
