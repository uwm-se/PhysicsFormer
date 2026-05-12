"""
Physics-Only Training Script for PhysicsFormer

Trains PhysicsFormer on physics prediction only.
Counting, arithmetic, and symbolic stages have been archived.

This is the MAIN training script for ICML submission.
"""

import os
import sys
from pathlib import Path
import traceback
import torch

# Enable PyTorch CUDA memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent / 'training'))

from training.pipelines.cls_pipeline import CLSTrainingPipeline
from training.configs.config import TrainingConfig
from training.common import (
    setup_metrics_logger,
    log_model_metrics,
    finalize_metrics,
    print_curriculum_state,
    calculate_batch_size_for_seq_length
)


def main():
    import argparse
    from datetime import datetime
    
    # Setup error logging
    error_log_dir = os.environ.get('PHYSICS_DATA_DIR', '$PHYSICS_DATA_DIR')
    error_log_path = Path(error_log_dir) / "training_error.log"
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Train PhysicsFormer - Physics Only')
    
    # Configuration
    parser.add_argument('--config', type=str, default='aggressive',
                        choices=['conservative', 'aggressive', 'maximum', 'a100'],
                        help='Training configuration (a100 for A100 80GB GPU)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default: use config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs (default: use config)')
    
    # CLS options
    parser.add_argument('--no-cls', action='store_true',
                        help='Disable CLS (for ablation study)')
    parser.add_argument('--consolidation-freq', type=float, default=0.2,
                        help='Consolidation frequency (0.0-1.0)')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='$CHECKPOINT_DIR',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    
    # Curriculum override
    parser.add_argument('--schema-level', type=int, default=None,
                        help='Override starting schema curriculum level (1-13)')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("PHYSICSFORMER PHYSICS TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Config: {args.config}")
    print(f"  CLS Enabled: {not args.no_cls}")
    print(f"  Consolidation Frequency: {args.consolidation_freq:.1%}")
    print(f"  Checkpoint Dir: {args.checkpoint_dir}")
    print("=" * 70)
    
    # Load configuration
    if args.config == 'a100':
        from training.configs.config_a100 import A100Config
        print("\n[CONFIG] Using A100Config (optimized for A100 80GB)")
        config = A100Config()
    else:
        print("\n[CONFIG] Using TrainingConfig")
        config = TrainingConfig()
    
    # Override batch size if specified
    if args.batch_size:
        config.batch_size_physics = args.batch_size
        print(f"[CONFIG] Batch size override: {args.batch_size}")
    
    # Override epochs if specified
    if args.epochs:
        config.physics_epochs = args.epochs
        print(f"[CONFIG] Epochs override: {args.epochs}")
    
    # Initialize pipeline
    print("\n" + "=" * 70)
    print("INITIALIZING TRAINING PIPELINE")
    print("=" * 70)
    
    pipeline = CLSTrainingPipeline(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        use_cls=not args.no_cls,
        consolidation_frequency=args.consolidation_freq
    )
    
    print("[OK] Pipeline initialized")
    
    # Initialize model
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    pipeline.setup()
    print("[OK] Model initialized")
    
    # Resume from checkpoint if specified
    epoch = 0
    if args.resume:
        print("\n" + "=" * 70)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 70)
        print(f"Checkpoint: {args.resume}")
        
        resume_path = Path(args.resume)
        if not resume_path.is_absolute() and not resume_path.exists():
            checkpoint_dir_path = Path(args.checkpoint_dir) / args.resume
            if checkpoint_dir_path.exists():
                resume_path = checkpoint_dir_path
        
        checkpoint = pipeline.load_checkpoint(resume_path, strict=True)
        epoch = checkpoint.get('epoch', 0)
        
        print(f"[OK] Resumed from epoch {epoch}")
        print_curriculum_state(pipeline.progressive_curriculum)
    
    # Override schema level if specified
    if args.schema_level is not None:
        print(f"\n[OVERRIDE] Schema level -> {args.schema_level}")
        pipeline.progressive_curriculum.current_schema_level = args.schema_level
        pipeline.progressive_curriculum.current_phase = args.schema_level - 1
        pipeline.progressive_curriculum.epochs_in_phase = 0
        pipeline.progressive_curriculum.best_loss_in_phase = float('inf')
        pipeline.progressive_curriculum.epochs_without_improvement = 0
    
    # Update config with curriculum state
    seq_length = pipeline.progressive_curriculum.current_seq_length
    schema_level = pipeline.progressive_curriculum.current_schema_level
    config.max_seq_length = seq_length
    config.schema_curriculum_level = schema_level
    
    # Calculate recommended batch size
    recommended_batch = calculate_batch_size_for_seq_length(seq_length)
    if args.batch_size is None:
        config.batch_size_physics = recommended_batch
    
    print(f"\nTraining Configuration:")
    print(f"  Schema level: {schema_level}/13")
    print(f"  Sequence length: {seq_length}")
    print(f"  Batch size: {config.batch_size_physics}")
    print(f"  Epochs: {config.physics_epochs}")
    
    # Create physics dataloader
    print("\n" + "=" * 70)
    print("LOADING PHYSICS DATA")
    print("=" * 70)
    
    from training.datasets.cached_physics_dataset import CachedPhysicsDataset
    from training.datasets.hdf5_physics_dataset import HDF5PhysicsDataset
    from torch.utils.data import DataLoader
    
    physics_dir = config.physics_dir
    
    if config.cache_dataset_to_ram:
        print("[INFO] Using RAM-cached dataset (faster)")
        physics_dataset = CachedPhysicsDataset(
            data_dir=physics_dir,
            max_seq_length=config.max_seq_length,
            max_objects=config.max_objects,
            state_dim=config.state_dim,
            max_episodes_per_file=getattr(config, 'max_episodes_per_file', None),
            schema_curriculum_level=schema_level
        )
    else:
        print("[INFO] Using on-demand HDF5 loading")
        physics_dataset = HDF5PhysicsDataset(
            data_dir=physics_dir,
            max_seq_length=config.max_seq_length,
            max_objects=config.max_objects,
            state_dim=config.state_dim,
            max_episodes_per_file=None,
            schema_curriculum_level=schema_level
        )
    
    print(f"[OK] Loaded {len(physics_dataset)} physics episodes")
    
    physics_loader = DataLoader(
        physics_dataset,
        batch_size=config.batch_size_physics,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory if torch.cuda.is_available() else False,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
    )
    
    # Initialize metrics logger
    experiment_name = f"physics_{args.config}_bs{config.batch_size_physics}"
    if args.no_cls:
        experiment_name += "_no_cls"
    
    logger = setup_metrics_logger(
        experiment_name=experiment_name,
        config=config,
        log_dir="logs",
        save_interval=10,
        track_gpu=torch.cuda.is_available()
    )
    
    log_model_metrics(logger, pipeline.model)
    print(f"\n[OK] Logs: {logger.experiment_dir}")
    
    # Create validation dataloader (subset of training data)
    print(f"\n[OK] Creating validation dataloader...")
    max_val_episodes = getattr(config, 'max_episodes_per_file', None)
    if max_val_episodes:
        max_val_episodes = min(100, max_val_episodes // 5)  # Use 20% for validation
    else:
        max_val_episodes = 100  # Default validation size
    val_dataset = CachedPhysicsDataset(
        data_dir=config.physics_dir,
        max_seq_length=config.max_seq_length,
        max_objects=config.max_objects,
        state_dim=config.state_dim,
        max_episodes_per_file=max_val_episodes,
        schema_curriculum_level=1
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_physics,
        shuffle=False,  # No shuffle for validation
        num_workers=0,  # Windows compatibility - avoid multiprocessing issues
        pin_memory=False
    )
    validation_dataloaders = {'physics': val_loader}
    print(f"[OK] Validation dataset: {len(val_dataset)} episodes")
    
    # Train physics
    print("\n" + "=" * 70)
    print("STARTING PHYSICS TRAINING")
    print("=" * 70)
    
    epochs = args.epochs or config.physics_epochs
    total_epochs_completed = epoch  # Track total epochs across curriculum levels
    
    try:
        # Training loop with curriculum advancement support
        while True:
            result = pipeline.train_physics(
                physics_loader, 
                epochs=epochs, 
                metrics_logger=logger, 
                validation_dataloaders=validation_dataloaders, 
                start_epoch=total_epochs_completed
            )
            
            # Check if we need to reload dataloader for new curriculum level
            if isinstance(result, dict) and result.get('reload_dataloader'):
                new_schema_level = result.get('new_schema_level')
                epochs_completed = result.get('epochs_completed', 0)
                total_epochs_completed += epochs_completed
                reason = result.get('reason', 'curriculum_advance')
                
                print(f"\n{'='*70}")
                print(f"[CURRICULUM] Reloading dataloader for schema level {new_schema_level}")
                print(f"   Reason: {reason}")
                print(f"   Total epochs completed: {total_epochs_completed}")
                print(f"{'='*70}")
                
                # Reload training dataset with new schema level
                if getattr(config, 'cache_dataset_to_ram', True):
                    physics_dataset = CachedPhysicsDataset(
                        data_dir=physics_dir,
                        max_seq_length=config.max_seq_length,
                        max_objects=config.max_objects,
                        state_dim=config.state_dim,
                        max_episodes_per_file=getattr(config, 'max_episodes_per_file', None),
                        schema_curriculum_level=new_schema_level
                    )
                else:
                    physics_dataset = HDF5PhysicsDataset(
                        data_dir=physics_dir,
                        max_seq_length=config.max_seq_length,
                        max_objects=config.max_objects,
                        state_dim=config.state_dim,
                        max_episodes_per_file=None,
                        schema_curriculum_level=new_schema_level
                    )
                
                physics_loader = DataLoader(
                    physics_dataset,
                    batch_size=config.batch_size_physics,
                    shuffle=True,
                    num_workers=config.num_workers,
                    pin_memory=config.pin_memory if torch.cuda.is_available() else False,
                    persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
                    prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
                )
                
                # Reload validation dataset with new schema level
                val_dataset = CachedPhysicsDataset(
                    data_dir=config.physics_dir,
                    max_seq_length=config.max_seq_length,
                    max_objects=config.max_objects,
                    state_dim=config.state_dim,
                    max_episodes_per_file=max_val_episodes,
                    schema_curriculum_level=new_schema_level
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config.batch_size_physics,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False
                )
                validation_dataloaders = {'physics': val_loader}
                
                print(f"[OK] Reloaded {len(physics_dataset)} training episodes")
                print(f"[OK] Reloaded {len(val_dataset)} validation episodes")
                
                # Continue training with remaining epochs
                # Note: epochs stays the same, we continue from where we left off
                continue
            else:
                # Training complete (no reload signal)
                break
                
    except KeyboardInterrupt:
        print("\n\n[WARN] Training interrupted by user!")
        print("[SAVE] Saving emergency checkpoint...")
        emergency_path = pipeline.checkpoint_dir / "emergency_physics.pt"
        pipeline.save_checkpoint(emergency_path, stage="physics", epoch=0, optimizer=None)
        raise
    
    # Final report
    print("\n" + "=" * 70)
    print("[SUCCESS] PHYSICS TRAINING COMPLETE!")
    print("=" * 70)
    
    pipeline.print_final_report()
    finalize_metrics(logger)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARN] Training interrupted by user")
        print("[INFO] Training state saved in most recent checkpoint")
        print("[INFO] Resume with --resume flag")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] TRAINING FAILED!")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
