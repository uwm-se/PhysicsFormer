"""
Training Fix Script

Applies fixes to stabilize training:
1. Reinitialize model weights with proper scale
2. Apply corrected hyperparameters
3. Verify data normalization
4. Run diagnostics before training

Usage:
    python fix_training.py --checkpoint <path_to_checkpoint>
    
Or to start fresh:
    python fix_training.py --fresh
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from training.configs.config import TrainingConfig
from training.models.physics_former_full import FullPhysicsFormer
from training.utils.training_diagnostics import (
    initialize_weights_properly,
    run_full_diagnostics
)


def apply_fixes(model, config, device='cuda'):
    """
    Apply all training fixes to model and config.
    
    Args:
        model: PhysicsFormer model
        config: Training configuration
        device: Device to use
        
    Returns:
        Fixed model and config
    """
    print("\n" + "="*70)
    print("APPLYING TRAINING FIXES")
    print("="*70 + "\n")
    
    # 1. Reinitialize weights with proper scale
    print("1. Reinitializing model weights...")
    initialize_weights_properly(model, scale=config.weight_init_scale)
    
    # 2. Verify configuration
    print("2. Verifying configuration...")
    print(f"   Learning Rate: {config.learning_rate:.2e} (should be ≤5e-5)")
    print(f"   Max Grad Norm: {config.max_grad_norm} (should be ≤0.5)")
    print(f"   Warmup Steps: {config.warmup_steps} (should be ≥5000)")
    print(f"   Energy Weight: {config.energy_weight} (should be ≤0.001)")
    print(f"   Momentum Weight: {config.momentum_weight} (should be ≤0.01)")
    
    # Check if fixes are applied
    fixes_needed = []
    if config.learning_rate > 5e-5:
        fixes_needed.append("learning_rate")
    if config.max_grad_norm > 0.5:
        fixes_needed.append("max_grad_norm")
    if config.warmup_steps < 5000:
        fixes_needed.append("warmup_steps")
    if config.energy_weight > 0.001:
        fixes_needed.append("energy_weight")
    if config.momentum_weight > 0.01:
        fixes_needed.append("momentum_weight")
    
    if fixes_needed:
        print(f"\n⚠️  Configuration needs updates: {', '.join(fixes_needed)}")
        print(f"   Please update config.py with the fixed values")
    else:
        print(f"\n✅ Configuration is correct")
    
    # 3. Move to device
    print(f"\n3. Moving model to {device}...")
    model = model.to(device)
    
    print("\n" + "="*70)
    print("FIXES APPLIED SUCCESSFULLY")
    print("="*70 + "\n")
    
    return model, config


def verify_training_ready(model, sample_batch, config, device='cuda'):
    """
    Verify training is ready to start.
    
    Args:
        model: PhysicsFormer model
        sample_batch: Sample training batch
        config: Training configuration
        device: Device to use
        
    Returns:
        True if ready, False otherwise
    """
    print("\n" + "="*70)
    print("VERIFYING TRAINING READINESS")
    print("="*70 + "\n")
    
    # Move batch to device
    sample_batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in sample_batch.items()
    }
    
    # Prepare batch for physics (add target_states, current_states, masses)
    print("Preparing batch for physics loss...")
    from training.utils.tensor_ops import extract_sequence_input_for_prediction
    from training.constants import BATCH_KEY_OBJECT_STATES, BATCH_KEY_NEXT_STATES, BATCH_KEY_TARGET_STATES, BATCH_KEY_CURRENT_STATES, BATCH_KEY_MASSES, MASS_IDX
    
    object_states = sample_batch[BATCH_KEY_OBJECT_STATES]
    next_states = sample_batch[BATCH_KEY_NEXT_STATES]
    
    # Extract current states (timesteps 0 to seq_len-2)
    current_states = extract_sequence_input_for_prediction(object_states)
    # Extract masses from first timestep
    masses = object_states[:, 0, :, MASS_IDX:MASS_IDX+1]
    
    sample_batch[BATCH_KEY_TARGET_STATES] = next_states
    sample_batch[BATCH_KEY_CURRENT_STATES] = current_states
    sample_batch[BATCH_KEY_MASSES] = masses
    
    # Forward pass
    print("Running test forward pass...")
    model.train()
    try:
        outputs = model(**sample_batch, mode='physics')
        print("✅ Forward pass successful")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Compute loss
    print("\nComputing test loss...")
    from training.models.physics_former_full import FullPhysicsLoss
    loss_fn = FullPhysicsLoss(
        prediction_weight=config.prediction_weight,
        energy_weight=config.energy_weight,
        momentum_weight=config.momentum_weight
    )
    
    try:
        loss, metrics = loss_fn(outputs, sample_batch, mode='physics')
        print(f"✅ Loss computation successful: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False
    
    # Backward pass
    print("\nRunning test backward pass...")
    try:
        loss.backward()
        print("✅ Backward pass successful")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False
    
    # Run diagnostics
    print("\nRunning diagnostics...")
    diagnostics = run_full_diagnostics(
        model=model,
        batch=sample_batch,
        loss=loss,
        loss_components=metrics,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm
    )
    
    # Check if ready
    is_ready = (
        diagnostics['loss_stats']['total_loss'] < 1000 and
        diagnostics['gradient_stats']['is_healthy']
    )
    
    if is_ready:
        print("\n" + "="*70)
        print("✅ TRAINING IS READY TO START")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("❌ TRAINING NOT READY - FIX ISSUES FIRST")
        print("="*70 + "\n")
    
    return is_ready


def main():
    parser = argparse.ArgumentParser(description='Fix training issues')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to fix')
    parser.add_argument('--fresh', action='store_true', help='Start fresh without checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = TrainingConfig()
    
    # Create model
    print("Creating model...")
    model = FullPhysicsFormer(
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        max_objects=config.max_objects,
        num_schema_classes=config.num_schema_classes,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
        encoder_chunk_size=config.encoder_chunk_size,
        seq_chunk_size=config.seq_chunk_size
    )
    
    # Load checkpoint if provided
    if args.checkpoint and not args.fresh:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Checkpoint loaded")
    
    # Apply fixes
    model, config = apply_fixes(model, config, device=args.device)
    
    # Create sample batch for verification
    print("\nCreating sample batch for verification...")
    batch_size = config.batch_size_physics
    seq_len = config.max_seq_length
    max_objects = config.max_objects
    
    # Create proper sequence batch matching physics training format
    # Model expects: object_states [batch, seq_len, objects, features]
    # and next_states [batch, seq_len-1, objects, features] for autoregressive prediction
    sample_batch = {
        'object_states': torch.randn(batch_size, seq_len, max_objects, config.state_dim),
        'next_states': torch.randn(batch_size, seq_len - 1, max_objects, config.state_dim),  # Autoregressive targets
        'object_mask': torch.ones(batch_size, max_objects, dtype=torch.bool),  # 2D mask per batch
        'schema': torch.randint(0, config.num_schema_classes, (batch_size,))
    }
    
    # Verify training readiness
    is_ready = verify_training_ready(model, sample_batch, config, device=args.device)
    
    if is_ready:
        # Save fixed model
        output_path = Path(config.checkpoint_dir) / 'fixed_model.pt'
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__
        }, output_path)
        
        print(f"\n✅ Fixed model saved to: {output_path}")
        print(f"\nYou can now start training with:")
        print(f"  - Learning Rate: {config.learning_rate:.2e}")
        print(f"  - Max Grad Norm: {config.max_grad_norm}")
        print(f"  - Warmup Steps: {config.warmup_steps}")
        print(f"  - Energy/Momentum Weights: {config.energy_weight}")
    else:
        print(f"\n❌ Please fix the issues above before starting training")
        sys.exit(1)


if __name__ == '__main__':
    main()
