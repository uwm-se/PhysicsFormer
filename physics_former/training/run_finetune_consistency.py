"""
Fine-tune PhysicsFormer with Velocity Consistency Loss

This script fine-tunes an existing checkpoint with dual supervision:
- Delta loss: MSE(predicted_delta, target_delta) [existing]
- Consistency loss: MSE(current_vel + predicted_delta, target_absolute_vel) [new]

The consistency loss provides an additional gradient signal that anchors
velocity predictions to absolute values, preventing drift during rollouts.

Usage:
    python run_finetune_consistency.py --checkpoint path/to/checkpoint.pt
    python run_finetune_consistency.py --gamma 0.2  # Higher consistency weight
"""

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from physics_former.training.models.physics_former_full import FullPhysicsFormer, FullPhysicsLoss
from physics_former.training.datasets.hdf5_physics_dataset import HDF5PhysicsDataset
from physics_former.training.configs.config import FineTuningConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load existing PhysicsFormer checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        # Try default locations
        default_dir = Path("$CHECKPOINT_DIR")
        for name in ['stage1_best.pt', 'physics_former_best.pt', 'checkpoint_latest.pt']:
            if (default_dir / name).exists():
                checkpoint_path = default_dir / name
                break
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or in {default_dir}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Infer model config from checkpoint
    state_dict = checkpoint['model_state_dict']
    num_schema_classes = state_dict['schema_classifier.3.bias'].shape[0]
    hidden_dim = state_dict['encoder.object_encoder.0.weight'].shape[0]
    
    # Count transformer layers
    num_layers = sum(1 for k in state_dict.keys() if 'transformer_layers' in k and 'attention.q_proj.weight' in k)
    
    logger.info(f"  Detected: hidden_dim={hidden_dim}, num_layers={num_layers}, schemas={num_schema_classes}")
    
    model = FullPhysicsFormer(
        state_dim=35,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=16,
        ff_dim=hidden_dim * 4,
        max_objects=20,
        dropout=0.1,
        num_schema_classes=num_schema_classes
    ).to(device)
    
    model.load_state_dict(state_dict)
    
    # Get training info if available
    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', checkpoint.get('val_loss', float('inf')))
    
    logger.info(f"  Loaded from epoch {epoch}, best_loss={best_loss:.4f}")
    
    return model, checkpoint, epoch


def run_finetune(
    checkpoint_path: str = None,
    gamma: float = 0.1,
    max_epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    patience: int = 10,
    min_delta: float = 1e-4,
    max_grad_norm: float = 1.0,
    max_episodes: int = None,
    checkpoint_dir: str = "$CHECKPOINT_DIR",
    data_dir: str = "D:/physics_hdf5"
):
    """
    Fine-tune PhysicsFormer with velocity consistency loss.
    
    Args:
        checkpoint_path: Path to checkpoint to fine-tune (or None for auto-detect)
        gamma: Velocity consistency loss weight (0.1 = 10% of total loss)
        max_epochs: Maximum fine-tuning epochs
        batch_size: Batch size
        learning_rate: Learning rate (should be lower than original training)
        patience: Early stopping patience
        min_delta: Minimum improvement threshold
        max_grad_norm: Gradient clipping norm
        max_episodes: Limit dataset size (None = use all)
        checkpoint_dir: Where to save fine-tuned checkpoints
        data_dir: Physics data directory
    """
    print("=" * 70)
    print("FINE-TUNING WITH VELOCITY CONSISTENCY LOSS")
    print("=" * 70)
    print(f"\nConsistency loss weight (γ): {gamma}")
    print("L_total = L_delta + γ * L_consistency")
    print("where L_consistency = MSE(v_t + Δv_pred, v_t+1_target)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load checkpoint
    print("\n[1/5] Loading pre-trained checkpoint...")
    if checkpoint_path is None:
        checkpoint_path = Path(checkpoint_dir) / "stage1_best.pt"
    
    model, orig_checkpoint, start_epoch = load_checkpoint(checkpoint_path, device)
    model.train()
    
    # Create loss function with consistency loss enabled
    print("\n[2/5] Creating loss function with consistency loss...")
    loss_fn = FullPhysicsLoss(
        prediction_weight=1.0,
        energy_weight=0.001,
        momentum_weight=0.001,
        threshold_weight=0.05,
        accuracy_threshold=0.5,
        threshold_margin=0.1,
        velocity_consistency_weight=gamma  # Enable consistency loss
    )
    print(f"  velocity_consistency_weight = {gamma}")
    
    # Load dataset
    print("\n[3/5] Loading physics dataset...")
    dataset = HDF5PhysicsDataset(
        data_dir=data_dir,
        hdf5_dir=data_dir,
        max_objects=20,
        max_seq_length=128
    )
    
    if max_episodes and max_episodes < len(dataset):
        indices = torch.randperm(len(dataset))[:max_episodes].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"  Subsampled to {max_episodes} episodes")
    
    # Split into train/val
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # Setup optimizer
    print("\n[4/5] Setting up optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    scaler = GradScaler()
    
    print(f"  Learning rate: {learning_rate} (fine-tuning rate)")
    print(f"  Max grad norm: {max_grad_norm}")
    
    # Training loop
    print("\n[5/5] Fine-tuning...")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Patience: {patience}")
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = []
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_losses = {'total': 0.0, 'prediction': 0.0, 'consistency': 0.0}
        num_batches = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        for batch in progress:
            # Move to device
            object_states = batch['object_states'].to(device)
            next_states = batch['next_states'].to(device)
            object_mask = batch['object_mask'].to(device)
            masses = batch.get('masses', torch.ones(object_states.shape[0], object_states.shape[-2], 1)).to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                # Forward pass
                outputs = model.forward_physics(object_states, object_mask)
                
                # Compute targets
                if object_states.dim() == 4:
                    current_states = object_states[:, -1]
                else:
                    current_states = object_states
                
                if next_states.dim() == 4:
                    target_states = next_states[:, -1]
                else:
                    target_states = next_states
                
                targets = {
                    'current_states': current_states,
                    'target_states': target_states,
                    'delta_states': target_states - current_states,
                    'object_mask': object_mask if object_mask.dim() == 2 else object_mask[:, -1],
                    'masses': masses
                }
                
                # Compute loss with consistency term
                total_loss, losses = loss_fn(outputs, targets, mode='physics')
            
            # Backward pass
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            # Track losses
            train_losses['total'] += total_loss.item()
            train_losses['prediction'] += losses.get('prediction', 0.0) if isinstance(losses.get('prediction', 0.0), float) else losses.get('prediction', torch.tensor(0.0)).item()
            train_losses['consistency'] += losses.get('velocity_consistency_loss', 0.0) if isinstance(losses.get('velocity_consistency_loss', 0.0), float) else losses.get('velocity_consistency_loss', torch.tensor(0.0)).item()
            num_batches += 1
            
            progress.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'consist': f"{losses.get('velocity_consistency_loss', 0.0):.4f}" if 'velocity_consistency_loss' in losses else "N/A"
            })
        
        # Average training losses
        for key in train_losses:
            train_losses[key] /= max(num_batches, 1)
        
        # Validation
        model.eval()
        val_losses = {'total': 0.0, 'prediction': 0.0, 'consistency': 0.0}
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]"):
                object_states = batch['object_states'].to(device)
                next_states = batch['next_states'].to(device)
                object_mask = batch['object_mask'].to(device)
                masses = batch.get('masses', torch.ones(object_states.shape[0], object_states.shape[-2], 1)).to(device)
                
                with autocast():
                    outputs = model.forward_physics(object_states, object_mask)
                    
                    if object_states.dim() == 4:
                        current_states = object_states[:, -1]
                    else:
                        current_states = object_states
                    
                    if next_states.dim() == 4:
                        target_states = next_states[:, -1]
                    else:
                        target_states = next_states
                    
                    targets = {
                        'current_states': current_states,
                        'target_states': target_states,
                        'delta_states': target_states - current_states,
                        'object_mask': object_mask if object_mask.dim() == 2 else object_mask[:, -1],
                        'masses': masses
                    }
                    
                    total_loss, losses = loss_fn(outputs, targets, mode='physics')
                
                val_losses['total'] += total_loss.item()
                val_losses['prediction'] += losses.get('prediction', 0.0) if isinstance(losses.get('prediction', 0.0), float) else losses.get('prediction', torch.tensor(0.0)).item()
                val_losses['consistency'] += losses.get('velocity_consistency_loss', 0.0) if isinstance(losses.get('velocity_consistency_loss', 0.0), float) else losses.get('velocity_consistency_loss', torch.tensor(0.0)).item()
                num_val_batches += 1
        
        for key in val_losses:
            val_losses[key] /= max(num_val_batches, 1)
        
        # Log epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train - Total: {train_losses['total']:.4f}, Pred: {train_losses['prediction']:.4f}, Consist: {train_losses['consistency']:.4f}")
        print(f"  Val   - Total: {val_losses['total']:.4f}, Pred: {val_losses['prediction']:.4f}, Consist: {val_losses['consistency']:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train': train_losses.copy(),
            'val': val_losses.copy()
        })
        
        # Early stopping check
        if val_losses['total'] < best_val_loss - min_delta:
            best_val_loss = val_losses['total']
            epochs_without_improvement = 0
            
            # Save best checkpoint
            save_path = Path(checkpoint_dir) / "finetune_consistency_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'gamma': gamma,
                'history': history
            }, save_path)
            print(f"  ✓ New best! Saved to {save_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")
            
            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping triggered after {epoch + 1} epochs")
                break
        
        torch.cuda.empty_cache()
    
    # Save final checkpoint
    final_path = Path(checkpoint_dir) / "finetune_consistency_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,
        'val_loss': val_losses['total'],
        'gamma': gamma,
        'history': history
    }, final_path)
    
    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Consistency weight (γ): {gamma}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"  - finetune_consistency_best.pt (best)")
    print(f"  - finetune_consistency_final.pt (final)")
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune with velocity consistency loss')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to fine-tune')
    parser.add_argument('--gamma', type=float, default=0.1, help='Consistency loss weight (default: 0.1)')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--max_episodes', type=int, default=None, help='Limit dataset size')
    parser.add_argument('--checkpoint_dir', type=str, default='$CHECKPOINT_DIR', help='Checkpoint directory')
    parser.add_argument('--data_dir', type=str, default='D:/physics_hdf5', help='Data directory')
    
    args = parser.parse_args()
    
    run_finetune(
        checkpoint_path=args.checkpoint,
        gamma=args.gamma,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        max_episodes=args.max_episodes,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir
    )
