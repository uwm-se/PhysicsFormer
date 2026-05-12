"""
Run Object Permanence Training for PhysicsFormer

Fine-tunes PhysicsFormer with object permanence training to:
1. Track objects that leave the field of view
2. Learn when to forget objects (not fixed decay)
3. Predict if objects will return after occlusion

This is critical for real-world applications where occlusion is common.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from physics_former.training.models.physics_former_full import FullPhysicsFormer
from physics_former.training.object_permanence import (
    ObjectPermanenceModule,
    ObjectPermanenceConfig,
    create_object_permanence_module
)
from physics_former.training.datasets.hdf5_physics_dataset import HDF5PhysicsDataset


def load_checkpoint(checkpoint_dir: str = "D:\\physics-former-data\\checkpoints"):
    """Load existing PhysicsFormer checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for ckpt_name in ['stage1_best.pt', 'stage2_best.pt', 'physics_former_best.pt']:
        ckpt_path = checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            break
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    
    print(f"Loading from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    num_schema_classes = checkpoint['model_state_dict']['schema_classifier.3.bias'].shape[0]
    
    model = FullPhysicsFormer(
        state_dim=35,
        hidden_dim=512,
        num_layers=6,
        num_heads=16,
        ff_dim=2048,
        max_objects=20,
        dropout=0.1,
        num_schema_classes=num_schema_classes
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, device, checkpoint


def run_permanence_training(
    max_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    patience: int = 5,
    min_delta: float = 0.001,
    max_grad_norm: float = 10.0,
    max_episodes: int = 5000,
    forget_weight: float = 0.5,
    checkpoint_dir: str = "D:\\physics-former-data\\checkpoints",
    data_dir: str = "D:\\physics_hdf5"
):
    """
    Train object permanence module with early stopping.
    
    Args:
        max_epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        patience: Early stopping patience
        min_delta: Minimum improvement threshold
        forget_weight: Weight for forget gate loss
        checkpoint_dir: Where to load/save checkpoints
        data_dir: Physics data directory
    """
    print("="*70)
    print("OBJECT PERMANENCE TRAINING")
    print("="*70)
    
    print("\n[1/5] Loading pre-trained PhysicsFormer...")
    physics_model, device, _ = load_checkpoint(checkpoint_dir)
    physics_model.eval()
    for param in physics_model.parameters():
        param.requires_grad = False
    print(f"  Device: {device}")
    print("  PhysicsFormer frozen (only training permanence module)")
    
    print("\n[2/5] Creating object permanence module...")
    config = ObjectPermanenceConfig(
        max_occlusion_time=30,
        confidence_decay=0.95,
        use_learned_forget=True,
        forget_hidden_dim=64
    )
    
    permanence = ObjectPermanenceModule(physics_model, config).to(device)
    
    trainable_params = sum(p.numel() for p in permanence.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n[3/5] Loading physics dataset...")
    dataset = HDF5PhysicsDataset(
        data_dir=data_dir,
        hdf5_dir=data_dir,
        max_objects=20,
        max_seq_length=32
    )
    
    if max_episodes and max_episodes < len(dataset):
        indices = torch.randperm(len(dataset))[:max_episodes].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"  Subsampled to {max_episodes} episodes for faster training")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batches per epoch: {len(dataloader)}")
    
    print("\n[4/5] Setting up optimizer...")
    trainable = [p for p in permanence.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    
    print("\n[5/5] Training with early stopping...")
    print(f"  Patience: {patience}, Min Delta: {min_delta}")
    best_total_loss = float('inf')
    epochs_without_improvement = 0
    history = []
    
    for epoch in range(max_epochs):
        epoch_losses = {
            'total': 0.0,
            'permanence': 0.0,
            'forget': 0.0,
            'return_pred': 0.0
        }
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress:
            states = batch['object_states'].to(device)
            mask = batch['object_mask'].to(device)
            
            if states.dim() == 3:
                states = states.unsqueeze(1).expand(-1, 16, -1, -1).clone()
            
            if mask.dim() == 3:
                mask = mask[:, 0, :]
            
            num_objects = min(states.shape[2], 20)
            states = states[:, :, :num_objects, :]
            mask = mask[:, :num_objects]
            
            optimizer.zero_grad()
            
            losses = permanence.compute_combined_loss(states, mask, forget_weight)
            losses['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            
            # Store losses before cleanup
            loss_values = {k: v.item() for k, v in losses.items()}
            
            # Memory cleanup
            del states, mask, losses
            torch.cuda.empty_cache()
            
            epoch_losses['total'] += loss_values['total']
            epoch_losses['permanence'] += loss_values['permanence_total']
            epoch_losses['forget'] += loss_values['forget']
            epoch_losses['return_pred'] += loss_values['return_prediction']
            num_batches += 1
            
            progress.set_postfix({
                'loss': f"{loss_values['total']:.4f}",
                'forget': f"{loss_values['forget']:.4f}"
            })
        
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        history.append(epoch_losses)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_losses['total']:.4f}")
        print(f"  Permanence Loss: {epoch_losses['permanence']:.4f}")
        print(f"  Forget Gate Loss: {epoch_losses['forget']:.4f}")
        print(f"  Return Prediction Loss: {epoch_losses['return_pred']:.4f}")
        
        if epoch_losses['total'] < best_total_loss - min_delta:
            best_total_loss = epoch_losses['total']
            epochs_without_improvement = 0
            
            save_path = Path(checkpoint_dir) / "permanence_best.pt"
            torch.save({
                'permanence_state_dict': permanence.state_dict(),
                'epoch': epoch,
                'total_loss': best_total_loss,
                'config': config.__dict__
            }, save_path)
            print(f"  ✓ New best! Saved checkpoint: {save_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")
            
            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Memory cleanup at end of epoch
        torch.cuda.empty_cache()
    
    final_path = Path(checkpoint_dir) / "permanence_final.pt"
    torch.save({
        'permanence_state_dict': permanence.state_dict(),
        'epochs': epoch + 1,
        'history': history,
        'config': config.__dict__
    }, final_path)
    
    print("\n" + "="*70)
    print("OBJECT PERMANENCE TRAINING COMPLETE")
    print("="*70)
    print(f"Best total loss: {best_total_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    return permanence, history


def test_permanence(checkpoint_dir: str = "D:\\physics-former-data\\checkpoints"):
    """Test trained object permanence module."""
    print("\n" + "="*70)
    print("TESTING OBJECT PERMANENCE")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    permanence_path = Path(checkpoint_dir) / "permanence_best.pt"
    if not permanence_path.exists():
        print("Permanence checkpoint not found. Run training first.")
        return
    
    print("\n[1/3] Loading models...")
    physics_model, device, _ = load_checkpoint(checkpoint_dir)
    physics_model.eval()
    
    perm_ckpt = torch.load(permanence_path, map_location=device, weights_only=False)
    config = ObjectPermanenceConfig(**perm_ckpt['config'])
    
    permanence = ObjectPermanenceModule(physics_model, config).to(device)
    permanence.load_state_dict(perm_ckpt['permanence_state_dict'])
    permanence.eval()
    
    print("\n[2/3] Creating test scenario...")
    num_objects = 5
    state_dim = 35
    
    visible_states = torch.randn(num_objects, state_dim, device=device)
    visible_mask = torch.ones(num_objects, device=device)
    
    for i in range(num_objects):
        visible_states[i, 0] = i * 2.0
        visible_states[i, 1] = 0.0
        visible_states[i, 2] = 0.5
    
    print("\n[3/3] Running permanence tracking...")
    permanence.reset()
    
    print("\nFrame 1: All objects visible")
    all_states, all_mask, meta = permanence(visible_states, visible_mask)
    print(f"  Tracked: {len(meta['object_ids'])} objects")
    print(f"  Visible: {meta['visible_count']}, Occluded: {meta['occluded_count']}")
    
    print("\nFrame 2: Object 2 disappears")
    visible_mask_2 = visible_mask.clone()
    visible_mask_2[2] = 0
    all_states, all_mask, meta = permanence(visible_states, visible_mask_2)
    print(f"  Tracked: {len(meta['object_ids'])} objects")
    print(f"  Visible: {meta['visible_count']}, Occluded: {meta['occluded_count']}")
    print(f"  Confidences: {[f'{c:.2f}' for c in meta['confidences']]}")
    
    print("\nFrame 3-5: Object 2 still hidden")
    for frame in range(3, 6):
        all_states, all_mask, meta = permanence(visible_states, visible_mask_2)
        print(f"  Frame {frame}: Occluded confidence = {meta['confidences'][2]:.3f}")
    
    print("\nFrame 6: Object 2 returns")
    all_states, all_mask, meta = permanence(visible_states, visible_mask)
    print(f"  Tracked: {len(meta['object_ids'])} objects")
    print(f"  Visible: {meta['visible_count']}, Occluded: {meta['occluded_count']}")
    print(f"  Object 2 re-identified: confidence = {meta['confidences'][2]:.2f}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object permanence training')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement threshold')
    parser.add_argument('--max_episodes', type=int, default=5000, help='Max episodes for faster training')
    parser.add_argument('--forget_weight', type=float, default=0.5, help='Forget loss weight')
    parser.add_argument('--test_only', action='store_true', help='Only test existing checkpoint')
    
    args = parser.parse_args()
    
    if args.test_only:
        test_permanence()
    else:
        permanence, history = run_permanence_training(
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            min_delta=args.min_delta,
            max_episodes=args.max_episodes,
            forget_weight=args.forget_weight
        )
        test_permanence()
