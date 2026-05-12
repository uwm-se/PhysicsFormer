"""
Run Causal Training for PhysicsFormer

Fine-tunes an existing PhysicsFormer checkpoint with causal intervention
training to improve counterfactual reasoning capability.

This addresses the CLEVRER-Humans gap (98.5% → 54%) by teaching the model
to distinguish correlation from causation.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from physics_former.training.models.physics_former_full import FullPhysicsFormer
from physics_former.training.causal_training import (
    CausalTrainingConfig,
    CausalPhysicsTrainer,
    add_causal_training_to_pipeline
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


def run_causal_finetuning(
    max_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    patience: int = 5,
    min_delta: float = 0.001,
    max_grad_norm: float = 10.0,
    max_episodes: int = 5000,
    checkpoint_dir: str = "D:\\physics-former-data\\checkpoints",
    data_dir: str = "D:\\physics_hdf5"
):
    """
    Fine-tune PhysicsFormer with causal training.
    
    Args:
        epochs: Number of causal training epochs
        batch_size: Batch size
        learning_rate: Learning rate (lower for fine-tuning)
        checkpoint_dir: Where to load/save checkpoints
        data_dir: Physics data directory
    """
    print("="*70)
    print("CAUSAL TRAINING FOR PHYSICSFORMER")
    print("="*70)
    
    print("\n[1/5] Loading pre-trained model...")
    model, device, checkpoint = load_checkpoint(checkpoint_dir)
    model.train()
    print(f"  Device: {device}")
    
    print("\n[2/5] Loading physics dataset...")
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
    
    print("\n[3/5] Setting up causal training...")
    config = CausalTrainingConfig(
        dropout_prob=0.3,
        intervention_weight=1.0,
        contrastive_weight=0.5,
        causal_margin=0.1
    )
    
    trainer = CausalPhysicsTrainer(model, config)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    print("\n[4/5] Running causal training with early stopping...")
    print(f"  Patience: {patience}, Min Delta: {min_delta}")
    best_intervention_loss = float('inf')
    epochs_without_improvement = 0
    history = []
    
    for epoch in range(max_epochs):
        epoch_losses = {
            'total': 0.0,
            'prediction': 0.0,
            'intervention': 0.0,
            'causal_sensitivity': 0.0
        }
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress:
            states = batch['object_states'].to(device)
            mask = batch['object_mask'].to(device)
            
            if states.dim() == 3:
                states = states.unsqueeze(1).expand(-1, 10, -1, -1)
            
            if mask.dim() == 3:
                mask = mask[:, 0, :]
            
            num_objects = min(states.shape[2], 20)
            states = states[:, :, :num_objects, :]
            mask = mask[:, :num_objects]
            
            optimizer.zero_grad()
            
            losses = trainer.train_step(states, mask)
            losses['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            # Store losses before cleanup
            loss_values = {k: v.item() for k, v in losses.items()}
            
            # Memory cleanup
            del states, mask, losses
            torch.cuda.empty_cache()
            
            for key in epoch_losses:
                epoch_losses[key] += loss_values[key]
            num_batches += 1
            
            progress.set_postfix({
                'loss': f"{loss_values['total']:.4f}",
                'interv': f"{loss_values['intervention']:.4f}"
            })
        
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        history.append(epoch_losses)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_losses['total']:.4f}")
        print(f"  Prediction Loss: {epoch_losses['prediction']:.4f}")
        print(f"  Intervention Loss: {epoch_losses['intervention']:.4f}")
        print(f"  Causal Sensitivity Loss: {epoch_losses['causal_sensitivity']:.4f}")
        
        if epoch_losses['intervention'] < best_intervention_loss - min_delta:
            best_intervention_loss = epoch_losses['intervention']
            epochs_without_improvement = 0
            
            save_path = Path(checkpoint_dir) / "causal_finetuned_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'intervention_loss': best_intervention_loss,
                'config': config.__dict__
            }, save_path)
            print(f"  New best! Saved checkpoint: {save_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")
            
            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Memory cleanup at end of epoch
        torch.cuda.empty_cache()
    
    print("\n[5/5] Saving final model...")
    final_path = Path(checkpoint_dir) / "causal_finetuned_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs': epoch + 1,
        'history': history,
        'config': config.__dict__
    }, final_path)
    
    print("\n" + "="*70)
    print("CAUSAL TRAINING COMPLETE")
    print("="*70)
    print(f"Best intervention loss: {best_intervention_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    return model, history


def test_causal_improvement(checkpoint_dir: str = "D:\\physics-former-data\\checkpoints"):
    """Test if causal training improved counterfactual reasoning."""
    print("\n" + "="*70)
    print("TESTING CAUSAL IMPROVEMENT")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    original_path = Path(checkpoint_dir) / "stage1_best.pt"
    causal_path = Path(checkpoint_dir) / "causal_finetuned_best.pt"
    
    if not causal_path.exists():
        print("Causal checkpoint not found. Run training first.")
        return
    
    print("\n[1/3] Loading original model...")
    orig_ckpt = torch.load(original_path, map_location=device, weights_only=False)
    num_schema = orig_ckpt['model_state_dict']['schema_classifier.3.bias'].shape[0]
    
    original_model = FullPhysicsFormer(
        state_dim=35, hidden_dim=512, num_layers=6, num_heads=16,
        ff_dim=2048, max_objects=20, dropout=0.1, num_schema_classes=num_schema
    ).to(device)
    original_model.load_state_dict(orig_ckpt['model_state_dict'])
    original_model.eval()
    
    print("\n[2/3] Loading causal-trained model...")
    causal_ckpt = torch.load(causal_path, map_location=device, weights_only=False)
    
    causal_model = FullPhysicsFormer(
        state_dim=35, hidden_dim=512, num_layers=6, num_heads=16,
        ff_dim=2048, max_objects=20, dropout=0.1, num_schema_classes=num_schema
    ).to(device)
    causal_model.load_state_dict(causal_ckpt['model_state_dict'])
    causal_model.eval()
    
    print("\n[3/3] Running counterfactual test...")
    
    batch_size = 1
    seq_len = 10
    num_objects = 3
    state_dim = 35
    
    states = torch.zeros(batch_size, seq_len, num_objects, state_dim, device=device)
    for t in range(seq_len):
        states[0, t, 0, 0] = -5.0 + t * 1.0
        states[0, t, 0, 10] = 1.0
        states[0, t, 1, 0] = 5.0 - t * 0.5
        states[0, t, 1, 10] = 2.0
        states[0, t, 2, 1] = 5.0
        states[0, t, 2, 10] = 1.0
    
    mask = torch.ones(batch_size, num_objects, device=device)
    
    cf_states = states.clone()
    cf_states[:, :, 2, :] = 0
    cf_mask = mask.clone()
    cf_mask[:, 2] = 0
    
    with torch.no_grad():
        orig_pred, _, _ = original_model.forward_physics(states, mask)
        orig_cf_pred, _, _ = original_model.forward_physics(cf_states, cf_mask)
        
        causal_pred, _, _ = causal_model.forward_physics(states, mask)
        causal_cf_pred, _, _ = causal_model.forward_physics(cf_states, cf_mask)
    
    orig_diff_obj0 = torch.norm(orig_pred[0, :, 0] - orig_cf_pred[0, :, 0]).item()
    orig_diff_obj1 = torch.norm(orig_pred[0, :, 1] - orig_cf_pred[0, :, 1]).item()
    
    causal_diff_obj0 = torch.norm(causal_pred[0, :, 0] - causal_cf_pred[0, :, 0]).item()
    causal_diff_obj1 = torch.norm(causal_pred[0, :, 1] - causal_cf_pred[0, :, 1]).item()
    
    print("\n" + "="*60)
    print("COUNTERFACTUAL TEST: Remove bystander (Object 2)")
    print("Expected: Objects 0,1 should NOT change (no causal link)")
    print("="*60)
    
    print(f"\nOriginal Model (correlation-based):")
    print(f"  Object 0 trajectory change: {orig_diff_obj0:.4f}")
    print(f"  Object 1 trajectory change: {orig_diff_obj1:.4f}")
    
    print(f"\nCausal-Trained Model:")
    print(f"  Object 0 trajectory change: {causal_diff_obj0:.4f}")
    print(f"  Object 1 trajectory change: {causal_diff_obj1:.4f}")
    
    orig_spurious = orig_diff_obj0 + orig_diff_obj1
    causal_spurious = causal_diff_obj0 + causal_diff_obj1
    improvement = (orig_spurious - causal_spurious) / orig_spurious * 100
    
    print(f"\nSpurious correlation reduction: {improvement:.1f}%")
    
    if causal_spurious < orig_spurious:
        print("✓ Causal training IMPROVED counterfactual reasoning!")
    else:
        print("✗ Causal training did not improve (may need more epochs)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Causal training for PhysicsFormer')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement threshold')
    parser.add_argument('--max_episodes', type=int, default=5000, help='Max episodes for faster training')
    parser.add_argument('--test_only', action='store_true', help='Only test existing checkpoint')
    
    args = parser.parse_args()
    
    if args.test_only:
        test_causal_improvement()
    else:
        model, history = run_causal_finetuning(
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            min_delta=args.min_delta,
            max_episodes=args.max_episodes
        )
        test_causal_improvement()
