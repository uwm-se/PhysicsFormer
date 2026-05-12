"""
Checkpoint Management Module

Handles saving and loading model checkpoints.
"""

import torch
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model,
        optimizer,
        epoch: int,
        stage: int,
        loss: float,
        extra_state: Optional[Dict] = None
    ) -> Path:
        """
        Save checkpoint.
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"stage{stage}_epoch{epoch}.pt"
        
        state = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        
        if extra_state:
            state.update(extra_state)
        
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load(
        self,
        checkpoint_path: Path,
        model,
        optimizer: Optional = None
    ) -> Dict:
        """
        Load checkpoint.
        
        Returns:
            Dict with epoch, stage, loss, and any extra state
        """
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'stage': checkpoint.get('stage', 1),
            'loss': checkpoint.get('loss', float('inf'))
        }
    
    def get_latest(self, stage: int) -> Optional[Path]:
        """Get latest checkpoint for a stage."""
        checkpoints = list(self.checkpoint_dir.glob(f"stage{stage}_epoch*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
