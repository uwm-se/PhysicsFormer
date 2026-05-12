"""
Progressive Curriculum for Sequence Length AND Schema Difficulty

Automatically increases:
1. Sequence length: 128 (fixed for speed)
2. Schema difficulty: Level 1 → 2 → ... → 13

37 Isaac Sim Physics Schemas (11 Groups, 35D State Vector):
- Group 1: Gravity & Free Fall (4) - multi_drop, varied_mass_drop, varied_height_drop, simultaneous_drop
- Group 2: Collisions (6) - head_on, angled, multi_body, chain, cluster, asymmetric
- Group 3: Stacking (5) - simple, tall, pyramid, unstable, offset
- Group 4: Rolling & Sliding (4) - cube_slide, ramp_roll, ramp_slide, friction_compare
- Group 5: Projectiles (4) - horizontal_throw, angled_throw, lob_throw, multi_projectile
- Group 6: Domino (1) - domino_line
- Group 7: Scattering (4) - explosion, impact, funnel, directed
- Group 8: Physics Variety (5) - obstacle_drop, wedge_deflect, block_stacking, projectile_trajectory, friction_variety
- Group 9: Articulated (1) - hinged_door
- Group 10: Rotation (1) - angular_momentum
- Group 11: Complex Dynamics (2) - billiard_break, bowling_strike
- Group 12: CAUSAL TRAINING - Object dropout with causal graph intervention loss
- Group 13: COUNTERFACTUAL TRAINING - "What if" reasoning with contrastive learning

Convergence criteria:
- Loss plateau (< 5% improvement over N epochs)
- Minimum epochs per phase
- Loss threshold

USAGE:
------
# In training loop (after each epoch):
curriculum = ProgressiveCurriculum(
    initial_seq_length=256, 
    target_seq_length=256,
    initial_schema_level=1,
    target_schema_level=11
)

for epoch in range(total_epochs):
    # ... train epoch ...
    epoch_loss = sum(losses) / len(losses)
    
    # Check if should progress
    result = curriculum.update(epoch_loss)
    
    if result['should_progress']:
        print(result['message'])
        # Reload dataset with new sequence length and/or schema level
        if 'new_seq_length' in result:
            config.max_seq_length = result['new_seq_length']
        if 'new_schema_level' in result:
            config.schema_curriculum_level = result['new_schema_level']
        dataloader = create_dataloader(config, batch_size=result['new_batch_size'])
        # Continue training with new config
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressiveCurriculum:
    """Manages automatic progression of sequence length during training."""
    
    def __init__(
        self,
        initial_seq_length: int = 256,
        target_seq_length: int = 256,
        initial_schema_level: int = 1,
        target_schema_level: int = 13,
        min_epochs_per_phase: int = 10,
        max_epochs_per_phase: int = 30,  # HARD LIMIT: Force advancement after this many epochs
        convergence_patience: int = 5,
        improvement_threshold: float = 0.02,
        min_accuracy_to_progress: float = 0.1  # RELAXED: Allow progression with 10% accuracy
    ):
        """
        Args:
            initial_seq_length: Starting sequence length
            target_seq_length: Final target sequence length
            initial_schema_level: Starting schema difficulty level (1-11)
            target_schema_level: Final schema difficulty level (1-11)
            min_epochs_per_phase: Minimum epochs before considering progression
            max_epochs_per_phase: Maximum epochs per phase - force advancement after this
            convergence_patience: Epochs to wait for improvement before progressing
            improvement_threshold: Minimum relative improvement to continue (5% = 0.05)
            min_accuracy_to_progress: Minimum validation accuracy (0-1) required to advance
        """
        self.initial_seq_length = initial_seq_length
        self.target_seq_length = target_seq_length
        self.initial_schema_level = initial_schema_level
        self.target_schema_level = target_schema_level
        self.min_epochs_per_phase = min_epochs_per_phase
        self.max_epochs_per_phase = max_epochs_per_phase
        self.convergence_patience = convergence_patience
        self.improvement_threshold = improvement_threshold
        self.min_accuracy_to_progress = min_accuracy_to_progress
        self.current_accuracy = 0.0  # Track validation accuracy
        
        # Reduced sequence length - 128 frames (2.1 seconds at 60fps)
        # Good balance of speed and capturing physics events
        self.sequence_progression = [128]  # Balanced for speed + quality
        
        # Schema progression (1 → 2 → 3 → ... → 11) for 37 Isaac Sim schemas
        # Groups 1-8: Core physics (33 schemas)
        # Groups 9-11: Advanced mechanics (4 schemas)
        self.schema_progression = list(range(initial_schema_level, target_schema_level + 1))
        
        # Current state
        self.current_phase = 0
        self.current_seq_length = self.sequence_progression[0]
        self.current_schema_level = self.schema_progression[0]
        self.epochs_in_phase = 0
        self.best_loss_in_phase = float('inf')
        self.epochs_without_improvement = 0
        self.loss_history: List[float] = []
        
        logger.info(f"Progressive Curriculum initialized:")
        logger.info(f"  Sequence progression: {self.sequence_progression}")
        logger.info(f"  Schema progression: {self.schema_progression}")
        logger.info(f"  Min epochs per phase: {min_epochs_per_phase}")
        logger.info(f"  Max epochs per phase: {max_epochs_per_phase} (HARD LIMIT)")
        logger.info(f"  Convergence patience: {convergence_patience}")
        logger.info(f"  Improvement threshold: {improvement_threshold * 100}%")
    
    def update(self, epoch_loss: float, accuracy: float = None) -> Dict[str, any]:
        """
        Update curriculum with epoch loss and check if should progress.
        
        Args:
            epoch_loss: Average loss for the completed epoch
            accuracy: Optional validation accuracy (0-1) for progression check
            
        Returns:
            Dict with:
                - should_progress: bool
                - new_seq_length: int (if progressing)
                - new_batch_size: int (if progressing)
                - message: str
        """
        if accuracy is not None:
            self.current_accuracy = accuracy
        self.epochs_in_phase += 1
        self.loss_history.append(epoch_loss)
        
        # Check for improvement
        if epoch_loss < self.best_loss_in_phase:
            improvement = (self.best_loss_in_phase - epoch_loss) / self.best_loss_in_phase
            self.best_loss_in_phase = epoch_loss
            
            if improvement > self.improvement_threshold:
                # Significant improvement - reset patience
                self.epochs_without_improvement = 0
            else:
                # Minor improvement - increment patience
                self.epochs_without_improvement += 1
        else:
            # No improvement
            self.epochs_without_improvement += 1
        
        # Check if should progress to next phase
        should_progress = self._should_progress()
        
        if should_progress:
            return self._progress_to_next_phase()
        else:
            return {
                'should_progress': False,
                'message': self._get_status_message()
            }
    
    def is_complete(self) -> bool:
        """Check if curriculum has reached final target (all schemas, longest sequences)."""
        return (self.current_schema_level >= self.target_schema_level and 
                self.current_phase >= len(self.sequence_progression) - 1)
    
    def _should_progress(self) -> bool:
        """Check if conditions are met to progress to next phase."""
        # Check if we can progress either sequence length OR schema level
        can_progress_seq = self.current_phase < len(self.sequence_progression) - 1
        can_progress_schema = self.current_schema_level < self.target_schema_level
        
        # Already at target for both
        if not can_progress_seq and not can_progress_schema:
            return False
        
        # HARD LIMIT: Force advancement after max_epochs_per_phase regardless of other conditions
        if self.epochs_in_phase >= self.max_epochs_per_phase:
            logger.info(f"[CURRICULUM] HARD LIMIT reached: {self.epochs_in_phase} >= {self.max_epochs_per_phase} epochs - forcing advancement")
            return True
        
        # Not enough epochs yet
        if self.epochs_in_phase < self.min_epochs_per_phase:
            logger.debug(f"Not progressing: epochs_in_phase={self.epochs_in_phase} < min={self.min_epochs_per_phase}")
            return False
        
        # Loss hasn't converged yet
        if self.epochs_without_improvement < self.convergence_patience:
            logger.debug(f"Not progressing: epochs_without_improvement={self.epochs_without_improvement} < patience={self.convergence_patience}")
            return False
        
        # Check minimum accuracy requirement
        if self.current_accuracy < self.min_accuracy_to_progress:
            logger.info(f"[CURRICULUM] Not progressing: accuracy={self.current_accuracy:.1%} < min={self.min_accuracy_to_progress:.1%}")
            return False
        
        return True
    
    def _progress_to_next_phase(self) -> Dict[str, any]:
        """Progress to next sequence length and/or schema level."""
        old_seq_length = self.current_seq_length
        old_schema_level = self.current_schema_level
        old_phase = self.current_phase
        old_best_loss = self.best_loss_in_phase
        
        result = {
            'should_progress': True,
            'old_seq_length': old_seq_length,
            'old_schema_level': old_schema_level,
            'phase': self.current_phase + 1
        }
        
        # Prioritize schema progression over sequence length
        # (Learn all schemas at current seq length before increasing seq length)
        if self.current_schema_level < self.target_schema_level:
            # Progress schema level
            self.current_schema_level += 1
            result['new_schema_level'] = self.current_schema_level
            result['schema_progressed'] = True
            progression_type = "SCHEMA LEVEL"
        elif self.current_phase < len(self.sequence_progression) - 1:
            # Progress sequence length
            self.current_phase += 1
            self.current_seq_length = self.sequence_progression[self.current_phase]
            result['new_seq_length'] = self.current_seq_length
            result['seq_progressed'] = True
            progression_type = "SEQUENCE LENGTH"
            
            # Calculate new batch size (inversely proportional to sequence length)
            ratio = self.current_seq_length / old_seq_length
            new_batch_size = max(1, int(48 / ratio))
            result['new_batch_size'] = new_batch_size
        else:
            # Should not reach here due to _should_progress check
            return {'should_progress': False, 'message': 'Already at target'}
        
        # Reset phase tracking
        self.epochs_in_phase = 0
        self.best_loss_in_phase = float('inf')
        self.epochs_without_improvement = 0
        
        # Mark ablation study checkpoints at key milestones
        # These are important for comparing model performance at different stages
        ABLATION_MILESTONES = {
            1: "baseline_gravity",      # Just gravity/drops
            4: "ablation_collisions",   # + collisions + stacking
            8: "ablation_full_physics", # All basic physics
            11: "ablation_all_schemas", # All schemas before causal
            12: "ablation_causal",      # After causal training
            13: "ablation_counterfactual"  # After counterfactual training
        }
        
        if self.current_schema_level in ABLATION_MILESTONES:
            result['save_ablation'] = True
            result['ablation_name'] = ABLATION_MILESTONES[self.current_schema_level]
        
        # Build message
        message_lines = [
            f"\n{'='*70}",
            f"PROGRESSIVE CURRICULUM: ADVANCING {progression_type}",
            f"{'='*70}",
            f"Phase {old_phase + 1} -> {self.current_phase + 1}"
        ]
        
        if 'new_schema_level' in result:
            message_lines.append(f"Schema Level: {old_schema_level} -> {self.current_schema_level}")
            if result.get('save_ablation'):
                message_lines.append(f"[ABLATION CHECKPOINT]: {result['ablation_name']}")
        if 'new_seq_length' in result:
            message_lines.append(f"Sequence Length: {old_seq_length} -> {self.current_seq_length}")
            message_lines.append(f"Recommended Batch Size: {result.get('new_batch_size', 48)}")
        
        message_lines.extend([
            f"Previous phase best loss: {old_best_loss:.4f}",
            f"{'='*70}\n"
        ])
        
        message = "\n".join(message_lines)
        result['message'] = message
        
        logger.info(message)
        print(message)
        
        return result
    
    def _get_status_message(self) -> str:
        """Get current status message."""
        remaining_epochs = max(0, self.min_epochs_per_phase - self.epochs_in_phase)
        epochs_until_hard_limit = max(0, self.max_epochs_per_phase - self.epochs_in_phase)
        
        # Check various blocking conditions
        if remaining_epochs > 0:
            reason = f"Need {remaining_epochs} more epoch(s) in phase (hard limit in {epochs_until_hard_limit})"
        elif self.epochs_without_improvement < self.convergence_patience:
            patience_remaining = self.convergence_patience - self.epochs_without_improvement
            reason = f"Waiting for convergence ({patience_remaining} patience, hard limit in {epochs_until_hard_limit})"
        elif self.current_accuracy < self.min_accuracy_to_progress:
            reason = f"Accuracy too low ({self.current_accuracy:.1%} < {self.min_accuracy_to_progress:.1%}, hard limit in {epochs_until_hard_limit})"
        else:
            reason = "Converged, ready to progress"
        
        return (
            f"Phase {self.current_phase + 1}/{len(self.sequence_progression)}: "
            f"schema_lvl={self.current_schema_level}/{self.target_schema_level}, "
            f"seq_len={self.current_seq_length}, "
            f"epoch {self.epochs_in_phase}/{self.max_epochs_per_phase}, "
            f"best_loss={self.best_loss_in_phase:.4f}, "
            f"{reason}"
        )
    
    def get_current_config(self) -> Dict[str, int]:
        """Get current sequence length and recommended batch size."""
        ratio = self.current_seq_length / self.initial_seq_length
        batch_size = max(1, int(48 / ratio))  # Base of 48 (matches curriculum_utils)
        
        return {
            'seq_length': self.current_seq_length,
            'batch_size': batch_size,
            'phase': self.current_phase + 1,
            'total_phases': len(self.sequence_progression)
        }
    
    def is_complete(self) -> bool:
        """Check if reached target sequence length."""
        return self.current_seq_length >= self.target_seq_length
    
    def force_progress(self) -> Dict[str, any]:
        """
        Manually force progression to next sequence length.
        Useful for manual intervention during training.
        
        Returns:
            Same dict as update() when progressing
        """
        if self.current_phase >= len(self.sequence_progression) - 1:
            return {
                'should_progress': False,
                'message': 'Already at maximum sequence length'
            }
        
        return self._progress_to_next_phase()
    
    def force_advance(self) -> bool:
        """
        Force advancement to next schema level (used by early stopping).
        Bypasses normal progression checks.
        
        Returns:
            True if advanced, False if already at max level
        """
        if self.current_schema_level >= self.target_schema_level:
            logger.info(f"[CURRICULUM] Cannot advance: already at max schema level {self.target_schema_level}")
            return False
        
        old_level = self.current_schema_level
        self.current_schema_level += 1
        
        # Reset phase tracking for new level
        self.epochs_in_phase = 0
        self.best_loss_in_phase = float('inf')
        self.epochs_without_improvement = 0
        self.loss_history = []
        
        logger.info(f"[CURRICULUM] Forced advance: schema level {old_level} -> {self.current_schema_level}")
        return True
    
    def set_sequence_length(self, seq_length: int) -> Dict[str, any]:
        """
        Manually set to a specific sequence length.
        
        Args:
            seq_length: Target sequence length (must be in progression)
            
        Returns:
            Dict with progression info
        """
        if seq_length not in self.sequence_progression:
            return {
                'should_progress': False,
                'message': f'Invalid sequence length. Must be one of: {self.sequence_progression}'
            }
        
        target_phase = self.sequence_progression.index(seq_length)
        if target_phase == self.current_phase:
            return {
                'should_progress': False,
                'message': f'Already at sequence length {seq_length}'
            }
        
        old_seq_length = self.current_seq_length
        old_phase = self.current_phase
        
        # Jump to target phase
        self.current_phase = target_phase
        self.current_seq_length = seq_length
        
        # Calculate new batch size
        ratio = self.current_seq_length / self.initial_seq_length
        new_batch_size = max(1, int(48 / ratio))  # Base of 48 (matches curriculum_utils)
        
        # Reset phase tracking
        self.epochs_in_phase = 0
        self.best_loss_in_phase = float('inf')
        self.epochs_without_improvement = 0
        
        message = (
            f"\n{'='*70}\n"
            f"MANUAL SEQUENCE LENGTH CHANGE\n"
            f"{'='*70}\n"
            f"Phase {old_phase + 1} → {self.current_phase + 1}\n"
            f"Sequence Length: {old_seq_length} → {self.current_seq_length}\n"
            f"Recommended Batch Size: {new_batch_size}\n"
            f"{'='*70}\n"
        )
        
        logger.info(message)
        print(message)
        
        return {
            'should_progress': True,
            'new_seq_length': self.current_seq_length,
            'new_batch_size': new_batch_size,
            'old_seq_length': old_seq_length,
            'phase': self.current_phase + 1,
            'message': message
        }
