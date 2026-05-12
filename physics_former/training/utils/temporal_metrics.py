"""
Temporal Metrics for Analyzing RNN Impact on Counting and Arithmetic.

Provides detailed metrics to determine if RNN temporal context helps or hurts
static tasks like counting and arithmetic.
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class TemporalMetricsTracker:
    """
    Track metrics stratified by temporal properties to analyze RNN impact.
    
    Answers questions like:
    - Does counting accuracy differ for moving vs stationary objects?
    - Does arithmetic work better for fast vs slow objects?
    - Do bouncing objects get counted differently?
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'counting': defaultdict(list),
            'arithmetic': defaultdict(list)
        }
        self.confusion_matrices = {
            'counting': defaultdict(lambda: defaultdict(int)),
            'arithmetic': defaultdict(lambda: defaultdict(int))
        }
    
    def compute_motion_properties(self, object_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute motion properties from object states.
        
        Args:
            object_states: [batch, seq_len, objects, features] or [batch, objects, features]
            
        Returns:
            Dictionary with motion properties per sample
        """
        is_sequence = object_states.dim() == 4
        
        if is_sequence:
            # Compute velocity magnitude over time
            velocities = object_states[:, :, :, 3:6]  # [batch, seq, objects, 3]
            velocity_mag = torch.norm(velocities, dim=-1)  # [batch, seq, objects]
            
            # Average velocity per object
            avg_velocity = velocity_mag.mean(dim=1)  # [batch, objects]
            
            # Velocity variance (indicates bouncing/acceleration)
            velocity_var = velocity_mag.var(dim=1)  # [batch, objects]
            
            # Max velocity (peak speed)
            max_velocity = velocity_mag.max(dim=1)[0]  # [batch, objects]
            
        else:
            # Single timestep
            velocities = object_states[:, :, 3:6]  # [batch, objects, 3]
            velocity_mag = torch.norm(velocities, dim=-1)  # [batch, objects]
            
            avg_velocity = velocity_mag
            velocity_var = torch.zeros_like(velocity_mag)
            max_velocity = velocity_mag
        
        # Aggregate to batch level (average across objects)
        batch_avg_velocity = avg_velocity.mean(dim=1)  # [batch]
        batch_velocity_var = velocity_var.mean(dim=1)  # [batch]
        batch_max_velocity = max_velocity.mean(dim=1)  # [batch]
        
        return {
            'avg_velocity': batch_avg_velocity,
            'velocity_variance': batch_velocity_var,
            'max_velocity': batch_max_velocity,
            'is_moving': batch_avg_velocity > 0.1,  # Threshold for "moving"
            'is_bouncing': batch_velocity_var > 0.5,  # Threshold for "bouncing"
            'is_fast': batch_avg_velocity > 1.0  # Threshold for "fast"
        }
    
    def log_counting_result(
        self,
        object_states: torch.Tensor,
        predicted: torch.Tensor,
        target: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None
    ):
        """
        Log counting results stratified by motion properties.
        
        Args:
            object_states: Input states
            predicted: Predicted counts [batch]
            target: True counts [batch]
            object_mask: Optional mask
        """
        motion_props = self.compute_motion_properties(object_states)
        
        # Convert to numpy for easier processing
        predicted_np = predicted.cpu().numpy()
        target_np = target.cpu().numpy()
        
        batch_size = len(predicted_np)
        
        for i in range(batch_size):
            pred = predicted_np[i]
            true = target_np[i]
            correct = (pred == true)
            
            # Overall metrics
            self.metrics['counting']['overall'].append(correct)
            
            # Stratify by motion properties
            if motion_props['is_moving'][i]:
                self.metrics['counting']['moving'].append(correct)
            else:
                self.metrics['counting']['stationary'].append(correct)
            
            if motion_props['is_bouncing'][i]:
                self.metrics['counting']['bouncing'].append(correct)
            else:
                self.metrics['counting']['not_bouncing'].append(correct)
            
            if motion_props['is_fast'][i]:
                self.metrics['counting']['fast'].append(correct)
            else:
                self.metrics['counting']['slow'].append(correct)
            
            # Velocity bins
            avg_vel = motion_props['avg_velocity'][i].item()
            if avg_vel < 0.1:
                vel_bin = 'stationary'
            elif avg_vel < 0.5:
                vel_bin = 'slow'
            elif avg_vel < 1.0:
                vel_bin = 'medium'
            else:
                vel_bin = 'fast'
            self.metrics['counting'][f'velocity_{vel_bin}'].append(correct)
            
            # Confusion matrix
            self.confusion_matrices['counting'][true][pred] += 1
    
    def log_arithmetic_result(
        self,
        group_a_states: torch.Tensor,
        group_b_states: torch.Tensor,
        predicted: torch.Tensor,
        target: torch.Tensor,
        operation: str
    ):
        """
        Log arithmetic results stratified by motion properties.
        
        Args:
            group_a_states: States of first group
            group_b_states: States of second group
            predicted: Predicted results [batch]
            target: True results [batch]
            operation: Operation type ('add', 'subtract', etc.)
        """
        motion_a = self.compute_motion_properties(group_a_states)
        motion_b = self.compute_motion_properties(group_b_states)
        
        # Convert to numpy
        predicted_np = predicted.cpu().numpy()
        target_np = target.cpu().numpy()
        
        batch_size = len(predicted_np)
        
        for i in range(batch_size):
            pred = predicted_np[i]
            true = target_np[i]
            correct = (abs(pred - true) < 0.5)  # Allow small rounding error
            
            # Overall metrics
            self.metrics['arithmetic'][f'{operation}_overall'].append(correct)
            
            # Stratify by relative motion
            a_moving = motion_a['is_moving'][i]
            b_moving = motion_b['is_moving'][i]
            
            if a_moving and b_moving:
                category = 'both_moving'
            elif a_moving or b_moving:
                category = 'one_moving'
            else:
                category = 'both_stationary'
            
            self.metrics['arithmetic'][f'{operation}_{category}'].append(correct)
            
            # Stratify by velocity difference
            vel_diff = abs(motion_a['avg_velocity'][i] - motion_b['avg_velocity'][i]).item()
            if vel_diff < 0.1:
                vel_category = 'similar_velocity'
            elif vel_diff < 0.5:
                vel_category = 'different_velocity'
            else:
                vel_category = 'very_different_velocity'
            
            self.metrics['arithmetic'][f'{operation}_{vel_category}'].append(correct)
            
            # Confusion matrix
            self.confusion_matrices['arithmetic'][f'{operation}_{int(true)}'][int(pred)] += 1
    
    def get_summary(self, task: str = 'counting') -> Dict[str, float]:
        """
        Get summary statistics for a task.
        
        Args:
            task: 'counting' or 'arithmetic'
            
        Returns:
            Dictionary with accuracy for each category
        """
        summary = {}
        
        for category, results in self.metrics[task].items():
            if len(results) > 0:
                accuracy = np.mean(results)
                count = len(results)
                summary[category] = {
                    'accuracy': accuracy,
                    'count': count,
                    'percentage': accuracy * 100
                }
        
        return summary
    
    def print_report(self, task: str = 'counting'):
        """Print detailed report for a task."""
        print(f"\n{'='*70}")
        print(f"TEMPORAL METRICS REPORT: {task.upper()}")
        print(f"{'='*70}")
        
        summary = self.get_summary(task)
        
        if not summary:
            print("No data collected yet.")
            return
        
        # Overall accuracy
        if 'overall' in summary:
            overall = summary['overall']
            print(f"\nOverall Accuracy: {overall['percentage']:.2f}% ({overall['count']} samples)")
        
        # Motion-based breakdown
        print(f"\n{'-'*70}")
        print("MOTION-BASED BREAKDOWN:")
        print(f"{'-'*70}")
        
        motion_categories = [
            ('moving', 'stationary'),
            ('bouncing', 'not_bouncing'),
            ('fast', 'slow')
        ]
        
        for cat_a, cat_b in motion_categories:
            if cat_a in summary and cat_b in summary:
                acc_a = summary[cat_a]['percentage']
                acc_b = summary[cat_b]['percentage']
                count_a = summary[cat_a]['count']
                count_b = summary[cat_b]['count']
                
                diff = abs(acc_a - acc_b)
                
                print(f"\n{cat_a.upper()} vs {cat_b.upper()}:")
                print(f"  {cat_a:20s}: {acc_a:6.2f}% ({count_a:4d} samples)")
                print(f"  {cat_b:20s}: {acc_b:6.2f}% ({count_b:4d} samples)")
                print(f"  Difference:          {diff:6.2f}%", end="")
                
                if diff > 5.0:
                    print(" [WARN] SIGNIFICANT DIFFERENCE!")
                elif diff > 2.0:
                    print(" [NOTE] Notable difference")
                else:
                    print(" [OK] Similar performance")
        
        # Velocity bins
        print(f"\n{'-'*70}")
        print("VELOCITY BINS:")
        print(f"{'-'*70}")
        
        vel_bins = ['stationary', 'slow', 'medium', 'fast']
        for vel_bin in vel_bins:
            key = f'velocity_{vel_bin}'
            if key in summary:
                acc = summary[key]['percentage']
                count = summary[key]['count']
                print(f"  {vel_bin.capitalize():15s}: {acc:6.2f}% ({count:4d} samples)")
        
        # Confusion matrix
        if task in self.confusion_matrices and self.confusion_matrices[task]:
            print(f"\n{'-'*70}")
            print("CONFUSION MATRIX (Top errors):")
            print(f"{'-'*70}")
            
            # Find most common errors
            errors = []
            for true_val, predictions in self.confusion_matrices[task].items():
                for pred_val, count in predictions.items():
                    if true_val != pred_val:
                        errors.append((count, true_val, pred_val))
            
            errors.sort(reverse=True)
            for count, true_val, pred_val in errors[:10]:
                print(f"  True: {true_val:3} → Predicted: {pred_val:3} ({count:4d} times)")
        
        print(f"\n{'='*70}\n")
    
    def detect_rnn_impact(self, task: str = 'counting', threshold: float = 5.0) -> Dict[str, bool]:
        """
        Detect if RNN temporal context is impacting performance.
        
        Args:
            task: 'counting' or 'arithmetic'
            threshold: Percentage difference threshold for "significant"
            
        Returns:
            Dictionary indicating if RNN impact detected in each category
        """
        summary = self.get_summary(task)
        impact = {}
        
        # Check moving vs stationary
        if 'moving' in summary and 'stationary' in summary:
            diff = abs(summary['moving']['percentage'] - summary['stationary']['percentage'])
            impact['motion_dependent'] = diff > threshold
            impact['motion_difference'] = diff
        
        # Check bouncing vs not bouncing
        if 'bouncing' in summary and 'not_bouncing' in summary:
            diff = abs(summary['bouncing']['percentage'] - summary['not_bouncing']['percentage'])
            impact['bounce_dependent'] = diff > threshold
            impact['bounce_difference'] = diff
        
        # Check fast vs slow
        if 'fast' in summary and 'slow' in summary:
            diff = abs(summary['fast']['percentage'] - summary['slow']['percentage'])
            impact['velocity_dependent'] = diff > threshold
            impact['velocity_difference'] = diff
        
        return impact
    
    def save_metrics(self, filepath: str):
        """Save metrics to file for later analysis."""
        from .serialization import save_json
        
        # Prepare data (serialization handled automatically)
        data = {
            'metrics': self.metrics,
            'confusion_matrices': self.confusion_matrices
        }
        
        save_json(data, filepath)
        print(f"Metrics saved to {filepath}")


# Factory function for creating instances (replaces global singleton)
def create_temporal_metrics_tracker() -> TemporalMetricsTracker:
    """
    Create a new TemporalMetricsTracker instance.
    
    Use this instead of accessing a global singleton to ensure:
    - Clean state for each training run
    - Testability (can create fresh instances)
    - No hidden dependencies
    
    Returns:
        New TemporalMetricsTracker instance
    """
    return TemporalMetricsTracker()
