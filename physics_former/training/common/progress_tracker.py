"""
Real-time Progress Tracker for Training

Provides detailed progress information including:
- Visual progress bar
- ETA calculations
- Throughput metrics
- Loss trends
- GPU/CPU monitoring
"""

import time
import torch
from typing import Optional, List
from collections import deque


class ProgressTracker:
    """
    Track training progress with detailed metrics and ETA.
    
    Usage:
        tracker = ProgressTracker(total_batches=1000, epoch=1, total_epochs=10)
        
        for batch_idx, batch in enumerate(dataloader):
            # ... training code ...
            
            tracker.update(
                batch_idx=batch_idx,
                loss=loss.item(),
                learning_rate=lr,
                grad_norm=grad_norm
            )
    """
    
    def __init__(
        self,
        total_batches: int,
        epoch: int = 1,
        total_epochs: int = 1,
        update_interval: int = 10,
        window_size: int = 50
    ):
        """
        Initialize progress tracker.
        
        Args:
            total_batches: Total number of batches in epoch
            epoch: Current epoch (1-indexed)
            total_epochs: Total number of epochs
            update_interval: Print update every N batches
            window_size: Window size for moving averages
        """
        self.total_batches = total_batches
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.update_interval = update_interval
        self.window_size = window_size
        
        # Timing
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.batch_times = deque(maxlen=window_size)
        
        # Metrics
        self.losses = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        self.grad_norms = deque(maxlen=window_size)
        
        # Counters
        self.batches_processed = 0
        self.samples_processed = 0
        
        print(f"\n{'='*80}")
        print(f"📅 EPOCH {epoch}/{total_epochs}")
        print(f"{'='*80}")
        print(f"Total batches: {total_batches}")
        print(f"Update interval: Every {update_interval} batches")
        print(f"{'='*80}\n")
    
    def update(
        self,
        batch_idx: int,
        loss: float,
        learning_rate: Optional[float] = None,
        grad_norm: Optional[float] = None,
        batch_size: Optional[int] = None,
        consolidations: Optional[int] = None,
        **kwargs
    ):
        """
        Update progress tracker with current batch metrics.
        
        Args:
            batch_idx: Current batch index (0-indexed)
            loss: Current loss value
            learning_rate: Current learning rate
            grad_norm: Gradient norm
            batch_size: Batch size (for throughput calculation)
            consolidations: Number of consolidations (CLS)
            **kwargs: Additional metrics to display
        """
        current_time = time.time()
        batch_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update metrics
        self.losses.append(loss)
        self.batch_times.append(batch_time)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        
        self.batches_processed = batch_idx + 1
        if batch_size:
            self.samples_processed += batch_size
        
        # Print update at specified interval
        if batch_idx % self.update_interval == 0 or batch_idx == self.total_batches - 1:
            self._print_progress(
                batch_idx,
                consolidations=consolidations,
                batch_size=batch_size,
                **kwargs
            )
    
    def _print_progress(
        self,
        batch_idx: int,
        consolidations: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """Print detailed progress information."""
        # Progress percentage
        progress_pct = (batch_idx / self.total_batches) * 100 if self.total_batches > 0 else 0
        
        # Progress bar
        bar_width = 40
        filled = int(bar_width * progress_pct / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        # Loss statistics
        if len(self.losses) > 0:
            avg_loss = sum(self.losses) / len(self.losses)
            min_loss = min(self.losses)
            max_loss = max(self.losses)
            current_loss = self.losses[-1]
        else:
            avg_loss = min_loss = max_loss = current_loss = 0.0
        
        # ETA calculation
        if len(self.batch_times) >= 5:
            avg_batch_time = sum(self.batch_times) / len(self.batch_times)
            remaining_batches = self.total_batches - batch_idx
            eta_seconds = avg_batch_time * remaining_batches
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # Throughput
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            batches_per_sec = self.batches_processed / elapsed
            if batch_size:
                samples_per_sec = self.samples_processed / elapsed
            else:
                samples_per_sec = None
        else:
            batches_per_sec = 0
            samples_per_sec = None
        
        # GPU memory
        gpu_mem_str = ""
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_mem_str = f" | GPU: {gpu_mem_allocated:.2f}/{gpu_mem_reserved:.2f}GB"
        
        # Print formatted output
        print(f"\n  [{bar}] {progress_pct:5.1f}%")
        print(f"  Batch {batch_idx:4d}/{self.total_batches} | ETA: {eta_str}")
        print(f"  Loss: {current_loss:.4f} (avg: {avg_loss:.4f}, min: {min_loss:.4f}, max: {max_loss:.4f})")
        
        # Learning rate and gradient
        if len(self.learning_rates) > 0:
            current_lr = self.learning_rates[-1]
            print(f"  LR: {current_lr:.2e}", end="")
        else:
            print(f"  LR: N/A", end="")
        
        if len(self.grad_norms) > 0:
            current_grad = self.grad_norms[-1]
            avg_grad = sum(self.grad_norms) / len(self.grad_norms)
            print(f" | Grad: {current_grad:.3f} (avg: {avg_grad:.3f})", end="")
        
        print()  # New line
        
        # Throughput
        if samples_per_sec:
            print(f"  Speed: {batches_per_sec:.1f} batches/s | {samples_per_sec:.1f} samples/s{gpu_mem_str}")
        else:
            print(f"  Speed: {batches_per_sec:.1f} batches/s{gpu_mem_str}")
        
        # Consolidations (CLS)
        if consolidations is not None:
            consolidation_pct = (consolidations / self.batches_processed * 100) if self.batches_processed > 0 else 0
            print(f"  Consolidations: {consolidations} ({consolidation_pct:.1f}% of batches)")
        
        # Additional metrics
        if kwargs:
            extra_metrics = []
            for key, value in kwargs.items():
                if isinstance(value, float):
                    extra_metrics.append(f"{key}: {value:.4f}")
                else:
                    extra_metrics.append(f"{key}: {value}")
            if extra_metrics:
                print(f"  {' | '.join(extra_metrics)}")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def epoch_summary(self) -> dict:
        """
        Get epoch summary statistics.
        
        Returns:
            dict: Summary statistics
        """
        elapsed = time.time() - self.start_time
        
        summary = {
            'epoch': self.epoch,
            'total_epochs': self.total_epochs,
            'batches_processed': self.batches_processed,
            'total_batches': self.total_batches,
            'elapsed_time': elapsed,
            'batches_per_sec': self.batches_processed / elapsed if elapsed > 0 else 0
        }
        
        if len(self.losses) > 0:
            summary['avg_loss'] = sum(self.losses) / len(self.losses)
            summary['min_loss'] = min(self.losses)
            summary['max_loss'] = max(self.losses)
            summary['final_loss'] = self.losses[-1]
        
        if len(self.learning_rates) > 0:
            summary['final_lr'] = self.learning_rates[-1]
        
        if len(self.grad_norms) > 0:
            summary['avg_grad_norm'] = sum(self.grad_norms) / len(self.grad_norms)
        
        if self.samples_processed > 0:
            summary['samples_processed'] = self.samples_processed
            summary['samples_per_sec'] = self.samples_processed / elapsed if elapsed > 0 else 0
        
        return summary
    
    def print_summary(self):
        """Print epoch summary."""
        summary = self.epoch_summary()
        
        print(f"\n{'='*80}")
        print(f"📊 EPOCH {self.epoch}/{self.total_epochs} SUMMARY")
        print(f"{'='*80}")
        
        # Time
        elapsed_str = self._format_time(summary['elapsed_time'])
        print(f"\n⏱️  Time: {elapsed_str}")
        print(f"  Batches: {summary['batches_processed']}/{summary['total_batches']}")
        print(f"  Throughput: {summary['batches_per_sec']:.1f} batches/sec")
        
        if 'samples_per_sec' in summary:
            print(f"  Samples: {summary['samples_processed']:,} ({summary['samples_per_sec']:.1f} samples/sec)")
        
        # Loss
        if 'avg_loss' in summary:
            print(f"\n📉 Loss:")
            print(f"  Average: {summary['avg_loss']:.4f}")
            print(f"  Final: {summary['final_loss']:.4f}")
            print(f"  Range: [{summary['min_loss']:.4f}, {summary['max_loss']:.4f}]")
        
        # Learning rate
        if 'final_lr' in summary:
            print(f"\n📚 Learning Rate: {summary['final_lr']:.2e}")
        
        # Gradient
        if 'avg_grad_norm' in summary:
            print(f"\n📊 Gradient Norm: {summary['avg_grad_norm']:.3f}")
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n🎮 Peak GPU Memory: {gpu_mem_peak:.2f}GB")
        
        # ETA for remaining epochs
        if self.epoch < self.total_epochs:
            remaining_epochs = self.total_epochs - self.epoch
            eta_seconds = summary['elapsed_time'] * remaining_epochs
            eta_str = self._format_time(eta_seconds)
            print(f"\n⏱️  ETA for remaining {remaining_epochs} epochs: ~{eta_str}")
        
        print(f"{'='*80}\n")


class SimpleProgressBar:
    """
    Lightweight progress bar for quick integration.
    
    Usage:
        progress = SimpleProgressBar(total=1000, desc="Training")
        for i in range(1000):
            # ... work ...
            progress.update(loss=loss_value)
    """
    
    def __init__(self, total: int, desc: str = "Progress", bar_width: int = 40):
        self.total = total
        self.desc = desc
        self.bar_width = bar_width
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1, **metrics):
        """Update progress by n steps."""
        self.current += n
        
        # Progress
        pct = (self.current / self.total) * 100 if self.total > 0 else 0
        filled = int(self.bar_width * self.current / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (self.bar_width - filled)
        
        # ETA
        elapsed = time.time() - self.start_time
        if self.current > 0 and elapsed > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate
            eta_str = f"ETA: {remaining:.0f}s"
        else:
            eta_str = "ETA: --"
        
        # Metrics
        metric_str = ""
        if metrics:
            metric_parts = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    metric_parts.append(f"{key}={value:.4f}")
                else:
                    metric_parts.append(f"{key}={value}")
            metric_str = " | " + " ".join(metric_parts)
        
        # Print (overwrite previous line)
        print(f"\r{self.desc}: [{bar}] {pct:5.1f}% {self.current}/{self.total} | {eta_str}{metric_str}", end="", flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def close(self):
        """Finish progress bar."""
        if self.current < self.total:
            self.current = self.total
            self.update(n=0)
