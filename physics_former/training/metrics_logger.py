"""
Comprehensive Metrics Logger for PhysicsFormer Experiments

Tracks and logs:
- Dataset statistics
- Training metrics (loss, accuracy, etc.)
- Model performance metrics
- System metrics (memory, GPU, timing)
- Data loading efficiency
- Batch statistics

Usage:
    logger = MetricsLogger(experiment_name="physics_transformer_v1")
    
    # Log dataset metrics
    logger.log_dataset_metrics(dataset)
    
    # Log training metrics
    logger.log_training_step(epoch, step, loss, accuracy)
    
    # Log system metrics
    logger.log_system_metrics()
    
    # Save report
    logger.save_report()
"""

import time
import json
import logging
from .utils.serialization import save_json, to_serializable
import psutil
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
import csv


class MetricsLogger:
    """Comprehensive metrics logger for experiment tracking."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        save_interval: int = 100,
        track_gpu: bool = True
    ):
        """
        Initialize metrics logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save logs
            save_interval: Save metrics every N steps
            track_gpu: Whether to track GPU metrics
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.save_interval = save_interval
        self.track_gpu = track_gpu and torch.cuda.is_available()
        
        # Create log directory
        self.experiment_dir = self.log_dir / experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric storage
        self.metrics = defaultdict(list)
        self.dataset_stats = {}
        self.model_stats = {}
        self.system_stats = defaultdict(list)
        
        # Timing
        self.start_time = time.time()
        self.step_times = []
        self.epoch_times = []
        
        # Counters
        self.global_step = 0
        self.current_epoch = 0
        
        # CSV writers
        self.csv_files = {}
        self._init_csv_files()
        
        print(f"[METRICS] Metrics Logger initialized")
        print(f"   Experiment: {experiment_name}")
        print(f"   Log directory: {self.experiment_dir}")
        print(f"   GPU tracking: {'Enabled' if self.track_gpu else 'Disabled'}")
    
    def _init_csv_files(self):
        """Initialize CSV files for different metric types."""
        csv_configs = {
            'training': ['epoch', 'step', 'loss', 'learning_rate', 'grad_norm', 'time'],
            'validation': ['epoch', 'step', 'val_loss', 'val_accuracy', 'time'],
            'system': ['step', 'cpu_percent', 'memory_mb', 'gpu_memory_mb', 'gpu_utilization'],
            'data_loading': ['step', 'batch_load_time', 'collate_time', 'batch_size', 'num_objects'],
        }
        
        for name, headers in csv_configs.items():
            csv_path = self.experiment_dir / f"{name}_metrics.csv"
            f = open(csv_path, 'w', newline='')
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            self.csv_files[name] = {'file': f, 'writer': writer}
    
    def log_dataset_metrics(self, dataset, dataset_name: str = "train"):
        """
        Log comprehensive dataset statistics.
        
        Args:
            dataset: Dataset object
            dataset_name: Name of the dataset (train/val/test)
        """
        print(f"\n[METRICS] Logging {dataset_name} dataset metrics...")
        
        stats = {
            'name': dataset_name,
            'total_episodes': len(dataset),
            'timestamp': datetime.now().isoformat()
        }
        
        # Sample dataset to get statistics
        sample_size = min(100, len(dataset))
        samples = [dataset[i] for i in range(sample_size)]
        
        # Object counts
        object_counts = [s['num_objects'] for s in samples if 'num_objects' in s]
        if object_counts:
            stats['object_count'] = {
                'mean': float(np.mean(object_counts)),
                'std': float(np.std(object_counts)),
                'min': int(np.min(object_counts)),
                'max': int(np.max(object_counts)),
                'median': float(np.median(object_counts))
            }
        
        # Sequence lengths
        seq_lengths = [s['length'] for s in samples if 'length' in s]
        if seq_lengths:
            stats['sequence_length'] = {
                'mean': float(np.mean(seq_lengths)),
                'std': float(np.std(seq_lengths)),
                'min': int(np.min(seq_lengths)),
                'max': int(np.max(seq_lengths)),
                'median': float(np.median(seq_lengths))
            }
        
        # Schema distribution
        if hasattr(dataset, 'schema_to_id'):
            schema_counts = defaultdict(int)
            for s in samples:
                if 'schema_label' in s:
                    schema_id = s['schema_label'].item() if torch.is_tensor(s['schema_label']) else s['schema_label']
                    schema_name = dataset.id_to_schema.get(schema_id, 'unknown')
                    schema_counts[schema_name] += 1
            stats['schema_distribution'] = dict(schema_counts)
        
        # Trajectory statistics
        if samples and 'object_trajectories' in samples[0]:
            traj_shapes = []
            for s in samples:
                if isinstance(s['object_trajectories'], list):
                    for traj in s['object_trajectories']:
                        if torch.is_tensor(traj):
                            traj_shapes.append(traj.shape)
            
            if traj_shapes:
                stats['trajectory_shape'] = {
                    'typical_shape': str(traj_shapes[0]),
                    'feature_dim': traj_shapes[0][-1] if traj_shapes else None
                }
        
        # Pairwise distance statistics
        if samples and 'pairwise_distances' in samples[0]:
            distances = []
            for s in samples[:10]:  # Sample 10 for efficiency
                dist = s['pairwise_distances']
                if torch.is_tensor(dist):
                    # Get non-zero distances (actual object pairs)
                    non_zero = dist[dist > 0]
                    if len(non_zero) > 0:
                        distances.extend(non_zero.cpu().numpy().tolist())
            
            if distances:
                stats['pairwise_distances'] = {
                    'mean': float(np.mean(distances)),
                    'std': float(np.std(distances)),
                    'min': float(np.min(distances)),
                    'max': float(np.max(distances))
                }
        
        # Dataset configuration
        if hasattr(dataset, 'max_objects'):
            stats['config'] = {
                'max_objects': dataset.max_objects,
                'max_seq_length': dataset.max_seq_length,
                'lazy_load': dataset.lazy_load if hasattr(dataset, 'lazy_load') else None
            }
        
        # Cache statistics
        if hasattr(dataset, 'episode_index'):
            stats['cache_info'] = {
                'indexed_episodes': len(dataset.episode_index),
                'cache_enabled': True
            }
        
        self.dataset_stats[dataset_name] = stats
        
        # Print summary
        print(f"   Total episodes: {stats['total_episodes']:,}")
        if 'object_count' in stats:
            print(f"   Objects per episode: {stats['object_count']['mean']:.1f} ± {stats['object_count']['std']:.1f}")
        if 'sequence_length' in stats:
            print(f"   Sequence length: {stats['sequence_length']['mean']:.1f} ± {stats['sequence_length']['std']:.1f}")
        if 'schema_distribution' in stats:
            print(f"   Schemas: {len(stats['schema_distribution'])} types")
        
        # Save to JSON
        save_json(stats, self.experiment_dir / f"{dataset_name}_dataset_stats.json")
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float = None,
        grad_norm: float = None,
        **kwargs
    ):
        """
        Log training step metrics.
        
        Args:
            epoch: Current epoch
            step: Current step
            loss: Training loss
            learning_rate: Current learning rate
            grad_norm: Gradient norm
            **kwargs: Additional metrics
        """
        self.global_step = step
        self.current_epoch = epoch
        
        # Record metrics
        self.metrics['train_loss'].append(loss)
        if learning_rate is not None:
            self.metrics['learning_rate'].append(learning_rate)
        if grad_norm is not None:
            self.metrics['grad_norm'].append(grad_norm)
        
        # Additional metrics
        for key, value in kwargs.items():
            self.metrics[f'train_{key}'].append(value)
        
        # Write to CSV
        csv_data = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'learning_rate': learning_rate if learning_rate is not None else '',
            'grad_norm': grad_norm if grad_norm is not None else '',
            'time': time.time() - self.start_time
        }
        self.csv_files['training']['writer'].writerow(csv_data)
        
        # Periodic save
        if step % self.save_interval == 0:
            self._flush_csv()
    
    def log_validation_step(
        self,
        epoch: int,
        step: int,
        val_loss: float,
        val_accuracy: float = None,
        **kwargs
    ):
        """Log validation metrics."""
        self.metrics['val_loss'].append(val_loss)
        if val_accuracy is not None:
            self.metrics['val_accuracy'].append(val_accuracy)
        
        for key, value in kwargs.items():
            self.metrics[f'val_{key}'].append(value)
        
        # Write to CSV
        csv_data = {
            'epoch': epoch,
            'step': step,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy if val_accuracy is not None else '',
            'time': time.time() - self.start_time
        }
        self.csv_files['validation']['writer'].writerow(csv_data)
        self._flush_csv()
    
    def log_batch_metrics(
        self,
        batch_load_time: float,
        collate_time: float = None,
        batch_size: int = None,
        num_objects: int = None,
        **kwargs
    ):
        """Log data loading and batch metrics."""
        self.metrics['batch_load_time'].append(batch_load_time)
        
        csv_data = {
            'step': self.global_step,
            'batch_load_time': batch_load_time,
            'collate_time': collate_time if collate_time is not None else '',
            'batch_size': batch_size if batch_size is not None else '',
            'num_objects': num_objects if num_objects is not None else ''
        }
        self.csv_files['data_loading']['writer'].writerow(csv_data)
    
    def log_system_metrics(self):
        """Log system resource usage."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        
        self.system_stats['cpu_percent'].append(cpu_percent)
        self.system_stats['memory_mb'].append(memory_mb)
        
        csv_data = {
            'step': self.global_step,
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'gpu_memory_mb': '',
            'gpu_utilization': ''
        }
        
        # GPU metrics
        if self.track_gpu:
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                
                self.system_stats['gpu_memory_mb'].append(gpu_memory)
                self.system_stats['gpu_memory_reserved_mb'].append(gpu_memory_reserved)
                
                csv_data['gpu_memory_mb'] = gpu_memory
                
                # Try to get GPU utilization (requires nvidia-ml-py3)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    csv_data['gpu_utilization'] = utilization.gpu
                    self.system_stats['gpu_utilization'].append(utilization.gpu)
                except (ImportError, AttributeError) as e:
                    # pynvml not available or GPU not accessible
                    logging.debug(f"GPU metrics unavailable: {e}")
            except (psutil.Error, OSError) as e:
                # System metrics unavailable
                logging.debug(f"System metrics unavailable: {e}")
        
        self.csv_files['system']['writer'].writerow(csv_data)
    
    def log_model_stats(self, model):
        """Log model architecture statistics."""
        print("\n[METRICS] Logging model statistics...")
        
        stats = {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
            'timestamp': datetime.now().isoformat()
        }
        
        # Layer-wise parameters
        layer_params = {}
        for name, param in model.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name not in layer_params:
                layer_params[layer_name] = 0
            layer_params[layer_name] += param.numel()
        
        stats['parameters_by_layer'] = layer_params
        
        # Model configuration
        if hasattr(model, 'config'):
            stats['config'] = {
                k: v for k, v in vars(model.config).items()
                if not k.startswith('_') and isinstance(v, (int, float, str, bool))
            }
        
        self.model_stats = stats
        
        print(f"   Total parameters: {stats['total_parameters']:,}")
        print(f"   Trainable parameters: {stats['trainable_parameters']:,}")
        print(f"   Model size: {stats['model_size_mb']:.2f} MB")
        
        # Save to JSON
        save_json(stats, self.experiment_dir / "model_stats.json")
    
    def log_epoch_summary(self, epoch: int, epoch_time: float):
        """Log end-of-epoch summary."""
        self.epoch_times.append(epoch_time)
        
        summary = {
            'epoch': epoch,
            'epoch_time': epoch_time,
            'avg_train_loss': np.mean(self.metrics['train_loss'][-100:]) if self.metrics['train_loss'] else None,
            'avg_val_loss': np.mean(self.metrics['val_loss'][-10:]) if self.metrics['val_loss'] else None,
        }
        
        print(f"\n[METRICS] Epoch {epoch} Summary:")
        print(f"   Time: {epoch_time:.2f}s")
        if summary['avg_train_loss']:
            print(f"   Avg Train Loss: {summary['avg_train_loss']:.4f}")
        if summary['avg_val_loss']:
            print(f"   Avg Val Loss: {summary['avg_val_loss']:.4f}")
    
    def log_data_loader_efficiency(self, dataloader, num_batches: int = 10):
        """Benchmark and log data loader efficiency."""
        print(f"\n[METRICS] Benchmarking DataLoader efficiency...")
        
        times = []
        batch_sizes = []
        
        start = time.time()
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            batch_time = time.time() - start
            times.append(batch_time)
            
            # Get batch size
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch_sizes.append(value.shape[0] if len(value.shape) > 0 else 1)
                        break
            
            start = time.time()
        
        stats = {
            'avg_batch_time': np.mean(times),
            'std_batch_time': np.std(times),
            'min_batch_time': np.min(times),
            'max_batch_time': np.max(times),
            'throughput_batches_per_sec': 1.0 / np.mean(times) if times else 0,
            'avg_batch_size': np.mean(batch_sizes) if batch_sizes else 0
        }
        
        print(f"   Avg batch time: {stats['avg_batch_time']*1000:.2f}ms")
        print(f"   Throughput: {stats['throughput_batches_per_sec']:.2f} batches/sec")
        print(f"   Avg batch size: {stats['avg_batch_size']:.1f}")
        
        self.metrics['dataloader_efficiency'] = stats
        
        return stats
    
    def _flush_csv(self):
        """Flush CSV files to disk."""
        for csv_info in self.csv_files.values():
            csv_info['file'].flush()
    
    def save_report(self):
        """Generate and save comprehensive experiment report."""
        print(f"\n[METRICS] Generating experiment report...")
        
        total_time = time.time() - self.start_time
        
        report = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': total_time,
            'total_duration_formatted': self._format_duration(total_time),
            'total_steps': self.global_step,
            'total_epochs': self.current_epoch,
            'dataset_stats': self.dataset_stats,
            'model_stats': self.model_stats,
        }
        
        # Training summary
        if self.metrics['train_loss']:
            report['training_summary'] = {
                'final_train_loss': self.metrics['train_loss'][-1],
                'best_train_loss': min(self.metrics['train_loss']),
                'avg_train_loss': np.mean(self.metrics['train_loss']),
            }
        
        # Validation summary
        if self.metrics['val_loss']:
            report['validation_summary'] = {
                'final_val_loss': self.metrics['val_loss'][-1],
                'best_val_loss': min(self.metrics['val_loss']),
                'avg_val_loss': np.mean(self.metrics['val_loss']),
            }
        
        # System summary
        if self.system_stats['cpu_percent']:
            report['system_summary'] = {
                'avg_cpu_percent': np.mean(self.system_stats['cpu_percent']),
                'peak_memory_mb': max(self.system_stats['memory_mb']),
                'avg_memory_mb': np.mean(self.system_stats['memory_mb']),
            }
            
            if 'gpu_memory_mb' in self.system_stats and self.system_stats['gpu_memory_mb']:
                report['system_summary']['peak_gpu_memory_mb'] = max(self.system_stats['gpu_memory_mb'])
                report['system_summary']['avg_gpu_memory_mb'] = np.mean(self.system_stats['gpu_memory_mb'])
        
        # Timing summary
        if self.epoch_times:
            report['timing_summary'] = {
                'avg_epoch_time': np.mean(self.epoch_times),
                'total_epoch_time': sum(self.epoch_times),
                'fastest_epoch': min(self.epoch_times),
                'slowest_epoch': max(self.epoch_times),
            }
        
        # Data loading efficiency
        if 'dataloader_efficiency' in self.metrics:
            report['dataloader_efficiency'] = self.metrics['dataloader_efficiency']
        
        # Save report
        report_path = self.experiment_dir / "experiment_report.json"
        save_json(report, report_path)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        print(f"   Report saved to: {report_path}")
        print(f"   Total duration: {report['total_duration_formatted']}")
        
        return report
    
    def _generate_markdown_report(self, report: Dict):
        """Generate human-readable markdown report."""
        md_path = self.experiment_dir / "experiment_report.md"
        
        with open(md_path, 'w') as f:
            f.write(f"# Experiment Report: {self.experiment_name}\n\n")
            f.write(f"**Generated:** {report['end_time']}\n\n")
            f.write(f"**Duration:** {report['total_duration_formatted']}\n\n")
            
            # Dataset stats
            f.write("## Dataset Statistics\n\n")
            for dataset_name, stats in report['dataset_stats'].items():
                f.write(f"### {dataset_name.title()} Dataset\n\n")
                f.write(f"- **Total Episodes:** {stats['total_episodes']:,}\n")
                if 'object_count' in stats:
                    oc = stats['object_count']
                    f.write(f"- **Objects per Episode:** {oc['mean']:.1f} ± {oc['std']:.1f} (range: {oc['min']}-{oc['max']})\n")
                if 'sequence_length' in stats:
                    sl = stats['sequence_length']
                    f.write(f"- **Sequence Length:** {sl['mean']:.1f} ± {sl['std']:.1f} (range: {sl['min']}-{sl['max']})\n")
                if 'schema_distribution' in stats:
                    f.write(f"- **Schemas:** {len(stats['schema_distribution'])} types\n")
                f.write("\n")
            
            # Model stats
            if report['model_stats']:
                f.write("## Model Statistics\n\n")
                ms = report['model_stats']
                f.write(f"- **Total Parameters:** {ms['total_parameters']:,}\n")
                f.write(f"- **Trainable Parameters:** {ms['trainable_parameters']:,}\n")
                f.write(f"- **Model Size:** {ms['model_size_mb']:.2f} MB\n\n")
            
            # Training summary
            if 'training_summary' in report:
                f.write("## Training Summary\n\n")
                ts = report['training_summary']
                f.write(f"- **Final Loss:** {ts['final_train_loss']:.4f}\n")
                f.write(f"- **Best Loss:** {ts['best_train_loss']:.4f}\n")
                f.write(f"- **Average Loss:** {ts['avg_train_loss']:.4f}\n\n")
            
            # Validation summary
            if 'validation_summary' in report:
                f.write("## Validation Summary\n\n")
                vs = report['validation_summary']
                f.write(f"- **Final Loss:** {vs['final_val_loss']:.4f}\n")
                f.write(f"- **Best Loss:** {vs['best_val_loss']:.4f}\n")
                f.write(f"- **Average Loss:** {vs['avg_val_loss']:.4f}\n\n")
            
            # System summary
            if 'system_summary' in report:
                f.write("## System Resource Usage\n\n")
                ss = report['system_summary']
                f.write(f"- **Average CPU:** {ss['avg_cpu_percent']:.1f}%\n")
                f.write(f"- **Peak Memory:** {ss['peak_memory_mb']:.0f} MB\n")
                f.write(f"- **Average Memory:** {ss['avg_memory_mb']:.0f} MB\n")
                if 'peak_gpu_memory_mb' in ss:
                    f.write(f"- **Peak GPU Memory:** {ss['peak_gpu_memory_mb']:.0f} MB\n")
                f.write("\n")
            
            # Timing
            if 'timing_summary' in report:
                f.write("## Timing Summary\n\n")
                ts = report['timing_summary']
                f.write(f"- **Average Epoch Time:** {ts['avg_epoch_time']:.2f}s\n")
                f.write(f"- **Fastest Epoch:** {ts['fastest_epoch']:.2f}s\n")
                f.write(f"- **Slowest Epoch:** {ts['slowest_epoch']:.2f}s\n\n")
        
        print(f"   Markdown report saved to: {md_path}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def close(self):
        """Close all CSV files and save final report."""
        print("\n[METRICS] Closing metrics logger...")
        
        # Close CSV files
        for csv_info in self.csv_files.values():
            csv_info['file'].close()
        
        # Save final report
        self.save_report()
        
        print("   PASS: Metrics logger closed")


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = MetricsLogger(experiment_name="test_experiment")
    
    # Simulate training
    for epoch in range(3):
        epoch_start = time.time()
        
        for step in range(100):
            # Simulate training step
            loss = 1.0 / (step + 1)
            lr = 0.001 * (0.99 ** step)
            
            logger.log_training_step(
                epoch=epoch,
                step=step,
                loss=loss,
                learning_rate=lr,
                grad_norm=0.5
            )
            
            # Log system metrics every 10 steps
            if step % 10 == 0:
                logger.log_system_metrics()
            
            time.sleep(0.01)  # Simulate work
        
        # Validation
        logger.log_validation_step(
            epoch=epoch,
            step=step,
            val_loss=0.5,
            val_accuracy=0.85
        )
        
        epoch_time = time.time() - epoch_start
        logger.log_epoch_summary(epoch, epoch_time)
    
    # Close and generate report
    logger.close()
