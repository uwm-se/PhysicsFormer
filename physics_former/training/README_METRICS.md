## Comprehensive Metrics Logging System

Complete experiment tracking and reporting for PhysicsFormer training.

## Overview

The metrics logging system tracks:
- **Dataset Statistics** - Episode counts, object distributions, sequence lengths
- **Training Metrics** - Loss, learning rate, gradient norms
- **Validation Metrics** - Validation loss, accuracy
- **System Metrics** - CPU, memory, GPU usage
- **Data Loading** - Batch times, throughput
- **Model Statistics** - Parameters, layer sizes

## Quick Start

```python
from metrics_logger import MetricsLogger

# Initialize logger
logger = MetricsLogger(
    experiment_name="physics_transformer_v1",
    log_dir="logs",
    save_interval=100,
    track_gpu=True
)

# Log dataset metrics
logger.log_dataset_metrics(train_dataset, dataset_name="train")
logger.log_dataset_metrics(val_dataset, dataset_name="validation")

# Log model statistics
logger.log_model_stats(model)

# Training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        # ... training code ...
        
        # Log training metrics
        logger.log_training_step(
            epoch=epoch,
            step=step,
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]['lr'],
            grad_norm=grad_norm
        )
        
        # Log system metrics periodically
        if step % 10 == 0:
            logger.log_system_metrics()
    
    # Validation
    logger.log_validation_step(
        epoch=epoch,
        step=step,
        val_loss=val_loss,
        val_accuracy=val_accuracy
    )
    
    logger.log_epoch_summary(epoch, epoch_time)

# Generate final report
logger.close()
```

## Logged Metrics

### Dataset Metrics

**Automatically tracked:**
- Total episodes
- Object count statistics (mean, std, min, max, median)
- Sequence length statistics
- Schema distribution
- Trajectory shapes
- Pairwise distance statistics
- Cache information

**Example output:**
```
📊 Logging train dataset metrics...
   Total episodes: 10,000
   Objects per episode: 5.2 ± 1.3
   Sequence length: 87.5 ± 12.4
   Schemas: 3 types
```

### Training Metrics

**Per-step tracking:**
- Loss
- Learning rate
- Gradient norm
- Custom metrics (via kwargs)

**Saved to:** `training_metrics.csv`

**Columns:**
```
epoch, step, loss, learning_rate, grad_norm, time
```

### Validation Metrics

**Per-epoch tracking:**
- Validation loss
- Validation accuracy
- Custom metrics

**Saved to:** `validation_metrics.csv`

**Columns:**
```
epoch, step, val_loss, val_accuracy, time
```

### System Metrics

**Resource tracking:**
- CPU utilization (%)
- Memory usage (MB)
- GPU memory (MB)
- GPU utilization (%) *requires nvidia-ml-py3*

**Saved to:** `system_metrics.csv`

**Columns:**
```
step, cpu_percent, memory_mb, gpu_memory_mb, gpu_utilization
```

### Data Loading Metrics

**Efficiency tracking:**
- Batch load time
- Collate time
- Batch size
- Number of objects

**Saved to:** `data_loading_metrics.csv`

**Columns:**
```
step, batch_load_time, collate_time, batch_size, num_objects
```

### Model Statistics

**One-time tracking:**
- Total parameters
- Trainable parameters
- Model size (MB)
- Parameters by layer
- Model configuration

**Saved to:** `model_stats.json`

## Output Structure

```
logs/
└── experiment_name/
    └── 20231124_143022/
        ├── experiment_report.json          # Complete JSON report
        ├── experiment_report.md            # Human-readable report
        ├── training_metrics.csv            # Training metrics
        ├── validation_metrics.csv          # Validation metrics
        ├── system_metrics.csv              # System resource usage
        ├── data_loading_metrics.csv        # Data loading efficiency
        ├── train_dataset_stats.json        # Training dataset stats
        ├── validation_dataset_stats.json   # Validation dataset stats
        ├── model_stats.json                # Model architecture stats
        └── plots/                          # Visualizations (optional)
            ├── dashboard.png
            ├── training_metrics.png
            ├── validation_metrics.png
            ├── system_metrics.png
            ├── data_loading_metrics.png
            └── dataset_statistics.png
```

## Experiment Report

The final report includes:

### JSON Report (`experiment_report.json`)
```json
{
  "experiment_name": "physics_transformer_v1",
  "start_time": "2023-11-24T14:30:22",
  "end_time": "2023-11-24T16:45:33",
  "total_duration_seconds": 8111.5,
  "total_duration_formatted": "2h 15m 11s",
  "total_steps": 10000,
  "total_epochs": 10,
  "dataset_stats": { ... },
  "model_stats": { ... },
  "training_summary": {
    "final_train_loss": 0.0234,
    "best_train_loss": 0.0198,
    "avg_train_loss": 0.0456
  },
  "validation_summary": {
    "final_val_loss": 0.0312,
    "best_val_loss": 0.0287,
    "avg_val_loss": 0.0398
  },
  "system_summary": {
    "avg_cpu_percent": 45.2,
    "peak_memory_mb": 8192,
    "avg_memory_mb": 6144,
    "peak_gpu_memory_mb": 4096
  },
  "timing_summary": {
    "avg_epoch_time": 811.15,
    "fastest_epoch": 765.23,
    "slowest_epoch": 892.45
  }
}
```

### Markdown Report (`experiment_report.md`)

Human-readable report with:
- Experiment overview
- Dataset statistics
- Model statistics
- Training summary
- Validation summary
- System resource usage
- Timing summary

## Visualization

Generate plots from logged metrics:

```bash
python visualize_metrics.py --experiment logs/experiment_name/20231124_143022
```

**Generated plots:**
1. **dashboard.png** - Overview dashboard with key metrics
2. **training_metrics.png** - Training loss and learning rate
3. **validation_metrics.png** - Validation loss and accuracy
4. **system_metrics.png** - CPU, memory, GPU usage
5. **data_loading_metrics.png** - Batch loading times
6. **dataset_statistics.png** - Dataset distributions

### Custom output directory:
```bash
python visualize_metrics.py --experiment logs/exp/20231124_143022 --output custom_plots/
```

## Advanced Usage

### Custom Metrics

Log additional metrics:

```python
logger.log_training_step(
    epoch=epoch,
    step=step,
    loss=loss.item(),
    learning_rate=lr,
    custom_metric_1=value1,
    custom_metric_2=value2
)
```

### Benchmark DataLoader

Measure data loading efficiency:

```python
efficiency_stats = logger.log_data_loader_efficiency(
    dataloader=train_loader,
    num_batches=20
)

print(f"Throughput: {efficiency_stats['throughput_batches_per_sec']:.2f} batches/sec")
```

### Periodic System Monitoring

```python
# Log system metrics every N steps
if step % 10 == 0:
    logger.log_system_metrics()
```

### Save Interval

Control how often metrics are flushed to disk:

```python
logger = MetricsLogger(
    experiment_name="my_experiment",
    save_interval=50  # Flush every 50 steps
)
```

## Integration with Training Pipeline

See `example_training_with_metrics.py` for complete integration example.

### Key Integration Points

1. **Before training:**
   ```python
   logger = MetricsLogger(experiment_name="exp_name")
   logger.log_dataset_metrics(train_dataset, "train")
   logger.log_model_stats(model)
   ```

2. **During training:**
   ```python
   logger.log_training_step(epoch, step, loss, lr, grad_norm)
   logger.log_batch_metrics(batch_time, batch_size)
   logger.log_system_metrics()  # Periodic
   ```

3. **During validation:**
   ```python
   logger.log_validation_step(epoch, step, val_loss, val_accuracy)
   ```

4. **After epoch:**
   ```python
   logger.log_epoch_summary(epoch, epoch_time)
   ```

5. **After training:**
   ```python
   logger.close()  # Generates final report
   ```

## Performance Impact

The metrics logger is designed to have minimal overhead:

- **CSV writing:** Buffered, flushed periodically
- **System metrics:** Lightweight psutil calls
- **GPU metrics:** Optional, can be disabled
- **Typical overhead:** < 1% of training time

### Disable GPU tracking:
```python
logger = MetricsLogger(
    experiment_name="exp_name",
    track_gpu=False  # Disable GPU metrics
)
```

## Requirements

```
# Core requirements
torch>=1.9.0
numpy>=1.19.0
psutil>=5.8.0

# For visualization
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.2.0

# Optional: for GPU utilization tracking
nvidia-ml-py3>=7.352.0
```

## Best Practices

### 1. Meaningful Experiment Names
```python
experiment_name = f"transformer_d{d_model}_h{n_heads}_lr{lr}_bs{batch_size}"
```

### 2. Log Early and Often
- Log dataset stats before training
- Log model stats after initialization
- Log system metrics periodically (every 10-50 steps)

### 3. Save Checkpoints with Metrics
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics_dir': str(logger.experiment_dir)
}
```

### 4. Review Reports After Training
- Check `experiment_report.md` for human-readable summary
- Use `experiment_report.json` for programmatic analysis
- Generate visualizations for presentations

### 5. Track Multiple Experiments
```python
# Organize by date and configuration
experiment_name = f"{datetime.now().strftime('%Y%m%d')}_transformer_v1"
```

## Troubleshooting

### High Memory Usage
- Reduce `save_interval` to flush more frequently
- Disable GPU tracking if not needed
- Use lazy loading for datasets

### Slow Logging
- Increase `save_interval`
- Disable system metrics during training
- Log system metrics only at epoch boundaries

### Missing Plots
- Install visualization dependencies: `pip install matplotlib seaborn pandas`
- Check that CSV files exist in experiment directory

### GPU Metrics Not Available
- Install nvidia-ml-py3: `pip install nvidia-ml-py3`
- Or disable GPU tracking: `track_gpu=False`

## Examples

### Minimal Example
```python
logger = MetricsLogger("quick_test")
logger.log_dataset_metrics(dataset)
logger.log_model_stats(model)

for epoch in range(10):
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)
        logger.log_training_step(epoch, step, loss)

logger.close()
```

### Full Example
See `example_training_with_metrics.py` for complete training loop with all metrics.

## Summary

The metrics logging system provides:
- PASS: Comprehensive experiment tracking
- PASS: Minimal performance overhead
- PASS: Automatic report generation
- PASS: CSV export for analysis
- PASS: Visualization tools
- PASS: Easy integration
- PASS: Production-ready

**Use this system for all experiments to ensure reproducibility and thorough analysis!**
