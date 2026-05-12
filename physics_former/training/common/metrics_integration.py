"""
Metrics logging integration for training pipelines.

Provides consistent metrics tracking across all pipelines.
"""

import torch
from ..metrics_logger import MetricsLogger


def setup_metrics_logger(
    experiment_name,
    config=None,
    log_dir="logs",
    save_interval=100,
    track_gpu=True
):
    """
    Initialize metrics logger with standard configuration.
    
    Args:
        experiment_name: Name of the experiment
        config: Training configuration (optional)
        log_dir: Directory to save logs
        save_interval: Save metrics every N steps
        track_gpu: Whether to track GPU metrics
    
    Returns:
        MetricsLogger instance
    """
    logger = MetricsLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        save_interval=save_interval,
        track_gpu=track_gpu and torch.cuda.is_available()
    )
    
    return logger


def log_dataset_metrics(logger, dataloaders):
    """
    Log metrics for all datasets.
    
    Args:
        logger: MetricsLogger instance
        dataloaders: Dictionary of dataloaders
    """
    print("\n" + "=" * 70)
    print("LOGGING DATASET METRICS")
    print("=" * 70)
    
    for stage_name, dataloader in dataloaders.items():
        if dataloader is not None:
            print(f"\nLogging {stage_name} dataset metrics...")
            try:
                logger.log_dataset_metrics(dataloader.dataset, dataset_name=stage_name)
            except Exception as e:
                print(f"  WARNING:  Could not log {stage_name} dataset: {e}")
    
    print("=" * 70)


def log_model_metrics(logger, model):
    """
    Log model architecture statistics.
    
    Args:
        logger: MetricsLogger instance
        model: PyTorch model
    """
    print("\n" + "=" * 70)
    print("LOGGING MODEL METRICS")
    print("=" * 70)
    
    if model is None:
        print("WARNING:  Model not initialized yet - skipping model metrics")
        print("=" * 70)
        return
    
    try:
        logger.log_model_stats(model)
        print("PASS: Model metrics logged")
    except Exception as e:
        print(f"WARNING:  Could not log model metrics: {e}")
    
    print("=" * 70)


def log_training_metrics(
    logger,
    epoch,
    step,
    loss,
    learning_rate=None,
    grad_norm=None,
    **kwargs
):
    """
    Log training metrics in standard format.
    
    Args:
        logger: MetricsLogger instance
        epoch: Current epoch
        step: Current step
        loss: Training loss
        learning_rate: Current learning rate
        grad_norm: Gradient norm
        **kwargs: Additional metrics (stage, task, consolidations, etc.)
    """
    if logger is None:
        return
    
    logger.log_training_step(
        epoch=epoch,
        step=step,
        loss=loss,
        learning_rate=learning_rate,
        grad_norm=grad_norm,
        **kwargs
    )


def log_validation_metrics(
    logger,
    epoch,
    step,
    val_loss,
    val_accuracy=None,
    **kwargs
):
    """
    Log validation metrics.
    
    Args:
        logger: MetricsLogger instance
        epoch: Current epoch
        step: Current step
        val_loss: Validation loss
        val_accuracy: Validation accuracy
        **kwargs: Additional metrics
    """
    if logger is None:
        return
    
    logger.log_validation_step(
        epoch=epoch,
        step=step,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        **kwargs
    )


def log_system_metrics(logger):
    """
    Log system resource usage.
    
    Args:
        logger: MetricsLogger instance
    """
    if logger is None:
        return
    
    logger.log_system_metrics()


def log_epoch_summary(logger, epoch, epoch_time):
    """
    Log end-of-epoch summary.
    
    Args:
        logger: MetricsLogger instance
        epoch: Epoch number
        epoch_time: Epoch duration in seconds
    """
    if logger is None:
        return
    
    logger.log_epoch_summary(epoch, epoch_time)


def finalize_metrics(logger):
    """
    Close logger and generate final report.
    
    Args:
        logger: MetricsLogger instance
    """
    if logger is None:
        return
    
    print("\n" + "=" * 70)
    print("GENERATING METRICS REPORT")
    print("=" * 70)
    
    logger.close()
    
    print(f"\nPASS: Metrics saved to: {logger.experiment_dir}")
    print("=" * 70)
