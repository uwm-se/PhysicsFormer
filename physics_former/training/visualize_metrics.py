"""
Visualize Experiment Metrics

Creates plots and visualizations from logged metrics.

Usage:
    python visualize_metrics.py --experiment logs/experiment_name/20231124_143022
    python visualize_metrics.py --experiment logs/experiment_name/20231124_143022 --output plots/
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def plot_training_metrics(experiment_dir: Path, output_dir: Path):
    """Plot training loss and learning rate curves."""
    # Load training metrics
    training_csv = experiment_dir / "training_metrics.csv"
    if not training_csv.exists():
        print("WARNING: No training metrics found")
        return
    
    df = pd.read_csv(training_csv)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot loss
    axes[0].plot(df['step'], df['loss'], label='Training Loss', alpha=0.7)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'learning_rate' in df.columns and df['learning_rate'].notna().any():
        axes[1].plot(df['step'], df['learning_rate'], label='Learning Rate', color='orange')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
    print(f"PASS: Saved: training_metrics.png")
    plt.close()


def plot_validation_metrics(experiment_dir: Path, output_dir: Path):
    """Plot validation loss and accuracy."""
    validation_csv = experiment_dir / "validation_metrics.csv"
    if not validation_csv.exists():
        print("WARNING: No validation metrics found")
        return
    
    df = pd.read_csv(validation_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot validation loss
    axes[0].plot(df['epoch'], df['val_loss'], marker='o', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot validation accuracy
    if 'val_accuracy' in df.columns and df['val_accuracy'].notna().any():
        axes[1].plot(df['epoch'], df['val_accuracy'], marker='o', color='green', label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_metrics.png', dpi=300, bbox_inches='tight')
    print(f"PASS: Saved: validation_metrics.png")
    plt.close()


def plot_system_metrics(experiment_dir: Path, output_dir: Path):
    """Plot system resource usage."""
    system_csv = experiment_dir / "system_metrics.csv"
    if not system_csv.exists():
        print("WARNING: No system metrics found")
        return
    
    df = pd.read_csv(system_csv)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot CPU and Memory
    ax1 = axes[0]
    ax2 = ax1.twinx()
    
    ax1.plot(df['step'], df['cpu_percent'], label='CPU %', color='blue', alpha=0.7)
    ax2.plot(df['step'], df['memory_mb'], label='Memory (MB)', color='red', alpha=0.7)
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('CPU %', color='blue')
    ax2.set_ylabel('Memory (MB)', color='red')
    ax1.set_title('CPU and Memory Usage')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot GPU memory if available
    if 'gpu_memory_mb' in df.columns and df['gpu_memory_mb'].notna().any():
        axes[1].plot(df['step'], df['gpu_memory_mb'], label='GPU Memory (MB)', color='green')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('GPU Memory (MB)')
        axes[1].set_title('GPU Memory Usage')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'system_metrics.png', dpi=300, bbox_inches='tight')
    print(f"PASS: Saved: system_metrics.png")
    plt.close()


def plot_data_loading_metrics(experiment_dir: Path, output_dir: Path):
    """Plot data loading efficiency."""
    data_csv = experiment_dir / "data_loading_metrics.csv"
    if not data_csv.exists():
        print("WARNING: No data loading metrics found")
        return
    
    df = pd.read_csv(data_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot batch load time
    axes[0].plot(df['step'], df['batch_load_time'] * 1000, alpha=0.5)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Batch Load Time')
    axes[0].grid(True, alpha=0.3)
    
    # Add moving average
    window = min(50, len(df) // 10)
    if window > 1:
        moving_avg = df['batch_load_time'].rolling(window=window).mean() * 1000
        axes[0].plot(df['step'], moving_avg, color='red', linewidth=2, label=f'{window}-step MA')
        axes[0].legend()
    
    # Plot batch size distribution
    if 'batch_size' in df.columns and df['batch_size'].notna().any():
        axes[1].hist(df['batch_size'].dropna(), bins=20, edgecolor='black')
        axes[1].set_xlabel('Batch Size')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Batch Size Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_loading_metrics.png', dpi=300, bbox_inches='tight')
    print(f"PASS: Saved: data_loading_metrics.png")
    plt.close()


def plot_dataset_statistics(experiment_dir: Path, output_dir: Path):
    """Plot dataset statistics."""
    # Load dataset stats
    dataset_files = list(experiment_dir.glob("*_dataset_stats.json"))
    
    if not dataset_files:
        print("WARNING: No dataset statistics found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for dataset_file in dataset_files:
        with open(dataset_file, 'r') as f:
            stats = json.load(f)
        
        dataset_name = stats['name']
        
        # Plot object count distribution
        if 'object_count' in stats:
            oc = stats['object_count']
            axes[0, 0].bar(dataset_name, oc['mean'], yerr=oc['std'], capsize=5, alpha=0.7)
        
        # Plot sequence length distribution
        if 'sequence_length' in stats:
            sl = stats['sequence_length']
            axes[0, 1].bar(dataset_name, sl['mean'], yerr=sl['std'], capsize=5, alpha=0.7)
        
        # Plot schema distribution
        if 'schema_distribution' in stats:
            schemas = stats['schema_distribution']
            x = list(schemas.keys())
            y = list(schemas.values())
            axes[1, 0].bar(x, y, alpha=0.7, label=dataset_name)
    
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Average Objects per Episode')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].set_title('Average Sequence Length')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 0].set_xlabel('Schema')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Schema Distribution')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
    print(f"PASS: Saved: dataset_statistics.png")
    plt.close()


def create_summary_dashboard(experiment_dir: Path, output_dir: Path):
    """Create a summary dashboard with key metrics."""
    # Load experiment report
    report_file = experiment_dir / "experiment_report.json"
    if not report_file.exists():
        print("WARNING: No experiment report found")
        return
    
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    # Create dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f"Experiment Dashboard: {report['experiment_name']}", fontsize=16, fontweight='bold')
    
    # Key metrics text
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')
    
    text_content = f"""
    Duration: {report['total_duration_formatted']}
    Total Steps: {report['total_steps']:,}
    Total Epochs: {report['total_epochs']}
    """
    
    if 'training_summary' in report:
        ts = report['training_summary']
        text_content += f"""
    Final Train Loss: {ts['final_train_loss']:.4f}
    Best Train Loss: {ts['best_train_loss']:.4f}
    """
    
    if 'validation_summary' in report:
        vs = report['validation_summary']
        text_content += f"""
    Final Val Loss: {vs['final_val_loss']:.4f}
    Best Val Loss: {vs['best_val_loss']:.4f}
    """
    
    ax_text.text(0.1, 0.5, text_content, fontsize=12, verticalalignment='center', family='monospace')
    
    # Load and plot training loss
    training_csv = experiment_dir / "training_metrics.csv"
    if training_csv.exists():
        df = pd.read_csv(training_csv)
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(df['step'], df['loss'], alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
    
    # Load and plot validation loss
    validation_csv = experiment_dir / "validation_metrics.csv"
    if validation_csv.exists():
        df = pd.read_csv(validation_csv)
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(df['epoch'], df['val_loss'], marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss')
        ax2.grid(True, alpha=0.3)
    
    # System metrics
    system_csv = experiment_dir / "system_metrics.csv"
    if system_csv.exists():
        df = pd.read_csv(system_csv)
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.plot(df['step'], df['memory_mb'], label='Memory', alpha=0.7)
        if 'gpu_memory_mb' in df.columns and df['gpu_memory_mb'].notna().any():
            ax3.plot(df['step'], df['gpu_memory_mb'], label='GPU Memory', alpha=0.7)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_title('Memory Usage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Model parameters pie chart
    if 'model_stats' in report and 'parameters_by_layer' in report['model_stats']:
        ax4 = fig.add_subplot(gs[2, 0])
        params = report['model_stats']['parameters_by_layer']
        ax4.pie(params.values(), labels=params.keys(), autopct='%1.1f%%')
        ax4.set_title('Parameters by Layer')
    
    # Dataset statistics
    if 'dataset_stats' in report:
        ax5 = fig.add_subplot(gs[2, 1])
        datasets = []
        episode_counts = []
        for name, stats in report['dataset_stats'].items():
            datasets.append(name)
            episode_counts.append(stats['total_episodes'])
        ax5.bar(datasets, episode_counts)
        ax5.set_ylabel('Episodes')
        ax5.set_title('Dataset Sizes')
        ax5.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_dir / 'dashboard.png', dpi=300, bbox_inches='tight')
    print(f"PASS: Saved: dashboard.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment metrics')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots (default: experiment_dir/plots)')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment)
    if not experiment_dir.exists():
        print(f"FAIL: Experiment directory not found: {experiment_dir}")
        return
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = experiment_dir / "plots"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Visualizing metrics from: {experiment_dir}")
    print(f"   Output directory: {output_dir}\n")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Generate plots
    plot_training_metrics(experiment_dir, output_dir)
    plot_validation_metrics(experiment_dir, output_dir)
    plot_system_metrics(experiment_dir, output_dir)
    plot_data_loading_metrics(experiment_dir, output_dir)
    plot_dataset_statistics(experiment_dir, output_dir)
    create_summary_dashboard(experiment_dir, output_dir)
    
    print(f"\nPASS: All visualizations saved to: {output_dir}")
    print(f"\nGenerated plots:")
    for plot_file in sorted(output_dir.glob("*.png")):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    main()
