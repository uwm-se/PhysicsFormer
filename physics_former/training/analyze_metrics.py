"""
Advanced Metrics Analysis Tool

Implements the strategies from METRICS_ANALYSIS_STRATEGIES.md
Provides automated analysis of training metrics for unique insights.

Usage:
    python analyze_metrics.py --experiment logs/cls_pipeline_aggressive_bs64/20231124_143022
    python analyze_metrics.py --compare logs/cls_experiment logs/ablation_experiment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.serialization import save_json
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class MetricsAnalyzer:
    """Advanced metrics analysis for training experiments."""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.df = self.load_metrics()
        
    def load_metrics(self) -> pd.DataFrame:
        """Load training metrics from CSV."""
        csv_path = self.experiment_dir / 'training_metrics.csv'
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def analyze_forgetting(self) -> Dict:
        """Detect catastrophic forgetting patterns."""
        print("\n" + "="*70)
        print("CATASTROPHIC FORGETTING ANALYSIS")
        print("="*70)
        
        if 'task' not in self.df.columns:
            print("WARNING: No task information available")
            return {}
        
        forgetting_events = []
        task_performance = {}
        
        for task in self.df['task'].unique():
            task_df = self.df[self.df['task'] == task].copy()
            
            if 'val_accuracy' in task_df.columns:
                peak_acc = task_df['val_accuracy'].max()
                final_acc = task_df['val_accuracy'].iloc[-1]
                
                forgetting = (peak_acc - final_acc) / peak_acc if peak_acc > 0 else 0
                
                task_performance[task] = {
                    'peak_accuracy': peak_acc,
                    'final_accuracy': final_acc,
                    'forgetting_rate': forgetting
                }
                
                if forgetting > 0.1:  # >10% drop
                    forgetting_events.append({
                        'task': task,
                        'severity': forgetting,
                        'peak': peak_acc,
                        'final': final_acc
                    })
        
        # Print results
        print(f"\nForgetting Events Detected: {len(forgetting_events)}")
        for event in forgetting_events:
            print(f"  WARNING: {event['task']}: {event['severity']:.1%} drop "
                  f"({event['peak']:.2%} -> {event['final']:.2%})")
        
        if not forgetting_events:
            print("  PASS: No significant forgetting detected")
        
        return {
            'events': forgetting_events,
            'task_performance': task_performance
        }
    
    def analyze_learning_efficiency(self) -> Dict:
        """Analyze sample efficiency and convergence speed."""
        print("\n" + "="*70)
        print("LEARNING EFFICIENCY ANALYSIS")
        print("="*70)
        
        if 'val_accuracy' not in self.df.columns:
            print("WARNING: No validation accuracy available")
            return {}
        
        # Convergence analysis
        target_accuracy = 0.90
        converged = self.df[self.df['val_accuracy'] >= target_accuracy]
        
        if not converged.empty:
            convergence_step = converged.iloc[0]['step']
            convergence_epoch = converged.iloc[0]['epoch']
            print(f"\nConvergence to {target_accuracy:.0%}:")
            print(f"  Step: {convergence_step:,}")
            print(f"  Epoch: {convergence_epoch}")
        else:
            print(f"\nWARNING: Did not reach {target_accuracy:.0%} accuracy")
            convergence_step = None
        
        # Learning rate
        if len(self.df) > 1:
            accuracy_gain = self.df['val_accuracy'].diff().mean()
            print(f"\nAverage accuracy gain per step: {accuracy_gain:.4f}")
        
        # Plateau detection
        window = 50
        if len(self.df) > window:
            recent_improvement = (self.df['val_accuracy'].iloc[-1] - 
                                self.df['val_accuracy'].iloc[-window])
            
            if abs(recent_improvement) < 0.01:
                print(f"\nWARNING: Plateau detected (no improvement in last {window} steps)")
            else:
                print(f"\nPASS: Still improving ({recent_improvement:+.2%} in last {window} steps)")
        
        return {
            'convergence_step': convergence_step,
            'target_accuracy': target_accuracy,
            'final_accuracy': self.df['val_accuracy'].iloc[-1]
        }
    
    def analyze_gradient_health(self) -> Dict:
        """Analyze gradient dynamics and training stability."""
        print("\n" + "="*70)
        print("GRADIENT HEALTH ANALYSIS")
        print("="*70)
        
        if 'grad_norm' not in self.df.columns:
            print("WARNING: No gradient norm data available")
            return {}
        
        grad_norms = self.df['grad_norm'].dropna()
        
        # Statistics
        print(f"\nGradient Norm Statistics:")
        print(f"  Mean: {grad_norms.mean():.4f}")
        print(f"  Std: {grad_norms.std():.4f}")
        print(f"  Max: {grad_norms.max():.4f}")
        print(f"  Min: {grad_norms.min():.4f}")
        
        # Pathology detection
        explosion_rate = (grad_norms > 10.0).sum() / len(grad_norms)
        vanishing_rate = (grad_norms < 0.01).sum() / len(grad_norms)
        
        print(f"\nGradient Pathologies:")
        print(f"  Explosion rate: {explosion_rate:.1%}")
        print(f"  Vanishing rate: {vanishing_rate:.1%}")
        
        if explosion_rate > 0.01:
            print("  WARNING: Gradient explosion detected!")
        if vanishing_rate > 0.1:
            print("  WARNING: Gradient vanishing detected!")
        if explosion_rate < 0.01 and vanishing_rate < 0.1:
            print("  PASS: Gradients healthy")
        
        return {
            'mean_norm': grad_norms.mean(),
            'explosion_rate': explosion_rate,
            'vanishing_rate': vanishing_rate
        }
    
    def analyze_resource_utilization(self) -> Dict:
        """Analyze computational efficiency."""
        print("\n" + "="*70)
        print("RESOURCE UTILIZATION ANALYSIS")
        print("="*70)
        
        metrics = {}
        
        # GPU utilization
        if 'gpu_utilization' in self.df.columns:
            gpu_util = self.df['gpu_utilization'].mean()
            print(f"\nGPU Utilization: {gpu_util:.1f}%")
            
            if gpu_util < 70:
                print("  WARNING: Low GPU utilization - possible CPU bottleneck")
            else:
                print("  PASS: Good GPU utilization")
            
            metrics['gpu_utilization'] = gpu_util
        
        # Memory usage
        if 'gpu_memory_used' in self.df.columns:
            memory_used = self.df['gpu_memory_used'].mean()
            print(f"\nGPU Memory: {memory_used:.2f} GB")
            metrics['gpu_memory'] = memory_used
        
        # Throughput
        if 'batches_per_sec' in self.df.columns:
            throughput = self.df['batches_per_sec'].mean()
            throughput_std = self.df['batches_per_sec'].std()
            
            print(f"\nThroughput: {throughput:.2f} ± {throughput_std:.2f} batches/sec")
            
            if throughput_std / throughput > 0.2:
                print("  WARNING: High throughput variance - unstable data loading")
            else:
                print("  PASS: Stable throughput")
            
            metrics['throughput'] = throughput
        
        return metrics
    
    def detect_phase_transitions(self) -> List[Dict]:
        """Detect critical learning moments."""
        print("\n" + "="*70)
        print("PHASE TRANSITION DETECTION")
        print("="*70)
        
        if 'loss' not in self.df.columns or len(self.df) < 100:
            print("WARNING: Insufficient data for phase detection")
            return []
        
        # Calculate loss velocity
        self.df['loss_velocity'] = self.df['loss'].diff()
        
        # Detect breakthroughs (sudden loss drops)
        window = 50
        breakthroughs = []
        
        for i in range(window, len(self.df)):
            recent_loss = self.df['loss'].iloc[i-window:i].mean()
            current_loss = self.df['loss'].iloc[i]
            
            if (recent_loss - current_loss) > 0.1 * recent_loss:
                breakthroughs.append({
                    'step': self.df['step'].iloc[i],
                    'epoch': self.df['epoch'].iloc[i],
                    'loss_drop': recent_loss - current_loss
                })
        
        print(f"\nBreakthroughs Detected: {len(breakthroughs)}")
        for bt in breakthroughs[:5]:  # Show first 5
            print(f"  Step {bt['step']}: Loss dropped by {bt['loss_drop']:.4f}")
        
        return breakthroughs
    
    def generate_summary_report(self, output_path: str = None):
        """Generate comprehensive analysis report."""
        print("\n" + "="*70)
        print("GENERATING SUMMARY REPORT")
        print("="*70)
        
        report = {
            'experiment': str(self.experiment_dir),
            'total_steps': len(self.df),
            'forgetting_analysis': self.analyze_forgetting(),
            'efficiency_analysis': self.analyze_learning_efficiency(),
            'gradient_health': self.analyze_gradient_health(),
            'resource_utilization': self.analyze_resource_utilization(),
            'phase_transitions': self.detect_phase_transitions()
        }
        
        # Save report
        if output_path:
            save_json(report, output_path)
            print(f"\nSUCCESS: Report saved to: {output_path}")
        
        return report
    
    def plot_comprehensive_analysis(self, save_path: str = None):
        """Create comprehensive visualization."""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Loss and accuracy
        ax1 = plt.subplot(3, 2, 1)
        if 'loss' in self.df.columns:
            ax1.plot(self.df['step'], self.df['loss'], label='Loss', alpha=0.7)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Validation accuracy
        ax2 = plt.subplot(3, 2, 2)
        if 'val_accuracy' in self.df.columns:
            ax2.plot(self.df['step'], self.df['val_accuracy'], label='Val Accuracy', color='green')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Gradient norm
        ax3 = plt.subplot(3, 2, 3)
        if 'grad_norm' in self.df.columns:
            ax3.plot(self.df['step'], self.df['grad_norm'], alpha=0.5)
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Gradient Norm')
            ax3.set_title('Gradient Dynamics')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # 4. Learning rate
        ax4 = plt.subplot(3, 2, 4)
        if 'learning_rate' in self.df.columns:
            ax4.plot(self.df['step'], self.df['learning_rate'], color='orange')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        
        # 5. GPU utilization
        ax5 = plt.subplot(3, 2, 5)
        if 'gpu_utilization' in self.df.columns:
            ax5.plot(self.df['step'], self.df['gpu_utilization'], color='purple')
            ax5.set_xlabel('Step')
            ax5.set_ylabel('GPU Utilization (%)')
            ax5.set_title('GPU Utilization')
            ax5.grid(True, alpha=0.3)
        
        # 6. Throughput
        ax6 = plt.subplot(3, 2, 6)
        if 'batches_per_sec' in self.df.columns:
            ax6.plot(self.df['step'], self.df['batches_per_sec'], color='brown')
            ax6.set_xlabel('Step')
            ax6.set_ylabel('Batches/sec')
            ax6.set_title('Training Throughput')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SUCCESS: Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


def compare_experiments(exp1_dir: str, exp2_dir: str, output_dir: str = None):
    """Compare two experiments (e.g., CLS vs Ablation)."""
    print("\n" + "="*70)
    print("COMPARING EXPERIMENTS")
    print("="*70)
    print(f"Experiment 1: {exp1_dir}")
    print(f"Experiment 2: {exp2_dir}")
    print("="*70)
    
    # Load both experiments
    analyzer1 = MetricsAnalyzer(exp1_dir)
    analyzer2 = MetricsAnalyzer(exp2_dir)
    
    # Compare final performance
    print("\nFinal Performance:")
    if 'val_accuracy' in analyzer1.df.columns:
        acc1 = analyzer1.df['val_accuracy'].iloc[-1]
        acc2 = analyzer2.df['val_accuracy'].iloc[-1]
        print(f"  Experiment 1: {acc1:.2%}")
        print(f"  Experiment 2: {acc2:.2%}")
        print(f"  Difference: {(acc1 - acc2):.2%}")
    
    # Compare forgetting
    forgetting1 = analyzer1.analyze_forgetting()
    forgetting2 = analyzer2.analyze_forgetting()
    
    print(f"\nForgetting Events:")
    print(f"  Experiment 1: {len(forgetting1.get('events', []))}")
    print(f"  Experiment 2: {len(forgetting2.get('events', []))}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy comparison
    if 'val_accuracy' in analyzer1.df.columns:
        axes[0, 0].plot(analyzer1.df['step'], analyzer1.df['val_accuracy'], label='Exp 1')
        axes[0, 0].plot(analyzer2.df['step'], analyzer2.df['val_accuracy'], label='Exp 2')
        axes[0, 0].set_title('Validation Accuracy Comparison')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Loss comparison
    if 'loss' in analyzer1.df.columns:
        axes[0, 1].plot(analyzer1.df['step'], analyzer1.df['loss'], label='Exp 1', alpha=0.7)
        axes[0, 1].plot(analyzer2.df['step'], analyzer2.df['loss'], label='Exp 2', alpha=0.7)
        axes[0, 1].set_title('Training Loss Comparison')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Gradient norm comparison
    if 'grad_norm' in analyzer1.df.columns:
        axes[1, 0].plot(analyzer1.df['step'], analyzer1.df['grad_norm'], label='Exp 1', alpha=0.5)
        axes[1, 0].plot(analyzer2.df['step'], analyzer2.df['grad_norm'], label='Exp 2', alpha=0.5)
        axes[1, 0].set_title('Gradient Norm Comparison')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Performance distribution
    if 'val_accuracy' in analyzer1.df.columns:
        axes[1, 1].boxplot([analyzer1.df['val_accuracy'], analyzer2.df['val_accuracy']], 
                          labels=['Exp 1', 'Exp 2'])
        axes[1, 1].set_title('Accuracy Distribution')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'experiment_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSUCCESS: Comparison saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze training metrics')
    
    parser.add_argument('--experiment', type=str,
                       help='Path to experiment directory')
    parser.add_argument('--compare', nargs=2, metavar=('EXP1', 'EXP2'),
                       help='Compare two experiments')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare mode
        compare_experiments(args.compare[0], args.compare[1], args.output)
    elif args.experiment:
        # Single experiment analysis
        analyzer = MetricsAnalyzer(args.experiment)
        
        # Run all analyses
        analyzer.generate_summary_report(
            output_path=f"{args.output}/analysis_report.json" if args.output else None
        )
        
        # Generate plots
        analyzer.plot_comprehensive_analysis(
            save_path=f"{args.output}/analysis_plots.png" if args.output else None
        )
    else:
        print("Error: Must specify either --experiment or --compare")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
