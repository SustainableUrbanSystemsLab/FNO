# Training Logger for FNO Publication Metrics
# Saves comprehensive training data for academic papers

import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class TrainingLogger:
    """Comprehensive training logger for publication-quality metrics.
    
    Saves:
    - Per-epoch metrics as CSV (for plotting loss curves)
    - Hyperparameters and configuration as JSON
    - Final training summary with model statistics
    """
    
    def __init__(self, output_dir: str = "training_logs", experiment_name: Optional[str] = None):
        """Initialize the training logger.
        
        Args:
            output_dir: Directory to save logs
            experiment_name: Optional name for experiment (default: timestamp)
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(output_dir, self.experiment_name)
        
        # Create output directory
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # File paths
        self.metrics_csv = os.path.join(self.experiment_dir, "training_metrics.csv")
        self.config_json = os.path.join(self.experiment_dir, "config.json")
        self.summary_json = os.path.join(self.experiment_dir, "summary.json")
        
        # Tracking
        self.start_time = None
        self.epoch_metrics = []
        self.config = {}
        self.csv_writer = None
        self.csv_file = None
        
    def start_training(self, config: Dict[str, Any], model: torch.nn.Module = None):
        """Log training start with configuration.
        
        Args:
            config: Dictionary of hyperparameters and settings
            model: Optional PyTorch model for parameter counting
        """
        self.start_time = time.time()
        self.config = config.copy()
        
        # Add model info if provided
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.config['model_info'] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'parameter_count_millions': round(total_params / 1e6, 2)
            }
        
        # Add metadata
        self.config['training_started'] = datetime.now().isoformat()
        self.config['experiment_name'] = self.experiment_name
        
        # Save config
        with open(self.config_json, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Initialize CSV
        self._init_csv()
        
        print(f"[Logger] Training logs will be saved to: {self.experiment_dir}")
        
    def _init_csv(self):
        """Initialize CSV file with headers."""
        self.csv_file = open(self.metrics_csv, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.fieldnames = [
            'epoch', 'total_loss', 'mse_loss', 'gradient_loss', 'spectral_loss', 'peak_loss', 'wake_loss',
            'learning_rate', 'epoch_time_sec', 'best_loss', 'patience_counter'
        ]
        self.csv_writer.writerow(self.fieldnames)
        self.csv_file.flush()
        
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for a single epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary with keys like 'total_loss', 'mse_loss', 'gradient_loss', 
                    'spectral_loss', 'peak_loss', 'learning_rate', 'epoch_time', 'best_loss', 'patience'
        """
        # Lazily initialize CSV if start_training() was never called
        if not hasattr(self, 'fieldnames') or self.csv_writer is None:
            if self.start_time is None:
                self.start_time = time.time()
            self._init_csv()

        # Prepare row
        row = [int(epoch)]
        for k in self.fieldnames[1:]:
            val = metrics.get(k, 0.0)
            if isinstance(val, (int, float)):
                val = round(val, 6)
            row.append(val)
            
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        self.epoch_metrics.append(metrics)

        
    def finish_training(self, final_metrics: Dict[str, Any] = None):
        """Finalize logging and save summary.
        
        Args:
            final_metrics: Optional dictionary with final evaluation metrics
        """
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Build summary
        summary = {
            'experiment_name': self.experiment_name,
            'training_completed': datetime.now().isoformat(),
            'total_training_time_seconds': round(total_time, 2),
            'total_training_time_hours': round(total_time / 3600, 2),
            'total_epochs': len(self.epoch_metrics),
        }
        
        if self.epoch_metrics:
            losses = [m.get('total_loss', float('inf')) for m in self.epoch_metrics]
            summary['final_loss'] = losses[-1] if losses else None
            summary['best_loss'] = min(losses) if losses else None
            summary['best_epoch'] = losses.index(min(losses)) + 1 if losses else None
            
        if final_metrics:
            summary['final_metrics'] = final_metrics
            
        # Include config summary
        summary['config'] = {
            'batch_size': self.config.get('batch_size'),
            'learning_rate': self.config.get('learning_rate'),
            'epochs': self.config.get('epochs'),
            'modes': (self.config.get('modes1'), self.config.get('modes2')),
            'width': self.config.get('width'),
            'n_layers': self.config.get('n_layers'),
            'gradient_weight': self.config.get('gradient_weight'),
            'spectral_weight': self.config.get('spectral_weight'),
        }
        
        if 'model_info' in self.config:
            summary['model_info'] = self.config['model_info']
        
        # Save summary
        with open(self.summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Close CSV
        if self.csv_file:
            self.csv_file.close()
            
        print(f"\n[Logger] Training complete!")
        print(f"[Logger] Logs saved to: {self.experiment_dir}")
        print(f"[Logger] - Config:  {os.path.basename(self.config_json)}")
        print(f"[Logger] - Metrics: {os.path.basename(self.metrics_csv)}")
        print(f"[Logger] - Summary: {os.path.basename(self.summary_json)}")
        
        return summary


def create_publication_plots(metrics_csv: str, output_dir: str = None):
    """Generate publication-ready plots from training metrics.
    
    Requires matplotlib. Call this after training completes.
    
    Args:
        metrics_csv: Path to training_metrics.csv
        output_dir: Directory to save plots (default: same as CSV)
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("Install matplotlib and pandas for plotting: pip install matplotlib pandas")
        return
    
    df = pd.read_csv(metrics_csv)
    output_dir = output_dir or os.path.dirname(metrics_csv)
    
    # Publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
    })
    
    # Plot 1: Total Loss Curve
    fig, ax = plt.subplots()
    ax.semilogy(df['epoch'], df['total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss Curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Loss Components
    fig, ax = plt.subplots()
    if 'mse_loss' in df.columns and df['mse_loss'].sum() > 0:
        ax.semilogy(df['epoch'], df['mse_loss'], label='MSE Loss')
    if 'gradient_loss' in df.columns and df['gradient_loss'].sum() > 0:
        ax.semilogy(df['epoch'], df['gradient_loss'], label='Gradient Loss')
    if 'spectral_loss' in df.columns and df['spectral_loss'].sum() > 0:
        ax.semilogy(df['epoch'], df['spectral_loss'], label='Spectral Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Loss Components')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_components.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Logger] Plots saved to: {output_dir}")
