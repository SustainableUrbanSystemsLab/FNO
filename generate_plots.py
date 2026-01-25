#!/usr/bin/env python3
"""
Generate publication-ready plots from FNO training metrics.
Usage: python generate_plots.py [path_to_metrics_csv]
       python generate_plots.py  # Auto-finds latest training run
"""

import os
import sys
import glob


def find_latest_metrics():
    """Find the most recent training_metrics.csv file."""
    pattern = os.path.join("training_logs", "*", "training_metrics.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time, get newest
    return max(files, key=os.path.getmtime)


def generate_publication_plots(metrics_csv: str, output_dir: str = None):
    """Generate publication-ready plots from training metrics.
    
    Args:
        metrics_csv: Path to training_metrics.csv
        output_dir: Directory to save plots (default: same as CSV)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for servers
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as e:
        print(f"Error: Missing required package. Install with:")
        print(f"  pip install matplotlib pandas")
        return False
    
    if not os.path.exists(metrics_csv):
        print(f"Error: File not found: {metrics_csv}")
        return False
    
    df = pd.read_csv(metrics_csv)
    output_dir = output_dir or os.path.dirname(metrics_csv)
    
    print(f"[Plots] Reading metrics from: {metrics_csv}")
    print(f"[Plots] Output directory: {output_dir}")
    print(f"[Plots] Epochs found: {len(df)}")
    
    # Publication style settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Plot 1: Training Loss Curve (Main figure for paper)
    fig, ax = plt.subplots()
    ax.semilogy(df['epoch'], df['total_loss'], 'b-', linewidth=2, label='Training Loss')
    
    # Mark best epoch if available
    if 'best_loss' in df.columns:
        best_idx = df['total_loss'].idxmin()
        best_epoch = df.loc[best_idx, 'epoch']
        best_loss = df.loc[best_idx, 'total_loss']
        ax.scatter([best_epoch], [best_loss], c='red', s=100, zorder=5, 
                   label=f'Best (Epoch {best_epoch})')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('FNO Training Loss')
    ax.legend(loc='upper right')
    fig.tight_layout()
    
    loss_curve_path = os.path.join(output_dir, 'loss_curve.png')
    fig.savefig(loss_curve_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Plots] Saved: {loss_curve_path}")
    
    # Plot 2: Learning Rate Schedule
    if 'learning_rate' in df.columns and df['learning_rate'].std() > 0:
        fig, ax = plt.subplots()
        ax.plot(df['epoch'], df['learning_rate'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        fig.tight_layout()
        
        lr_path = os.path.join(output_dir, 'learning_rate.png')
        fig.savefig(lr_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[Plots] Saved: {lr_path}")
    
    # Plot 3: Loss Components (if available)
    has_components = any(col in df.columns and df[col].sum() > 0 
                         for col in ['mse_loss', 'gradient_loss', 'spectral_loss'])
    
    if has_components:
        fig, ax = plt.subplots()
        colors = {'mse_loss': 'blue', 'gradient_loss': 'orange', 'spectral_loss': 'green'}
        labels = {'mse_loss': 'MSE Loss', 'gradient_loss': 'Gradient Loss', 'spectral_loss': 'Spectral Loss'}
        
        for col, color in colors.items():
            if col in df.columns and df[col].sum() > 0:
                ax.semilogy(df['epoch'], df[col], color=color, linewidth=2, label=labels[col])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Loss Components')
        ax.legend(loc='upper right')
        fig.tight_layout()
        
        components_path = os.path.join(output_dir, 'loss_components.png')
        fig.savefig(components_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[Plots] Saved: {components_path}")
    
    # Plot 4: Combined figure for paper (loss + lr on same figure)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: Loss
    ax1.semilogy(df['epoch'], df['total_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('(a) Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Right: Patience / Early stopping
    if 'patience_counter' in df.columns:
        ax2.plot(df['epoch'], df['patience_counter'], 'r-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Patience Counter')
        ax2.set_title('(b) Early Stopping Progress')
        ax2.grid(True, alpha=0.3)
    elif 'learning_rate' in df.columns:
        ax2.plot(df['epoch'], df['learning_rate'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('(b) Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    combined_path = os.path.join(output_dir, 'training_summary.png')
    fig.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Plots] Saved: {combined_path}")
    
    print(f"\n[Plots] All plots saved to: {output_dir}")
    return True


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        metrics_csv = sys.argv[1]
    else:
        print("[Plots] Looking for latest training run...")
        metrics_csv = find_latest_metrics()
        if not metrics_csv:
            print("Error: No training logs found in 'training_logs/' directory.")
            print("Usage: python generate_plots.py [path/to/training_metrics.csv]")
            sys.exit(1)
        print(f"[Plots] Found: {metrics_csv}")
    
    success = generate_publication_plots(metrics_csv)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
