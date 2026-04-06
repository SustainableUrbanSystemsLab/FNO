#!/usr/bin/env python3
"""
Plot Comparison of Training Metrics across All FNO Models
=========================================================
Searches 'training_logs/' for the most recent runs of Standard, 
Hybrid, PINN, and Geo models and plots their loss curves side-by-side.

Usage:
  python tools/plot_comparison_curves.py
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Publication style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (16, 8),
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def find_latest_run_for_type(model_type):
    """
    Search training_logs/ for the newest run containing a specific config string.
    model_type: 'standard', 'hybrid', 'pinn', 'geo'
    """
    pattern = os.path.join("training_logs", "*", "config.json")
    config_files = glob.glob(pattern)
    
    matches = []
    for cfg_p in config_files:
        try:
            with open(cfg_p, 'r') as f:
                import json
                cfg = json.load(f)
                # Check for model type indicators in config
                # Standard FNO usually doesn't have a special type tag, but Hybrid/PINN/Geo do.
                # However, the folder name or the checkpoint filename in the log might help.
                # We'll check the parent folder's metrics file
                metrics_p = os.path.join(os.path.dirname(cfg_p), "training_metrics.csv")
                if not os.path.exists(metrics_p): continue
                
                # We identify via unique keys or values in config if possible
                # If no clear key, we look at the experiment name/folder
                folder_name = os.path.basename(os.path.dirname(cfg_p)).lower()
                
                # Check metrics for loss component existence as a fallback
                df_peek = pd.read_csv(metrics_p, nrows=5)
                
                is_pinn = 'continuity_loss' in df_peek.columns or 'pinn' in folder_name
                is_hybrid = ('hybrid' in folder_name) or ('u_net' in str(cfg).lower())
                is_geo = 'geo' in folder_name
                
                if model_type == 'pinn' and is_pinn: matches.append(metrics_p)
                elif model_type == 'hybrid' and is_hybrid and not is_pinn: matches.append(metrics_p)
                elif model_type == 'geo' and is_geo: matches.append(metrics_p)
                elif model_type == 'standard' and not (is_pinn or is_hybrid or is_geo): matches.append(metrics_p)
        except: continue
        
    if not matches: return None
    # Return file with latest modification time
    return max(matches, key=os.path.getmtime)

def main():
    model_types = ['standard', 'hybrid', 'pinn', 'geo']
    data_map = {}
    
    print("Searching for latest training logs...")
    for t in model_types:
        path = find_latest_run_for_type(t)
        if path:
            print(f"  [Found] {t.upper()}: {path}")
            data_map[t] = pd.read_csv(path)
        else:
            print(f"  [Missing] {t.upper()} (No logs found)")
            
    if not data_map:
        print("Error: No training logs found. Run training scripts first.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    axes = axes.flatten()
    
    colors = {'standard': '#1f77b4', 'hybrid': '#ff7f0e', 'pinn': '#2ca02c', 'geo': '#d62728'}
    
    for i, t in enumerate(model_types):
        ax = axes[i]
        if t in data_map:
            df = data_map[t]
            ax.semilogy(df['epoch'], df['total_loss'], color=colors[t], linewidth=2, label=f'{t.upper()} Total Loss')
            
            # If sub-components exist (MSE, Grad, etc), plot them with alpha
            for col in ['mse_loss', 'gradient_loss', 'continuity_loss', 'wake_loss']:
                if col in df.columns and df[col].sum() > 0:
                    ax.semilogy(df['epoch'], df[col], alpha=0.4, linestyle='--', label=col.replace('_', ' ').title())
            
            ax.set_title(f"{t.upper()} Training History", fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (log)")
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, f"No Data Found for {t.upper()}", ha='center', va='center')
            ax.axis('off')

    plt.suptitle(f"Cross-Architecture Training Performance Comparison\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=16)
    
    out_path = "model_training_comparison.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"\nSuccess! Comparison curves saved to: {out_path}")

if __name__ == "__main__":
    main()
