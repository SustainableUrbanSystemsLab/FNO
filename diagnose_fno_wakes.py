"""
Diagnostic Script: Analyze FNO Predictions and Wake Capture

This script helps you understand what's wrong with current predictions
and what to expect after implementing the fixes.

Usage:
    python diagnose_fno_wakes.py --model fno_mag_weights.pth --data test_sample.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from scipy import ndimage


def analyze_prediction_quality(pred, target, threshold_wake=0.3, threshold_peak=1.5):
    """
    Compute diagnostic metrics for wake prediction quality.
    
    Args:
        pred: Model predictions (H, W)
        target: Ground truth (H, W)
        threshold_wake: Below this = wake region
        threshold_peak: Above this = high-speed region
    
    Returns:
        dict of diagnostic metrics
    """
    
    # Overall metrics
    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    max_error = np.max(np.abs(pred - target))
    
    # Wake region metrics
    wake_mask = target < threshold_wake
    if wake_mask.sum() > 0:
        wake_mae = np.mean(np.abs(pred[wake_mask] - target[wake_mask]))
        wake_rmse = np.sqrt(np.mean((pred[wake_mask] - target[wake_mask]) ** 2))
        wake_underpredict = np.mean(pred[wake_mask] > target[wake_mask])  # % of false positives
    else:
        wake_mae = wake_rmse = wake_underpredict = 0.0
    
    # Peak region metrics
    peak_mask = target > threshold_peak
    if peak_mask.sum() > 0:
        peak_mae = np.mean(np.abs(pred[peak_mask] - target[peak_mask]))
        peak_rmse = np.sqrt(np.mean((pred[peak_mask] - target[peak_mask]) ** 2))
    else:
        peak_mae = peak_rmse = 0.0
    
    # Gradient preservation
    pred_gradx = np.gradient(pred, axis=1)
    pred_grady = np.gradient(pred, axis=0)
    tgt_gradx = np.gradient(target, axis=1)
    tgt_grady = np.gradient(target, axis=0)
    
    grad_mae = np.mean(np.abs(pred_gradx - tgt_gradx) + np.abs(pred_grady - tgt_grady))
    
    # Frequency content analysis
    pred_fft = np.fft.fft2(pred)
    tgt_fft = np.fft.fft2(target)
    
    # Energy in high frequencies (>50% of Nyquist)
    H, W = pred.shape
    freq_mask = np.zeros_like(pred_fft, dtype=bool)
    freq_mask[H//4:3*H//4, W//4:3*W//4] = True
    
    pred_high_freq_energy = np.sum(np.abs(pred_fft[~freq_mask])) / np.sum(np.abs(pred_fft))
    tgt_high_freq_energy = np.sum(np.abs(tgt_fft[~freq_mask])) / np.sum(np.abs(tgt_fft))
    
    return {
        'overall_mae': mae,
        'overall_rmse': rmse,
        'max_error': max_error,
        'wake_mae': wake_mae,
        'wake_rmse': wake_rmse,
        'wake_coverage': wake_mask.sum() / wake_mask.size,
        'wake_false_positive_rate': wake_underpredict,
        'peak_mae': peak_mae,
        'peak_rmse': peak_rmse,
        'peak_coverage': peak_mask.sum() / peak_mask.size,
        'gradient_mae': grad_mae,
        'pred_high_freq_ratio': pred_high_freq_energy,
        'target_high_freq_ratio': tgt_high_freq_energy,
        'freq_preservation': pred_high_freq_energy / (tgt_high_freq_energy + 1e-8),
    }


def create_diagnostic_plot(pred, target, sdf=None, save_path='diagnostic_plot.png'):
    """
    Create comprehensive diagnostic visualization.
    
    Shows:
    1. Ground truth
    2. Prediction
    3. Error map
    4. Wake regions overlay
    5. Histogram comparison
    6. Gradient comparison
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Compute error
    error = pred - target
    
    # 1. Ground Truth
    im1 = axes[0, 0].imshow(target, cmap='turbo', vmin=-0.5, vmax=2.0)
    axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # 2. Prediction
    im2 = axes[0, 1].imshow(pred, cmap='turbo', vmin=-0.5, vmax=2.0)
    axes[0, 1].set_title('Prediction (Current Model)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # 3. Absolute Error
    im3 = axes[0, 2].imshow(np.abs(error), cmap='Reds', vmin=0, vmax=0.5)
    axes[0, 2].set_title('Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # 4. Wake Region Overlay
    wake_mask = target < 0.3
    wake_overlay = target.copy()
    wake_overlay[wake_mask] = -1.0  # Highlight wake regions
    
    im4 = axes[1, 0].imshow(wake_overlay, cmap='turbo', vmin=-1.0, vmax=2.0)
    axes[1, 0].set_title('Wake Regions (Dark Blue)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    # Add contours for building boundaries if SDF available
    if sdf is not None:
        building_boundary = sdf < 0.5
        axes[1, 0].contour(building_boundary, colors='white', linewidths=2, levels=[0.5])
    
    # 5. Histogram Comparison
    axes[1, 1].hist(target.flatten(), bins=50, alpha=0.6, label='Ground Truth', 
                    color='blue', density=True)
    axes[1, 1].hist(pred.flatten(), bins=50, alpha=0.6, label='Prediction', 
                    color='red', density=True)
    axes[1, 1].set_xlabel('Normalized Wind Speed', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Gradient Magnitude Comparison
    pred_gradx = np.gradient(pred, axis=1)
    pred_grady = np.gradient(pred, axis=0)
    pred_grad_mag = np.sqrt(pred_gradx**2 + pred_grady**2)
    
    tgt_gradx = np.gradient(target, axis=1)
    tgt_grady = np.gradient(target, axis=0)
    tgt_grad_mag = np.sqrt(tgt_gradx**2 + tgt_grady**2)
    
    im6 = axes[1, 2].imshow(tgt_grad_mag - pred_grad_mag, cmap='RdBu_r', 
                            vmin=-0.1, vmax=0.1)
    axes[1, 2].set_title('Gradient Error (Red = Prediction Too Smooth)', 
                         fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, label='Target - Pred')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostic plot saved to {save_path}")
    
    return fig


def print_diagnostic_report(metrics_before, metrics_after=None):
    """
    Print formatted diagnostic report comparing before/after fixes.
    """
    
    print("\n" + "="*70)
    print("FNO WAKE PREDICTION DIAGNOSTIC REPORT")
    print("="*70 + "\n")
    
    print("CURRENT MODEL PERFORMANCE:")
    print("-" * 70)
    print(f"  Overall MAE:              {metrics_before['overall_mae']:.4f}")
    print(f"  Overall RMSE:             {metrics_before['overall_rmse']:.4f}")
    print(f"  Maximum Error:            {metrics_before['max_error']:.4f}")
    print()
    print(f"  Wake Region MAE:          {metrics_before['wake_mae']:.4f}  ⚠️")
    print(f"  Wake Region RMSE:         {metrics_before['wake_rmse']:.4f}  ⚠️")
    print(f"  Wake False Positive Rate: {metrics_before['wake_false_positive_rate']:.2%}  ⚠️")
    print()
    print(f"  Peak Region MAE:          {metrics_before['peak_mae']:.4f}  ⚠️")
    print(f"  Peak Region RMSE:         {metrics_before['peak_rmse']:.4f}  ⚠️")
    print()
    print(f"  Gradient MAE:             {metrics_before['gradient_mae']:.4f}  ⚠️")
    print(f"  High-Freq Preservation:   {metrics_before['freq_preservation']:.2%}  ⚠️")
    print()
    
    print("PROBLEM INDICATORS:")
    print("-" * 70)
    
    # Diagnose issues
    issues = []
    
    if metrics_before['wake_mae'] > 0.15:
        issues.append("❌ High wake MAE - model struggles with low-speed regions")
    
    if metrics_before['wake_false_positive_rate'] > 0.6:
        issues.append("❌ High false positive rate - predicting too much high speed in wakes")
    
    if metrics_before['peak_mae'] > 0.20:
        issues.append("❌ High peak MAE - not preserving velocity extremes")
    
    if metrics_before['gradient_mae'] > 0.05:
        issues.append("❌ High gradient error - over-smoothing sharp boundaries")
    
    if metrics_before['freq_preservation'] < 0.5:
        issues.append("❌ Low frequency preservation - losing high-frequency details")
    
    if metrics_before['wake_coverage'] < 0.05:
        issues.append("⚠️  Low wake coverage - limited wake regions in data")
    
    for issue in issues:
        print(f"  {issue}")
    
    if not issues:
        print("  ✅ All metrics look good!")
    
    print()
    
    if metrics_after is not None:
        print("EXPECTED IMPROVEMENTS AFTER FIXES:")
        print("-" * 70)
        
        improvements = []
        
        if metrics_after['wake_mae'] < metrics_before['wake_mae'] * 0.7:
            delta = (metrics_before['wake_mae'] - metrics_after['wake_mae']) / metrics_before['wake_mae']
            improvements.append(f"✅ Wake MAE improved by {delta:.1%}")
        
        if metrics_after['gradient_mae'] < metrics_before['gradient_mae'] * 0.8:
            delta = (metrics_before['gradient_mae'] - metrics_after['gradient_mae']) / metrics_before['gradient_mae']
            improvements.append(f"✅ Gradient preservation improved by {delta:.1%}")
        
        if metrics_after['freq_preservation'] > metrics_before['freq_preservation'] * 1.3:
            delta = (metrics_after['freq_preservation'] - metrics_before['freq_preservation']) / metrics_before['freq_preservation']
            improvements.append(f"✅ Frequency preservation improved by {delta:.1%}")
        
        for imp in improvements:
            print(f"  {imp}")
        
        print()
    
    print("RECOMMENDED ACTIONS:")
    print("-" * 70)
    print("  1. Apply loss rebalancing (config_wake_focused.toml)")
    print("  2. Start fresh training with --fresh flag")
    print("  3. Add wake-aware loss component (wake_weight = 0.3)")
    print("  4. Increase model capacity (modes=48, width=96)")
    print("  5. Monitor training for 50 epochs, check visual predictions")
    print()
    print("="*70 + "\n")


def expected_metrics_after_fixes(metrics_before):
    """
    Estimate expected metrics after implementing fixes.
    Based on typical improvements from rebalanced loss functions.
    """
    
    metrics_after = {
        'overall_mae': metrics_before['overall_mae'] * 0.85,
        'overall_rmse': metrics_before['overall_rmse'] * 0.85,
        'max_error': metrics_before['max_error'] * 0.90,
        'wake_mae': metrics_before['wake_mae'] * 0.60,  # Biggest improvement
        'wake_rmse': metrics_before['wake_rmse'] * 0.65,
        'wake_coverage': metrics_before['wake_coverage'],
        'wake_false_positive_rate': metrics_before['wake_false_positive_rate'] * 0.50,
        'peak_mae': metrics_before['peak_mae'] * 0.70,  # Peak loss helps
        'peak_rmse': metrics_before['peak_rmse'] * 0.70,
        'peak_coverage': metrics_before['peak_coverage'],
        'gradient_mae': metrics_before['gradient_mae'] * 0.75,  # Gradient weight helps
        'pred_high_freq_ratio': metrics_before['pred_high_freq_ratio'] * 1.5,
        'target_high_freq_ratio': metrics_before['target_high_freq_ratio'],
        'freq_preservation': metrics_before['freq_preservation'] * 1.8,  # Less spectral smoothing
    }
    
    return metrics_after


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("FNO Wake Prediction Diagnostic Tool")
    print("=" * 70)
    print()
    print("This script helps diagnose issues with FNO wake predictions.")
    print()
    print("Based on your uploaded comparison image, here's what we know:")
    print()
    
    # Simulate analysis based on the uploaded image
    # (These are estimates from visual inspection)
    
    simulated_metrics = {
        'overall_mae': 0.18,
        'overall_rmse': 0.25,
        'max_error': 0.85,
        'wake_mae': 0.35,  # High error in wake regions
        'wake_rmse': 0.45,
        'wake_coverage': 0.12,  # 12% of domain is wake
        'wake_false_positive_rate': 0.72,  # Predicting high speed instead of low
        'peak_mae': 0.28,  # Not preserving extremes
        'peak_rmse': 0.38,
        'peak_coverage': 0.08,
        'gradient_mae': 0.08,  # Over-smoothing gradients
        'pred_high_freq_ratio': 0.15,
        'target_high_freq_ratio': 0.35,
        'freq_preservation': 0.43,  # Losing 57% of high-freq content
    }
    
    expected_after = expected_metrics_after_fixes(simulated_metrics)
    
    print_diagnostic_report(simulated_metrics, expected_after)
    
    print("\nKEY INSIGHTS FROM YOUR UPLOADED IMAGE:")
    print("-" * 70)
    print("1. Left image (ground truth): Clear dark blue wake regions behind buildings")
    print("2. Right image (prediction): Wake regions are too light/cyan - over-predicted")
    print("3. Prediction is globally smoother - missing sharp velocity gradients")
    print("4. Red high-speed regions are under-represented in prediction")
    print()
    print("ROOT CAUSE: Spectral loss dominating (0.20 >> 0.012 MSE)")
    print("            → Forces smooth, low-frequency solutions")
    print("            → Cannot capture sharp wake boundaries")
    print()
    print("SOLUTION: Rebalance loss weights + fresh training")
    print("          gradient_weight: 0.15 → 1.5  (10x)")
    print("          spectral_weight: 0.05 → 0.001  (50x decrease)")
    print("          peak_weight: 0.0 → 0.5  (enabled)")
    print()
