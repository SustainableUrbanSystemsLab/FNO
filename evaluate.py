#!/usr/bin/env python3
"""
Comprehensive FNO Evaluation Suite
==================================
This script comprehensively tests your FNO models on out-of-sample data.
It evaluates model predictions against ground truth (either from CSVs or .npy datasets)
and computes a full suite of physical and statistical metrics:
- MAE, RMSE, MAPE, R², SSIM, Gradient Correlation
- Peak Loss, Wake Loss, Spectral Loss, Gradient Loss

Outputs an aggregated summary JSON report and a CSV with per-sample metrics.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.models.fno2d import FNO2d
from core.models.hybrid import HybridFNO
from core.models.pinn_fno import PINNFNO
from core.models.geo_fno import GeoFNO
from core.utils.gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple
from core.utils.wake_loss_modifications import compute_wake_loss, compute_peak_loss, sensor_weighted_mse
from tools.infer_csv import load_weights, build_model
from data.wind_dataset import WindDataset

def get_metrics_for_sample(y_pred, y_true, mask=None, device='cpu'):
    """Compute comprehensive metrics for a single prediction vs ground truth."""
    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)
        
    y_pred_masked = y_pred[mask]
    y_true_masked = y_true[mask]
    diff = y_pred_masked - y_true_masked
    abs_diff = np.abs(diff)
    
    # Standard regression metrics
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff**2)))
    
    # MAPE
    valid_mape = np.abs(y_true_masked) > 0.1
    if np.any(valid_mape):
        mape = float(np.mean(abs_diff[valid_mape] / np.abs(y_true_masked[valid_mape])) * 100.0)
    else:
        mape = 0.0
        
    # R squared
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_true_masked - np.mean(y_true_masked)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # SSIM
    try:
        from skimage.metrics import structural_similarity
        data_range = max(y_true.max(), y_pred.max()) - min(y_true.min(), y_pred.min())
        ssim_val = float(structural_similarity(y_true, y_pred, data_range=data_range))
    except ImportError:
        ssim_val = 0.0

    # Gradient Correlation
    def _grad_corr(pred, true, m):
        pdx = np.diff(pred, axis=1, prepend=pred[:, :1])
        pdy = np.diff(pred, axis=0, prepend=pred[:1, :])
        tdx = np.diff(true, axis=1, prepend=true[:, :1])
        tdy = np.diff(true, axis=0, prepend=true[:1, :])
        pg = np.concatenate([pdx.flatten(), pdy.flatten()])
        tg = np.concatenate([tdx.flatten(), tdy.flatten()])
        mm = np.concatenate([m.flatten(), m.flatten()])
        pg, tg = pg[mm], tg[mm]
        if np.std(pg) < 1e-8 or np.std(tg) < 1e-8: return 0.0
        return float(np.corrcoef(pg, tg)[0, 1])
    
    grad_corr = _grad_corr(y_pred, y_true, mask)

    # Convert to batches for PyTorch loss functions [B, C, H, W]
    pth_pred = torch.tensor(y_pred, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    pth_true = torch.tensor(y_true, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    pth_mask = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Physical metrics via your loss modifications
    with torch.no_grad():
        _, components = sensor_weighted_mse(
            pth_pred, pth_true, sensor_mask=pth_mask,
            grad_weight=1.0, spectral_weight=1.0, 
            peak_weight=1.0, wake_weight=1.0,
            return_components=True
        )

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'SSIM': ssim_val,
        'GradCorr': grad_corr,
        'WakeLoss': components.get('wake_loss', 0.0),
        'PeakLoss': components.get('peak_loss', 0.0),
        'SpectralLoss': components.get('spectral_loss', 0.0),
        'GradientLoss': components.get('gradient_loss', 0.0)
    }


def evaluate_csv_dir(csv_dir, model, device):
    """Evaluates the model over a directory of Grasshopper/CT CSV files."""
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    csv_files = [f for f in csv_files if "_pred" not in os.path.basename(f)]
    
    if not csv_files:
        print(f"No valid CSV files found in {csv_dir}")
        return [], {}

    all_metrics = []
    
    for f in tqdm(csv_files, desc="Evaluating CSVs"):
        df = pd.read_csv(f)
        rename_map = {"X": "X_coords", "Y": "Y_coords", "x": "X_coords", "y": "Y_coords", "U_at_z": "U_over_Uref"}
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)
                
        # Target column handling
        t_col = next((c for c in ["mag_U", "actual_U", "mag_U_dimensionless"] if c in df.columns), None)
        if t_col is None:
            print(f"Skipping {os.path.basename(f)}: Ground truth 'mag_U' not found.")
            continue
            
        t_flat = df[t_col].to_numpy()
        gh = {c: df[c].to_numpy() for c in ["SDF", "Bldg_height", "Z_relative", "U_over_Uref", "X_coords", "Y_coords", "dir_sin", "dir_cos"]}
        X_batch, _ = build_input_tensor_from_gh(gh, device="cpu")
        nx, ny, _, _, idx_map = infer_grid_from_coords_simple(df["X_coords"], df["Y_coords"])
        
        with torch.no_grad():
            X_device = X_batch[0].unsqueeze(0).to(device)
            pred_delta_t = model(X_device)
        
        # De-normalize predictions
        p_delta_flat = np.array([pred_delta_t[0, 0, iy, ix].cpu().item() for (iy, ix) in idx_map])
        u_ref_flat = df["U_over_Uref"].to_numpy()
        # Ensure it captures the exact delta_u bounds mapped
        p_mag_flat = np.clip(u_ref_flat * (p_delta_flat + 1.0), 0.0, None)
        
        # Reconstruct into 2D grid for consistent validation masking & spatial losses
        p_mag_grid = np.zeros((ny, nx))
        t_mag_grid = np.zeros((ny, nx))
        
        for i, (iy, ix) in enumerate(idx_map):
            p_mag_grid[iy, ix] = p_mag_flat[i]
            t_mag_grid[iy, ix] = t_flat[i]
            
        # Circular CFD mask to exclude out-of-domain evaluation
        cy_m, cx_m = ny // 2, nx // 2
        Yc_m, Xc_m = np.ogrid[:ny, :nx]
        outside = np.sqrt((Xc_m - cx_m)**2 + (Yc_m - cy_m)**2) >= (min(ny, nx)//2 - 5)
        mask = ~outside
        
        # Build metrics
        metrics = get_metrics_for_sample(p_mag_grid, t_mag_grid, mask, device=device)
        metrics['filename'] = os.path.basename(f)
        all_metrics.append(metrics)
        
    return all_metrics


def evaluate_npy_dir(dataroot, model, device):
    """Evaluates the model over monolithic X.npy and Y.npy arrays."""
    x_path = os.path.join(dataroot, "X.npy")
    y_path = os.path.join(dataroot, "Y.npy")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"Error: Could not find X.npy and/or Y.npy in {dataroot}")
        return []
        
    print(f"Loading arrays from {dataroot}...")
    X_val = np.load(x_path)
    Y_val = np.load(y_path)
    
    # If shapes are (N, H, W, C), transpose to (N, C, H, W)
    if X_val.ndim == 4 and X_val.shape[-1] == 8:
        X_val = X_val.transpose(0, 3, 1, 2)
    if Y_val.ndim == 4 and Y_val.shape[-1] == 1:
        Y_val = Y_val.transpose(0, 3, 1, 2)
        
    num_samples = X_val.shape[0]
    print(f"Loaded {num_samples} Numpy samples (X: {X_val.shape}, Y: {Y_val.shape}) for evaluation.")
    
    # (Optional) Attempt to load stats.json for inverse normalization if it exists. 
    # Otherwise, rely on raw prediction matches assuming they are in the same domain space.
    stats_path = os.path.join(dataroot, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        out_min = np.array(stats["output_stats"]["min"])
        out_max = np.array(stats["output_stats"]["max"])
    else:
        out_min, out_max = 0.0, 1.0
        
    all_metrics = []

    for i in tqdm(range(num_samples), desc="Evaluating NPYs"):
        x_tensor = torch.from_numpy(X_val[i:i+1]).float().to(device)
        y_true_tensor = torch.from_numpy(Y_val[i:i+1]).float().to(device)
        
        with torch.no_grad():
            y_pred_tensor = model(x_tensor)
            
        y_pred = y_pred_tensor[0, 0].cpu().numpy()
        y_true = y_true_tensor[0, 0].cpu().numpy()

        # Simple circle mask
        H, W = y_true.shape
        cy_m, cx_m = H // 2, W // 2
        Yc_m, Xc_m = np.ogrid[:H, :W]
        outside = np.sqrt((Xc_m - cx_m)**2 + (Yc_m - cy_m)**2) >= (min(H, W)//2 - 5)
        mask = ~outside

        metrics = get_metrics_for_sample(y_pred, y_true, mask, device=device)
        metrics['filename'] = f"sample_{i:04d}"
        all_metrics.append(metrics)

    return all_metrics

def main():
    parser = argparse.ArgumentParser(description="Comprehensively evaluate FNO Models.")
    parser.add_argument("--model", type=str, required=True, help="Path to your .pth checkoint")
    parser.add_argument("--model_type", type=str, required=True, choices=["standard", "hybrid", "pinn", "geo"], help="The model architecture used")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to your directory containing test CSVs, OR the root folder containing test_A and test_B directories for npy arrays")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "npy"], help="Format of the test data (csv or npy)")
    parser.add_argument("--out", type=str, default="evaluation_report", help="Prefix or directory for output report JSON and CSV")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {args.model_type} FNO checkpoint from {args.model} on {device}")
    
    state_dict = load_weights(args.model, device)
    model = build_model(args.model_type, state_dict, device)
    
    print(f"Starting rigorous benchmark on: {args.test_dir}")
    if args.format == "csv":
        all_metrics = evaluate_csv_dir(args.test_dir, model, device)
    elif args.format == "npy":
        all_metrics = evaluate_npy_dir(args.test_dir, model, device)

    if not all_metrics:
        print("No metrics collected. Check your directory or ground truth presence.")
        return
        
    df_metrics = pd.DataFrame(all_metrics)
    summary_dict = df_metrics.drop(columns=['filename']).mean().to_dict()
    
    print("\n" + "="*50)
    print("        COMPREHENSIVE TEST RESULTS        ")
    print("="*50)
    for k, v in summary_dict.items():
        print(f"{k.rjust(15)} : {v:.4f}")
    print("="*50)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    csv_out = f"{args.out}_per_sample.csv"
    json_out = f"{args.out}_summary.json"
    
    df_metrics.to_csv(csv_out, index=False)
    
    with open(json_out, "w") as f:
        json.dump({
            "model": args.model,
            "architecture": args.model_type,
            "test_dir": args.test_dir,
            "samples_evaluated": len(all_metrics),
            "aggregate_metrics": summary_dict
        }, f, indent=4)
        
    print(f"Per-sample metrics saved to -> {csv_out}")
    print(f"Aggregated summary saved to   -> {json_out}\n")

if __name__ == "__main__":
    main()
