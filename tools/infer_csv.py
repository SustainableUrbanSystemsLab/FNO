#!/usr/bin/env python3
"""
FNO CSV Inference Script
=========================
Loads a trained FNO model (Standard, Hybrid, PINN, Geo), reads CSV input(s),
runs inference, and saves the prediction back out as a Grasshopper-compatible
CSV file + a visual PNG plot exactly matching the Conditional Transformer format.

Usage:
  python tools/infer_csv.py --csv test_csv/ML_FormFlux_1_135.csv --model geo_fno_weights.pth --model_type geo
"""
import os, sys, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt, glob, argparse
from matplotlib.colors import TwoSlopeNorm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from core.models.fno2d import FNO2d
from core.models.hybrid import HybridFNO
from core.models.pinn_fno import PINNFNO
from core.models.geo_fno import GeoFNO
from core.utils.gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple

def load_weights(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    sd = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    return {k.replace("module.", ""): v for k, v in sd.items() if k != "_metadata"}

def build_model(model_type, state_dict, DEVICE):
    import tomllib
    # Load defaults
    with open(os.path.join(os.path.dirname(__file__), '../config.toml'), "rb") as f:
        config = tomllib.load(f)
    modes = config.get("model", {}).get("modes1", 32)
    width = config.get("model", {}).get("width", 64)
    layers = config.get("model", {}).get("n_layers", 4)
    
    # Auto-detect parameters from state_dict to match the exact checkpoint
    if "in_proj.weight" in state_dict:
        width = state_dict["in_proj.weight"].shape[0]
        if "fourier_layers.0.0.weights1" in state_dict:
            w_shape = state_dict["fourier_layers.0.0.weights1"].shape
            modes = w_shape[2]
            
    elif "fno.fno_blocks.convs.0.bias" in state_dict:
        width = state_dict["fno.fno_blocks.convs.0.bias"].shape[0]
        if "fno.fno_blocks.convs.0.weight.tensor" in state_dict:
            modes = state_dict["fno.fno_blocks.convs.0.weight.tensor"].shape[2]
            
    print(f"Building {model_type} architecture -> Modes: {modes}, Width: {width}")
    
    # Init Classes
    typ = model_type.lower()
    if typ == "standard":
        model = FNO2d(in_channels=8, out_channels=1, modes1=modes, modes2=modes, width=width, n_layers=layers)
    elif typ == "hybrid":
        model = HybridFNO(in_channels=8, n_modes=(modes, modes), hidden_channels=width, n_layers=layers)
    elif typ == "pinn":
        model = PINNFNO(in_channels=8, n_modes=(modes, modes), hidden_channels=width, n_layers=layers)
    elif typ == "geo":
        model = GeoFNO(in_channels=8, n_modes=(modes, modes), hidden_channels=width, n_layers=layers)
    else:
        raise ValueError("Invalid model type. Choose from: standard, hybrid, pinn, geo")
        
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model

def process_single_csv(csv_path, model, DEVICE, output_dir=None):
    print(f"Processing: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Format exactly matching Conditional Transformer exports
    # Map lowercase 'x', 'y' to uppercase if export varied
    rename_map = {"X": "X_coords", "Y": "Y_coords", "x": "X_coords", "y": "Y_coords", "U_at_z": "U_over_Uref"}
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
            
    try:
        t_col = next((c for c in ["mag_U", "actual_U", "mag_U_dimensionless"] if c in df.columns))
        t_flat = df[t_col].to_numpy()
    except StopIteration:
        print("Warning: Ground truth target wind 'mag_U' not found in CSV. Visual Error plot will be skipped.")
        t_flat = None
        
    # Standardize data blocks
    gh = {c: df[c].to_numpy() for c in ["SDF", "Bldg_height", "Z_relative", "U_over_Uref", "X_coords", "Y_coords", "dir_sin", "dir_cos"]}
    X_batch, _ = build_input_tensor_from_gh(gh, device="cpu")
    nx, ny, _, _, idx_map = infer_grid_from_coords_simple(df["X_coords"], df["Y_coords"])
    
    # INFERENCE
    with torch.no_grad():
        X_device = X_batch[0].unsqueeze(0).to(DEVICE)
        pred_delta_t = model(X_device)
    
    # De-Normalize
    p_delta_flat = np.array([pred_delta_t[0, 0, iy, ix].cpu().item() for (iy, ix) in idx_map])
    u_ref_flat = df["U_over_Uref"].to_numpy()
    p_mag_flat = np.clip(u_ref_flat * (p_delta_flat + 1.0), 0.0, None)
    
    # 1. SAVE CSV
    df_out = df.copy()
    df_out['mag_U'] = p_mag_flat  # Overwrite or append exact predicted column
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        csv_out = os.path.join(output_dir, f"{base}_pred.csv")
        png_out = os.path.join(output_dir, f"{base}_pred.png")
    else:
        base = os.path.splitext(csv_path)[0]
        csv_out = f"{base}_pred.csv"
        png_out = f"{base}_pred.png"
        
    df_out.to_csv(csv_out, index=False)
    print(f"  -> Saved Grasshopper CSV Map: {csv_out}")

    # 2. SAVE PNG VISUALS
    p_mag_grid = np.zeros((ny, nx))
    bldg_mask = np.zeros((ny, nx))
    target_mag = np.zeros((ny, nx)) if t_flat is not None else None
    
    for i, (iy, ix) in enumerate(idx_map):
        p_mag_grid[iy, ix] = p_mag_flat[i]
        bldg_mask[iy, ix] = df["SDF"].iloc[i] > 0
        if target_mag is not None:
            target_mag[iy, ix] = t_flat[i]
            
    fig, axes = plt.subplots(1, 3 if target_mag is not None else 1, figsize=(15 if target_mag is not None else 6, 5))
    if target_mag is None: axes = [axes]
    
    # Ground Truth
    if target_mag is not None:
        im0 = axes[0].imshow(target_mag, origin='lower', cmap='viridis')
        axes[0].set_title("CFD Ground Truth")
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0])
        
    # Prediction
    ax_pred = axes[1] if target_mag is not None else axes[0]
    im1 = ax_pred.imshow(p_mag_grid, origin='lower', cmap='viridis')
    ax_pred.set_title("FNO Prediction")
    ax_pred.axis('off')
    fig.colorbar(im1, ax=ax_pred)
    
    # Error Diff
    if target_mag is not None:
        diff = p_mag_grid - target_mag
        im2 = axes[2].imshow(diff, origin='lower', cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0.0, vmin=-0.5, vmax=0.5))
        axes[2].set_title("Absolute Error")
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2])
        
    plt.tight_layout()
    plt.savefig(png_out, dpi=150)
    plt.close()
    print(f"  -> Saved Visual Plot: {png_out}")

def main():
    parser = argparse.ArgumentParser(description="FNO CSV Exporter matching Transformer Format")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file or directory containing CSVs")
    parser.add_argument("--model", type=str, required=True, help="Path to your .pth weights file")
    parser.add_argument("--model_type", type=str, required=True, choices=["standard", "hybrid", "pinn", "geo"], help="The architecture used to train the weights")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to save the output CSVs")
    args = parser.parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = load_weights(args.model, DEVICE)
    model = build_model(args.model_type, state_dict, DEVICE)
    
    if os.path.isdir(args.csv):
        files = glob.glob(os.path.join(args.csv, "*.csv"))
        files = [f for f in files if "_pred" not in os.path.basename(f)]
        for f in files: process_single_csv(f, model, DEVICE, args.output_dir)
    else:
        process_single_csv(args.csv, model, DEVICE, args.output_dir)

if __name__ == "__main__":
    main()
