#!/usr/bin/env python3
"""
FNO Inference Script using NeuralOperator Library
==================================================
Runs inference on test CSVs using the trained NeuralOperator FNO model.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# NeuralOperator imports
from neuralop.models import FNO

# Local imports
from gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple

# ============ Load Configuration ============
CONFIG_FILE = "config.toml"

def load_config():
    """Load configuration from config.toml file."""
    import tomllib
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            return tomllib.load(f)
    return {}

config = load_config()

# Model params from config
MODES = config.get('model', {}).get('modes1', 48)
WIDTH = config.get('model', {}).get('width', 64)
N_LAYERS = config.get('model', {}).get('n_layers', 5)

# Paths
TEST_FOLDER = config.get('paths', {}).get('test_folder', 'test_csv')
MODEL_PATH = config.get('paths', {}).get('model_output', 'fno_mag_weights.pth')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path, in_channels, device):
    """Load the trained NeuralOperator FNO model."""
    model = FNO(
        n_modes=(MODES, MODES),
        in_channels=in_channels,
        out_channels=1,
        hidden_channels=WIDTH,
        n_layers=N_LAYERS,
        positional_embedding='grid',
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        fno_skip='linear',
        norm='instance_norm',
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(csv_path, model, device):
    """Run inference on a single CSV file."""
    df = pd.read_csv(csv_path)
    
    # Column renaming for compatibility
    rename_map = {
        'X': 'X_coords', 'Y': 'Y_coords',
        'x': 'X_coords', 'y': 'Y_coords',
        'U_at_z': 'U_over_Uref',
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Required columns
    cols = ['SDF', 'Bldg_height', 'Z_relative', 'U_over_Uref', 'X_coords', 'Y_coords', 'dir_sin', 'dir_cos']
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")
    
    # Build input tensor
    gh_out = {c: df[c].tolist() for c in cols}
    X, chs = build_input_tensor_from_gh(gh_out, H=None, W=None, device=device)
    
    # Print input stats
    print(f"  Input shape: {X.shape}")
    print(f"  SDF: min={X[0,0].min():.3f}, max={X[0,0].max():.3f}")
    
    # Run model
    with torch.no_grad():
        pred = model(X.to(device))
    
    # Extract predictions
    mag_pred = pred[0, 0].cpu().numpy()
    nx, ny, _, _, idx_map = infer_grid_from_coords_simple(df['X_coords'], df['Y_coords'])
    flat = np.array([mag_pred[iy, ix] for (iy, ix) in idx_map])
    
    # Enforce physical bounds
    flat = np.clip(flat, -1.0, 0.5)
    
    print(f"  Pred delta: min={flat.min():.3f}, mean={flat.mean():.3f}, max={flat.max():.3f}")
    
    # Reconstruct: mag_U = U_over_Uref * (1 + delta)
    u_over_uref = df['U_over_Uref'].to_numpy()
    mag_U_final = u_over_uref * (1.0 + flat)
    
    # Save predictions
    df['delta_pred'] = np.round(flat, 4)
    df['mag_U'] = np.round(mag_U_final, 3)
    
    if 'U_ref' in df.columns:
        Uref = float(df['U_ref'].iloc[0])
        df['mag_U_pred'] = np.round(mag_U_final * Uref, 3)
    
    # Rename back to x, y for output
    df.rename(columns={'X_coords': 'x', 'Y_coords': 'y'}, inplace=True)
    
    return df


def main():
    print("=" * 50)
    print("NeuralOperator FNO Inference")
    print("=" * 50)
    
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
    
    print(f"Using model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Find test CSVs
    if not os.path.exists(TEST_FOLDER):
        print(f"Test folder not found: {TEST_FOLDER}")
        return
    
    csv_files = sorted(glob.glob(os.path.join(TEST_FOLDER, "*.csv")))
    csv_files = [f for f in csv_files if not f.endswith('_pred.csv')]
    
    print(f"Found {len(csv_files)} test files")
    
    if len(csv_files) == 0:
        return
    
    # Load model (need to know input channels from first file)
    model = None
    
    for csv_path in tqdm(csv_files, desc="Processing"):
        try:
            basename = os.path.basename(csv_path)
            print(f"\nProcessing: {basename}")
            
            # Load model on first iteration
            if model is None:
                # Peek at first file to get input channels
                df_peek = pd.read_csv(csv_path)
                rename_map = {'U_at_z': 'U_over_Uref', 'X': 'X_coords', 'Y': 'Y_coords'}
                df_peek.rename(columns=rename_map, inplace=True)
                cols = ['SDF', 'Bldg_height', 'Z_relative', 'U_over_Uref', 'X_coords', 'Y_coords', 'dir_sin', 'dir_cos']
                gh_out = {c: df_peek[c].tolist() for c in cols}
                X_peek, _ = build_input_tensor_from_gh(gh_out, H=None, W=None, device='cpu')
                in_channels = X_peek.shape[1]
                
                print(f"Loading model (in_channels={in_channels})...")
                model = load_model(MODEL_PATH, in_channels, DEVICE)
                print("Model loaded successfully!")
            
            # Run inference
            df_out = run_inference(csv_path, model, DEVICE)
            
            # Save output
            out_path = csv_path.replace('.csv', '_pred.csv')
            df_out.to_csv(out_path, index=False)
            print(f"  Saved: {os.path.basename(out_path)}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
