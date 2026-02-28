import os, sys, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.models.hybrid import HybridFNO
from core.utils.gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple
from tools.diagnose_fno_wakes import analyze_prediction_quality, create_diagnostic_plot

def run_diagnostics(model_path, data_path, output_name="hybrid_diagnostic.png"):
    print(f"--- Hybrid Model Diagnostic: {os.path.basename(model_path)} ---")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    df.rename(columns={"X": "X_coords", "Y": "Y_coords", "x": "X_coords", "y": "Y_coords", "U_at_z": "U_over_Uref"}, inplace=True)
    
    # Check for ground truth
    target_col = None
    for c in ["mag_U", "actual_U", "mag_U_dimensionless"]:
        if c in df.columns:
            target_col = c
            break
            
    if target_col is None:
        print("Error: No ground truth (mag_U) found in CSV. Diagnostic requires target values.")
        return
        
    required = ["SDF", "Bldg_height", "Z_relative", "U_over_Uref", "X_coords", "Y_coords", "dir_sin", "dir_cos"]
    gh = {c: df[c].to_numpy() for c in required}
    X, _ = build_input_tensor_from_gh(gh, device=DEVICE)
    
    # 2. Load Model
    # Architecture params must match training
    model = HybridFNO(in_channels=X.shape[1], hidden_channels=64).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # Handle DDP and legacy formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if k != "_metadata"}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # 3. Inference
    with torch.no_grad():
        pred_tensor = model(X)
        
    # Scale back to magnitude
    pred_delta = pred_tensor[0, 0].cpu().numpy()
    nx, ny, _, _, idx_map = infer_grid_from_coords_simple(df["X_coords"], df["Y_coords"])
    
    pred_delta_flat = np.array([pred_delta[iy, ix] for (iy, ix) in idx_map])
    baseline = df["U_over_Uref"].to_numpy()
    pred_mag_flat = np.clip(baseline * (pred_delta_flat + 1.0), 0.0, None)
    
    target_mag_flat = df[target_col].to_numpy()
    
    # Reshape to grid for diagnostic plotting
    pred_grid = np.zeros((ny, nx))
    target_grid = np.zeros((ny, nx))
    sdf_grid = np.zeros((ny, nx))
    
    for i, (iy, ix) in enumerate(idx_map):
        pred_grid[iy, ix] = pred_mag_flat[i]
        target_grid[iy, ix] = target_mag_flat[i]
        sdf_grid[iy, ix] = df["SDF"].iloc[i]
        
    # 4. Analyze & Plot
    metrics = analyze_prediction_quality(pred_grid, target_grid)
    
    print("\nHYBRID MODEL HEALTH METRICS:")
    print(f"  Overall MAE:      {metrics['overall_mae']:.4f}")
    print(f"  Wake MAE:         {metrics['wake_mae']:.4f} (Goal: < 0.20)")
    print(f"  Gradient MAE:     {metrics['gradient_mae']:.4f} (Goal: < 0.04)")
    print(f"  Freq Preservation: {metrics['freq_preservation']:.2%}")
    
    create_diagnostic_plot(pred_grid, target_grid, sdf=sdf_grid, save_path=output_name)
    print(f"\nDiagnostic plot saved to {output_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="weights/hybrid_fno_weights.pth")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="hybrid_diagnostic.png")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        # try FNO/weights/...
        alt_path = os.path.join("FNO", args.model)
        if os.path.exists(alt_path):
            args.model = alt_path
            
    run_diagnostics(args.model, args.data, args.output)
