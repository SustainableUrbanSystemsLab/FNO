import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple
from fno2d_model import FNO2d

# ================= CONFIG =================
CONFIG_FILE = "config_wake_focused.toml"

def load_config():
    import tomllib  # Python 3.11+
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "rb") as f:
            return tomllib.load(f)
    return {}

config = load_config()

# --- Model architecture (MUST match training) ---
MODES1   = config.get("model", {}).get("modes1", 64)
MODES2   = config.get("model", {}).get("modes2", 64)
WIDTH    = config.get("model", {}).get("width", 96)
N_LAYERS = config.get("model", {}).get("n_layers", 5)

# --- Paths ---
TEST_FOLDER = config.get("paths", {}).get("test_folder", "test_csv")
MODEL_PATH  = config.get("paths", {}).get("model_output", "best_model.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Loading model: {MODEL_PATH}")

# ================= LOAD MODEL =================
model = None

# ================= FIND TEST FILES =================
if not os.path.exists(TEST_FOLDER):
    raise RuntimeError(f"Test folder not found: {TEST_FOLDER}")

files = sorted(f for f in glob.glob(os.path.join(TEST_FOLDER, "*.csv"))
               if not f.endswith("_pred.csv"))

print(f"Found {len(files)} test CSV files")

# ================= INFERENCE LOOP =================
for CSV in tqdm(files, desc="Inference"):
    try:
        out_csv = CSV.replace(".csv", "_pred.csv")
        df = pd.read_csv(CSV)

        # --- Standardize column names ---
        df.rename(columns={
            "X": "X_coords", "Y": "Y_coords",
            "x": "X_coords", "y": "Y_coords",
            "U_at_z": "U_over_Uref"
        }, inplace=True)

        required = [
            "SDF", "Bldg_height", "Z_relative",
            "U_over_Uref", "X_coords", "Y_coords",
            "dir_sin", "dir_cos"
        ]
        for c in required:
            if c not in df.columns:
                raise RuntimeError(f"Missing column {c} in {CSV}")

        # --- Build input tensor ---
        gh = {c: df[c].to_numpy() for c in required}
        X, _ = build_input_tensor_from_gh(gh, device=DEVICE)

        # --- Init model once ---
        if model is None:
            in_ch = X.shape[1]
            model = FNO2d(
                in_channels=in_ch,
                out_channels=1,
                modes1=MODES1,
                modes2=MODES2,
                width=WIDTH,
                n_layers=N_LAYERS
            ).to(DEVICE)

            
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            
            # Fix DDP 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v  # remove "module."
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict)
            model.eval()

        # --- Forward pass ---
        with torch.no_grad():
            pred = model(X)

        # ================= POST-PROCESS =================
        # Model predicts DELTA (Normalized Difference)
        # Delta = (Mag - Uref) / Uref
        # -> Mag = (Delta + 1) * Uref
        
        delta_grid = pred[0, 0].cpu().numpy()

        # Map grid back to flat CSV ordering
        _, _, _, _, idx_map = infer_grid_from_coords_simple(
            df["X_coords"], df["Y_coords"]
        )

        delta_flat = np.array([delta_grid[iy, ix] for (iy, ix) in idx_map])
        
        # Get Reference Velocity (Inlet Profile)
        # U_over_Uref is effectively U_inlet / U_ref(scalar)
        # Since we work in dimensionless units, U_over_Uref IS the reference for this pixel.
        baseline = df["U_over_Uref"].to_numpy()
        
        # Reconstruction: Mag = Baseline * (Delta + 1)
        mag_reconstructed = baseline * (delta_flat + 1.0)
        
        # Clip to ensure physics (non-negative)
        mag_final = np.clip(mag_reconstructed, 0.0, None)

        # -------- SANITY PRINTS (keep for now) --------
        print(f"\n{os.path.basename(CSV)}")
        print(f"  Delta Pred: min={delta_flat.min():.3f}, max={delta_flat.max():.3f}")
        print(f"  Mag Final:  min={mag_final.min():.3f}, mean={mag_final.mean():.3f}, max={mag_final.max():.3f}")

        # ================= SAVE OUTPUT =================
        # Rename ground truth 'mag_U' -> 'actual_U' to avoid conflict
        if "mag_U" in df.columns:
            df.rename(columns={"mag_U": "actual_U"}, inplace=True)

        df["delta_pred"] = np.round(delta_flat, 6)
        df["mag_U"] = np.round(mag_final, 6) # Prediction (Normalized)

        # Performance Metrics (if ground truth exists)
        if "actual_U" in df.columns:
            mae = np.abs(df["actual_U"] - df["mag_U"]).mean()
            print(f"  MAE (mag_U): {mae:.6f}")
            
            # Print debug for Row 147 if available
            if len(df) > 147:
                k = 147
                true_val = df["actual_U"].iloc[k]
                pred_val = df["mag_U"].iloc[k]
                print(f"  [Row {k}] True: {true_val:.4f} | Pred: {pred_val:.4f} | Diff: {pred_val-true_val:.4f}")

        if "U_ref" in df.columns:
            Uref = float(df["U_ref"].iloc[0])
            df["mag_U_dimensional"] = mag_final * Uref # Dimensional prediction

        df.rename(columns={"X_coords": "x", "Y_coords": "y"}, inplace=True)
        df.to_csv(out_csv, index=False)

    except Exception as e:
        print(f"FAILED on {CSV}: {e}")
        import traceback
        traceback.print_exc()

print("\nInference finished.")
