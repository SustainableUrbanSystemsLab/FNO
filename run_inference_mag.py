import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple
from fno2d_model import FNO2d

# ================= CONFIG =================
CONFIG_FILE = "config.toml"

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

            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()

        # --- Forward pass ---
        with torch.no_grad():
            pred = model(X)

        # ================= POST-PROCESS =================
        # Model predicts DELTA = (U / U_baseline) - 1
        delta_grid = pred[0, 0].cpu().numpy()

        # Map grid back to flat CSV ordering
        _, _, _, _, idx_map = infer_grid_from_coords_simple(
            df["X_coords"], df["Y_coords"]
        )

        delta_flat = np.array([delta_grid[iy, ix] for (iy, ix) in idx_map])

        # -------- SANITY PRINTS (keep for now) --------
        print(f"\n{os.path.basename(CSV)}")
        print(f"  delta RAW: min={delta_flat.min():.3f}, "
              f"mean={delta_flat.mean():.3f}, "
              f"max={delta_flat.max():.3f}")

        baseline = df["U_over_Uref"].to_numpy()
        mag_raw = baseline * (1.0 + delta_flat)

        print(f"  mag RAW:   min={mag_raw.min():.3f}, "
              f"mean={mag_raw.mean():.3f}, "
              f"max={mag_raw.max():.3f}")

        # -------- OPTIONAL SAFETY CLIP (disabled for now) --------
        # Uncomment ONLY after you verify raw output is reasonable
        #
        # delta_flat = np.clip(delta_flat, -0.8, 1.5)
        # mag_raw = baseline * (1.0 + delta_flat)

        # ================= SAVE OUTPUT =================
        df["delta_pred"] = delta_flat
        df["mag_U"] = mag_raw

        if "U_ref" in df.columns:
            Uref = float(df["U_ref"].iloc[0])
            df["mag_U_pred"] = mag_raw * Uref

        df.rename(columns={"X_coords": "x", "Y_coords": "y"}, inplace=True)
        df.to_csv(out_csv, index=False)

    except Exception as e:
        print(f"FAILED on {CSV}: {e}")

print("\nInference finished.")
