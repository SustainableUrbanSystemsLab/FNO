import os, glob, torch, numpy as np, pandas as pd
from tqdm import tqdm
import tomllib

from gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple
from fno_hybrid_model import HybridFNO

# ================= CONFIG =================
CONFIG_FILE = "config.toml"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "rb") as f:
            return tomllib.load(f)
    return {}

config = load_config()

# --- Paths ---
TEST_FOLDER = config.get("paths", {}).get("test_folder", "test_csv")
MODEL_PATH  = "hybrid_fno_weights.pth" # Default hybrid weights file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Loading Hybrid Model: {MODEL_PATH}")

# ================= LOAD MODEL =================
model = None

# ================= FIND TEST FILES =================
if not os.path.exists(TEST_FOLDER):
    raise RuntimeError(f"Test folder not found: {TEST_FOLDER}")

files = sorted(f for f in glob.glob(os.path.join(TEST_FOLDER, "*.csv"))
               if not f.endswith("_pred.csv"))

print(f"Found {len(files)} test CSV files")

# ================= INFERENCE LOOP =================
for CSV in tqdm(files, desc="Hybrid Inference"):
    try:
        out_csv = CSV.replace(".csv", "_hybrid_pred.csv")
        df = pd.read_csv(CSV)

        # Standardize columns
        df.rename(columns={"X": "X_coords", "Y": "Y_coords", "x": "X_coords", "y": "Y_coords", "U_at_z": "U_over_Uref"}, inplace=True)
        required = ["SDF", "Bldg_height", "Z_relative", "U_over_Uref", "X_coords", "Y_coords", "dir_sin", "dir_cos"]
        
        gh = {c: df[c].to_numpy() for c in required}
        X, _ = build_input_tensor_from_gh(gh, device=DEVICE)

        if model is None:
            # Init Hybrid Model (Must match training params)
            model = HybridFNO(in_channels=X.shape[1], hidden_channels=64).to(DEVICE)
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            
            # Remove DDP prefix if present
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.eval()

        with torch.no_grad():
            pred = model(X)

        delta_grid = pred[0, 0].cpu().numpy()
        _, _, _, _, idx_map = infer_grid_from_coords_simple(df["X_coords"], df["Y_coords"])
        delta_flat = np.array([delta_grid[iy, ix] for (iy, ix) in idx_map])
        
        baseline = df["U_over_Uref"].to_numpy()
        mag_final = np.clip(baseline * (delta_flat + 1.0), 0.0, None)

        df["delta_pred"] = np.round(delta_flat, 6)
        df["mag_U"] = np.round(mag_final, 6)
        
        df.rename(columns={"X_coords": "x", "Y_coords": "y"}, inplace=True)
        df.to_csv(out_csv, index=False)

    except Exception as e:
        print(f"FAILED on {CSV}: {e}")

print("\nHybrid Inference finished.")
