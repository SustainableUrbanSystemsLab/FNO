import pandas as pd, numpy as np, torch, os, glob, re
from gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple
from fno2d_model import FNO2d

TEST_FOLDER = "test_csv"
MODEL_BASE = "fno_mag_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROUND = 2

# 1. Use the main model (which is now the BEST model from training)
model_path = MODEL_BASE
print(f"Using model: {model_path}")

# Load model once
model = None

if not os.path.exists(TEST_FOLDER):
    print(f"Test folder {TEST_FOLDER} does not exist.")
    files = []
else:
    files = sorted(glob.glob(os.path.join(TEST_FOLDER, "*.csv")))
    # Exclude _pred.csv files
    files = [f for f in files if "_pred.csv" not in f]
    print(f"Found {len(files)} files in {TEST_FOLDER}")

for CSV in files:
    out_name = CSV.replace('.csv', '_pred.csv')
    print(f"Processing {CSV} -> {out_name}")
    
    try:
        df = pd.read_csv(CSV)
        # Renaming known variations
        rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
        df.rename(columns=rename_map, inplace=True)

        cols = ['SDF','Bldg_height','Z_relative','U_at_z','X_coords','Y_coords','dir_sin','dir_cos']
        for c in cols:
            if c not in df.columns:
                raise RuntimeError(f"Missing input column {c} in {CSV}")
        
        gh = {c:df[c].tolist() for c in cols}
        X, chs = build_input_tensor_from_gh(gh, H=None, W=None, device=DEVICE)
        
        # DEBUG: Check if inputs are actually different
        # Channel 0=SDF, 4=X_norm, 5=Y_norm
        print(f"  Input Stats:")
        print(f"    SDF:  min={X[0,0].min():.4f}, max={X[0,0].max():.4f}, mean={X[0,0].mean():.4f}")
        print(f"    NormX: min={X[0,4].min():.4f}, max={X[0,4].max():.4f}")
        print(f"    NormY: min={X[0,5].min():.4f}, max={X[0,5].max():.4f}")

        
        # Init model if needed
        # Init model if needed
        if model is None:
            try:
                in_ch = X.shape[1]
                m = FNO2d(in_channels=in_ch, out_channels=1, modes1=20, modes2=20, width=64, n_layers=4).to(DEVICE)
                m.load_state_dict(torch.load(model_path, map_location=DEVICE))
                m.eval()
                model = m
            except Exception as e:
                print(f"CRITICAL: Failed to load model {model_path}: {e}")
                # Do not proceed with this file or future files if model is broken
                break

        with torch.no_grad():
            mag_star = model(X.to(DEVICE)).cpu().numpy()[0,0]  # dimensionless grid (H,W)

        # Robustly map predictions back to original points (handles sparse/unsorted)
        nx, ny, _, _, idx_map = infer_grid_from_coords_simple(df['X_coords'], df['Y_coords'])
        flat = np.array([mag_star[iy, ix] for (iy, ix) in idx_map])

        # write dimensionless predictions; if U_ref present also write physical mag
        df['mag_U'] = np.round(flat, ROUND)
        if 'U_ref' in df.columns:
            Uref = float(df['U_ref'].iloc[0])
            df['mag_U_pred'] = np.round(flat * Uref, ROUND)
        
        # Rename back to x, y for output consistency
        df.rename(columns={'X_coords': 'x', 'Y_coords': 'y'}, inplace=True)
        
        df.to_csv(out_name, index=False)
    except Exception as e:
        print(f"Failed to process {CSV}: {e}")

print("Batch inference finished.")
