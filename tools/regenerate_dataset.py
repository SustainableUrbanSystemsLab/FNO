"""
Regenerate X.npy and Y.npy from CSV training files.
=======================================================
Fixes two critical data format problems:
  1. Y.npy was storing raw mag_U — now stores delta_u = (mag - U_ref) / U_ref
  2. X.npy was storing unnormalized raw values — now uses build_input_tensor_from_gh

Run on the ICE cluster (single interactive node or batch job):
    uv run python tools/regenerate_dataset.py

Progress bar shows ETA. On 12CPU node with 4000 files ~30-60 minutes.
"""
import os, sys, glob, numpy as np, pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from core.utils.gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple

# ===== Paths =====
CSV_PATHS = [
    "/storage/ice1/2/4/athach7/Training_Dataset",
    "C:/LabShare/Dataset/FormFluxCases/Compressed/Training_Dataset",
    "./train_csv",
]
OUTPUT_DIR = None  # Will be set to same folder as CSVs

INPUT_COLS  = ['SDF','Bldg_height','Z_relative','U_over_Uref',
               'X_coords','Y_coords','dir_sin','dir_cos']
TARGET_COLS = ['mag_U_dimensionless','mag_U','mag_dimensionless']
RENAME_MAP  = {'X':'X_coords','Y':'Y_coords','x':'X_coords','y':'Y_coords',
               'U_at_z':'U_over_Uref'}

TARGET_H = 504
TARGET_W = 504


def process_file(fp):
    try:
        df = pd.read_csv(fp)
        df.rename(columns=RENAME_MAP, inplace=True)

        if any(c not in df.columns for c in INPUT_COLS):
            return None, f"Missing columns in {fp}"

        # Find mag column
        mag_col = next((c for c in TARGET_COLS if c in df.columns), None)
        if mag_col is None:
            # Try computing from U components
            comp = ['Ux_dimensionless','Uy_dimensionless','Uz_dimensionless']
            if all(c in df.columns for c in comp):
                mag_vals = np.sqrt(df['Ux_dimensionless']**2 +
                                   df['Uy_dimensionless']**2 +
                                   df['Uz_dimensionless']**2).to_numpy()
            else:
                return None, f"No mag/velocity column in {fp}"
        else:
            mag_vals = df[mag_col].to_numpy().astype(float)

        xs = df['X_coords'].to_numpy()
        ys = df['Y_coords'].to_numpy()
        result = infer_grid_from_coords_simple(xs, ys)
        if result is None:
            return None, f"Grid inference failed: {fp}"
        nx, ny, _, _, idx_map = result

        # Build normalized X tensor via official pipeline
        gh = {c: df[c].to_numpy() for c in INPUT_COLS}
        X_tensor, _ = build_input_tensor_from_gh(gh, device='cpu')
        X_np = X_tensor.squeeze(0).numpy()  # (8, ny, nx)

        # Build delta_u target
        uref    = df['U_over_Uref'].to_numpy().astype(float)
        Y_grid  = np.zeros((1, ny, nx), dtype=np.float32)
        for i, (iy, ix) in enumerate(idx_map):
            m, u = float(mag_vals[i]), float(uref[i])
            if np.isfinite(m) and u > 1e-6:
                delta = np.clip((m - u) / u, -2.0, 10.0)
            else:
                delta = 0.0
            Y_grid[0, iy, ix] = delta

        # Pad / crop to TARGET_H x TARGET_W so all samples stack
        def pad_or_crop(arr, th, tw):
            # arr: (C, h, w)
            C, h, w = arr.shape
            out = np.zeros((C, th, tw), dtype=arr.dtype)
            ch, cw = min(h, th), min(w, tw)
            out[:, :ch, :cw] = arr[:, :ch, :cw]
            return out

        X_np  = pad_or_crop(X_np,  TARGET_H, TARGET_W)
        Y_grid = pad_or_crop(Y_grid, TARGET_H, TARGET_W)

        return (X_np, Y_grid), None

    except Exception as e:
        import traceback
        return None, f"{fp}: {e}\n{traceback.format_exc()}"


def main():
    global OUTPUT_DIR

    # Find CSV folder
    csv_folder = None
    for p in CSV_PATHS:
        if os.path.isdir(p):
            csv_folder = p; break
    if csv_folder is None:
        print("ERROR: Could not find CSV training folder."); sys.exit(1)

    OUTPUT_DIR = csv_folder
    files = sorted(glob.glob(os.path.join(csv_folder, "**", "*.csv"), recursive=True))
    # Exclude _pred outputs
    files = [f for f in files if '_pred' not in os.path.basename(f)]

    if not files:
        print(f"ERROR: No CSV files found in {csv_folder}"); sys.exit(1)

    print(f"Found {len(files)} CSV files in {csv_folder}")
    print(f"Output: {OUTPUT_DIR}/X.npy  and  Y.npy")
    print(f"Grid: padded/cropped to {TARGET_H}×{TARGET_W}")

    workers = max(1, min(cpu_count() - 1, 16))
    print(f"Using {workers} parallel workers...\n")

    Xs, Ys, errors = [], [], []
    with Pool(workers) as pool:
        for result, err in tqdm(pool.imap(process_file, files), total=len(files)):
            if err:
                errors.append(err)
            else:
                x, y = result
                Xs.append(x)
                Ys.append(y)

    print(f"\nSuccessfully processed: {len(Xs)}/{len(files)}")
    if errors:
        print(f"Errors ({len(errors)}): first 3 shown below")
        for e in errors[:3]: print(" ", e)

    if not Xs:
        print("No valid data — check errors above."); sys.exit(1)

    X_arr = np.stack(Xs).astype(np.float32)  # (N, 8, H, W)
    Y_arr = np.stack(Ys).astype(np.float32)  # (N, 1, H, W)

    print(f"\nX shape: {X_arr.shape}")
    print(f"Y shape: {Y_arr.shape}")
    print(f"Y range: [{Y_arr.min():.3f}, {Y_arr.max():.3f}]  mean={Y_arr.mean():.3f}  std={Y_arr.std():.3f}")

    # Quick sanity check
    y_flat = Y_arr.flatten()
    near_zero = (np.abs(y_flat) < 0.05).mean()
    wake_frac  = (y_flat < -0.2).mean()
    print(f"Y near-zero: {near_zero:.1%}  |  Wake pixels (delta<-0.2): {wake_frac:.1%}")

    x_out = os.path.join(OUTPUT_DIR, "X.npy")
    y_out = os.path.join(OUTPUT_DIR, "Y.npy")

    # Back up old files first
    if os.path.exists(x_out):
        os.rename(x_out, x_out + ".bak")
        print(f"Backed up old X.npy → X.npy.bak")
    if os.path.exists(y_out):
        os.rename(y_out, y_out + ".bak")
        print(f"Backed up old Y.npy → Y.npy.bak")

    print(f"\nSaving X.npy ...")
    np.save(x_out, X_arr)
    print(f"Saving Y.npy ...")
    np.save(y_out, Y_arr)

    print(f"\n✓ Done! Regenerated dataset saved to:")
    print(f"  {x_out}")
    print(f"  {y_out}")
    print(f"\nNow retrain with:")
    print(f"  bash slurm/deploy_ice.sh --script pipelines/train/hybrid.py --gpu h100 --ngpus 2 --fresh")


if __name__ == '__main__':
    main()
