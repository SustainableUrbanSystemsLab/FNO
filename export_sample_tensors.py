import os
import glob
import torch
import numpy as np
import pandas as pd
from gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple

def export_sample():
    # 1. Find a sample file
    data_folder = "train_csv"
    possible_paths = [
        "/home/hice1/ikaradag3/scratch/FNO/Training_Dataset", # ICE
        "C:/LabShare/Dataset/FormFluxCases/Compressed/Training_Dataset", # Local Windows
        "train_csv" # Fallback
    ]
    for p in possible_paths:
        if os.path.exists(p):
            data_folder = p
            break
            
    files = sorted(glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True))
    if not files:
        print("No CSV files found.")
        return

    sample_file = files[0]
    print(f"Processing sample file: {sample_file}")

    # 2. Logic from train_fno_distributed.py (Simulated)
    try:
        df = pd.read_csv(sample_file)
        rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
        df.rename(columns=rename_map, inplace=True)
        
        # Grid Inference
        cols = ['SDF','Bldg_height','Z_relative','U_over_Uref','X_coords','Y_coords','dir_sin','dir_cos']
        xs = df['X_coords'].values
        ys = df['Y_coords'].values
        infer = infer_grid_from_coords_simple(xs, ys)
        if infer is None:
            print("Grid inference failed.")
            return
        nx, ny, _, _, idx_map = infer
        
        # Build INPUT (X)
        gh_data = {c: df[c].tolist() for c in cols}
        # This calls the normalization logic!
        X_tensor, ch_names = build_input_tensor_from_gh(gh_data, H=ny, W=nx) 
        
        # Build TARGET (Y)
        # Recreate target logic
        target_valid = np.zeros((1, ny, nx), dtype=np.float32)
        
        # Find mag column
        mag_vals = None
        for c in ['mag_U_dimensionless','mag_U','mag_dimensionless']:
            if c in df.columns:
                mag_vals = df[c].to_numpy().astype(float)
                break
        
        if mag_vals is None:
             if all(cc in df.columns for cc in ['Ux_dimensionless','Uy_dimensionless','Uz_dimensionless']):
                 ux = df['Ux_dimensionless'].values.astype(float)
                 uy = df['Uy_dimensionless'].values.astype(float)
                 uz = df['Uz_dimensionless'].values.astype(float)
                 mag_vals = np.sqrt(ux**2 + uy**2 + uz**2)

        if mag_vals is not None:
            uref = df['U_over_Uref'].to_numpy().astype(float)
            
            # Map flattened values to grid
            # Note: idx_map in infer_grid_from_coords_simple returns (y_idx, x_idx) for the flattened arrays
            # We need to construct the 2D target manually if not using the idx_map from gh_to_fno (which returns valid mask)
            # Actually infer_grid returns (nx, ny, idx_map, u_xs, u_ys)
            
            # Let's simple fill:
            for i, (iy, ix) in enumerate(idx_map):
                m = mag_vals[i]
                u = uref[i]
                if np.isfinite(m) and u > 0:
                    # Target Calculation
                    delta = (m - u) / (u + 1e-6)
                    # CLIP (The Fix)
                    delta = np.clip(delta, -2.0, 10.0)
                    target_valid[0, iy, ix] = delta
                else:
                    target_valid[0, iy, ix] = 0.0 # padding
        
        Y_tensor = torch.from_numpy(target_valid)

        # 3. Save as .pt (PyTorch) - Standard for FNO
        output_pt = "sample_data.pt"
        torch.save({'X': X_tensor, 'Y': Y_tensor, 'channel_names': ch_names}, output_pt)
        
        # 4. Save as .npz (NumPy) - Universal
        output_npz = "sample_data.npz"
        np.savez(output_npz, X=X_tensor.numpy(), Y=Y_tensor.numpy(), channel_names=ch_names)
        
        print(f"\nSuccess!")
        print(f"1) Processed Tensor Data saved to:")
        print(f"   - {output_pt} (for PyTorch users)")
        print(f"   - {output_npz} (for NumPy/Matlab users)")
        
        print("\n2) Data Structure:")
        print(f"   X (Input):  Shape {tuple(X_tensor.shape)} -> [Channels, Height, Width]")
        print(f"   Y (Target): Shape {tuple(Y_tensor.shape)} -> [1, Height, Width]")
        print(f"   Channels:   {ch_names}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_sample()
