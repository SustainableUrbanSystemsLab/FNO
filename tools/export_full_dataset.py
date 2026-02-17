import os, sys
import glob
import torch
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.utils.gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple

def get_tensor_for_file(fp):
    try:
        df = pd.read_csv(fp)
        rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
        df.rename(columns=rename_map, inplace=True)
        
        # Grid Inference
        cols = ['SDF','Bldg_height','Z_relative','U_over_Uref','X_coords','Y_coords','dir_sin','dir_cos']
        xs = df['X_coords'].values
        ys = df['Y_coords'].values
        infer = infer_grid_from_coords_simple(xs, ys)
        if infer is None: return None
        nx, ny, _, _, idx_map = infer
        
        # Build INPUT (X)
        gh_data = {c: df[c].tolist() for c in cols}
        X_tensor, ch_names = build_input_tensor_from_gh(gh_data, H=ny, W=nx) 
        
        # Build TARGET (Y)
        target_valid = np.zeros((1, ny, nx), dtype=np.float32)
        
        mag_vals = None
        for c in ['mag_U_dimensionless','mag_U','mag_dimensionless']:
            if c in df.columns:
                mag_vals = df[c].to_numpy().astype(float)
                break
        
        if mag_vals is None and all(cc in df.columns for cc in ['Ux_dimensionless','Uy_dimensionless','Uz_dimensionless']):
             ux = df['Ux_dimensionless'].values.astype(float)
             uy = df['Uy_dimensionless'].values.astype(float)
             uz = df['Uz_dimensionless'].values.astype(float)
             mag_vals = np.sqrt(ux**2 + uy**2 + uz**2)

        if mag_vals is not None:
            uref = df['U_over_Uref'].to_numpy().astype(float)
            for i, (iy, ix) in enumerate(idx_map):
                m = mag_vals[i]
                u = uref[i]
                if np.isfinite(m) and u > 0:
                    delta = (m - u) / (u + 1e-6)
                    delta = np.clip(delta, -2.0, 10.0) # Apply the Fix
                    target_valid[0, iy, ix] = delta

        Y_tensor = torch.from_numpy(target_valid)
        
        return X_tensor, Y_tensor, ch_names

    except Exception as e:
        return None

def main():
    # Detect data folder
    data_folder = "train_csv"
    possible_paths = [
        "/home/hice1/ikaradag3/scratch/FNO/Training_Dataset", # ICE
        "C:/LabShare/Dataset/FormFluxCases/Compressed/Training_Dataset", # Local Windows
        "train_csv" # Fallback
    ]
    for p in possible_paths:
        if os.path.exists(p):
            data_folder = p
            print(f"Dataset found at: {data_folder}")
            break
            
    files = sorted(glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True))
    if not files:
        print("No CSV files found.")
        return

    print(f"Processing {len(files)} files...")
    
    # Use Multiprocessing
    pool_size = min(32, cpu_count())
    with Pool(pool_size) as pool:
        results = list(tqdm(pool.imap(get_tensor_for_file, files), total=len(files)))
    
    # Filter None
    valid_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(valid_results)}/{len(files)}")
    
    if not valid_results:
        print("No valid data found.")
        return

    Xs = [r[0] for r in valid_results]
    Ys = [r[1] for r in valid_results]
    names = valid_results[0][2]
    
    # Check dimensions
    shapes = [x.shape for x in Xs]
    if len(set(shapes)) > 1:
        print("WARNING: Variable grid sizes detected! Cannot stack into single tensor.")
        print(f"Shapes found: {set(shapes)}")
        print("Saving as List of Tensors...")
        X_final = Xs
        Y_final = Ys
        stack_dim = "List (Variable Size)"
    else:
        print("Grid sizes are uniform. Stacking...")
        X_final = torch.stack(Xs) # [N, C, H, W]
        Y_final = torch.stack(Ys) # [N, 1, H, W]
        stack_dim = tuple(X_final.shape)

    # Save
    output_pt = "full_dataset_2000.pt"
    print(f"Saving to {output_pt}...")
    torch.save({'X': X_final, 'Y': Y_final, 'channel_names': names}, output_pt)
    
    # Save NPZ (Only if stacked, npz doesn't like lists of jagged arrays easily)
    if isinstance(X_final, torch.Tensor):
        output_npz = "full_dataset_2000.npz"
        print(f"Saving to {output_npz}...")
        np.savez_compressed(output_npz, X=X_final.numpy(), Y=Y_final.numpy(), channel_names=names)
    
    print("\nDone!")
    print(f"Final Info:")
    print(f"  Shape: {stack_dim}")
    print(f"  Channels: {names}")

if __name__ == "__main__":
    main()
