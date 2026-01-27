import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch

# Import the EXACT normalization logic used in training
from gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple

# Configuration
# ----------------
# We will bins to capture the distributions.
# Most features should be in [-1, 1], so we focus there but include tails.
BIN_EDGES = np.linspace(-2.5, 5.5, 161) # 160 bins, step 0.05. Covers [-2.5, 5.5]
# Special zoom for 0-1 features
BIN_EDGES_01 = np.linspace(-0.2, 1.2, 141) # 140 bins, step 0.01

def get_histograms_for_file(fp):
    try:
        df = pd.read_csv(fp)
        rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
        df.rename(columns=rename_map, inplace=True)

        cols = ['SDF','Bldg_height','Z_relative','U_over_Uref','X_coords','Y_coords','dir_sin','dir_cos']
        if any(c not in df.columns for c in cols):
            return None
        
        # 1. Infer Grid to reshape correctly (needed for GH normalization)
        xs = df['X_coords'].values
        ys = df['Y_coords'].values
        infer = infer_grid_from_coords_simple(xs, ys)
        if infer is None: return None
        nx, ny, _, _, _ = infer

        # 2. Get Normalized INPUTS via gh_to_fno (The Source of Truth)
        gh_data = {c: df[c].tolist() for c in cols}
        # This returns tensor [1, C, H, W]
        # Channels: ['SDF','Bldg_height','Z_relative','U_over_Uref','X_local','Y_local','dir_sin','dir_cos']
        X_tensor, ch_names = build_input_tensor_from_gh(gh_data, H=ny, W=nx)
        X_numpy = X_tensor.squeeze(0).numpy() # [C, H, W]
        
        # Flatten spatial dims to [C, N]
        X_flat = X_numpy.reshape(len(ch_names), -1)

        # 3. Compute Normalized TARGET (Replicating train_fno_distributed.py)
        target_valid = np.array([])
        
        # logic to find mag
        mag_vals = None
        for c in ['mag_U_dimensionless','mag_U','mag_dimensionless']:
            if c in df.columns:
                mag_vals = df[c].to_numpy().astype(float)
                break
        
        # Fallback to components
        if mag_vals is None:
             if all(cc in df.columns for cc in ['Ux_dimensionless','Uy_dimensionless','Uz_dimensionless']):
                 ux = df['Ux_dimensionless'].values.astype(float)
                 uy = df['Uy_dimensionless'].values.astype(float)
                 uz = df['Uz_dimensionless'].values.astype(float)
                 mag_vals = np.sqrt(ux**2 + uy**2 + uz**2)
        
        if mag_vals is not None:
            uref = df['U_over_Uref'].values.astype(float)
            # Filter valid
            mask = np.isfinite(mag_vals) & (uref > 0)
            if np.any(mask):
                m_v = mag_vals[mask]
                u_v = uref[mask]
                # Target formula: (mag - uref) / uref
                delta = (m_v - u_v) / (u_v + 1e-6)
                target_valid = delta

        # 4. Compute Histograms per file
        hists = {}
        
        # Inputs
        for i, name in enumerate(ch_names):
            vals = X_flat[i, :]
            # Remove NaNs if any (padding is 0, handled)
            vals = vals[np.isfinite(vals)]
            
            # Use appropriate bins
            bins = BIN_EDGES
            if name in ['SDF', 'Bldg_height', 'Z_relative']:
                bins = BIN_EDGES_01
            
            counts, _ = np.histogram(vals, bins=bins)
            hists[name] = counts

        # Target
        if len(target_valid) > 0:
            counts, _ = np.histogram(target_valid, bins=BIN_EDGES)
            hists['Target_Delta'] = counts
            
        return hists

    except Exception as e:
        # print(f"Error {fp}: {e}")
        return None

def main():
    # Detect data folder - prioritize ICE path if checking on server
    data_folder = "train_csv" # Default local
    
    # Check known paths
    paths = [
        "/home/hice1/ikaradag3/scratch/FNO/Training_Dataset", # ICE
        "C:/LabShare/Dataset/FormFluxCases/Compressed/Training_Dataset", # Local Windows
        "train_csv" # Fallback relative
    ]
    
    for p in paths:
        if os.path.exists(p):
            data_folder = p
            print(f"Dataset found at: {data_folder}")
            break
            
    files = sorted(glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True))
    if not files:
        print("No files found. Exiting.")
        return

    # Sample for speed if needed, or run all for accuracy
    # files = files[:1000] 
    print(f"Processing {len(files)} files histograms...")

    pool_size = min(32, cpu_count())
    with Pool(pool_size) as pool:
        results = list(tqdm(pool.imap(get_histograms_for_file, files), total=len(files)))

    # Aggregator
    agg_hists = {} # name -> counts array

    print("Aggregating results...")
    for res in results:
        if res is None: continue
        for k, counts in res.items():
            if k not in agg_hists:
                agg_hists[k] = np.zeros_like(counts)
            agg_hists[k] += counts

    # Plotting
    print("Generating plots...")
    
    # Names of normalized inputs + Target
    all_keys = list(agg_hists.keys())
    all_keys.sort()
    
    # Arrange in a grid
    n_plots = len(all_keys)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten()
    
    for i, k in enumerate(all_keys):
        ax = axes[i]
        counts = agg_hists[k]
        
        # Determine bins used
        bins = BIN_EDGES
        if k in ['SDF', 'Bldg_height', 'Z_relative']:
            bins = BIN_EDGES_01
            
        centers = (bins[:-1] + bins[1:]) / 2
        
        # Normalize to probability density for readability
        total = counts.sum()
        if total > 0:
            density = counts / total
        else:
            density = counts

        ax.bar(centers, density, width=(bins[1]-bins[0]), align='center', alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add stats
        # Reconstruct mean/std roughly from histogram
        mean_approx = np.average(centers, weights=counts) if total > 0 else 0
        var_approx = np.average((centers - mean_approx)**2, weights=counts) if total > 0 else 0
        std_approx = np.sqrt(var_approx)
        
        ax.set_title(f"{k}\nMean: {mean_approx:.2f}, Std: {std_approx:.2f}")
        ax.set_xlabel("Normalized Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # Add Reference Lines for -1, 0, 1
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(1, color='green', linestyle=':', alpha=0.5)
        ax.axvline(-1, color='green', linestyle=':', alpha=0.5)

    # Hide unused
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    out_file = "dataset_distribution.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved distribution plot to {out_file}")

if __name__ == "__main__":
    main()
