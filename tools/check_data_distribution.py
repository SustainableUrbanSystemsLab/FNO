import os, sys
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch
import collections

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import logic from codebase to match training exactly
from core.utils.gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple

def get_stats_for_file(fp):
    try:
        df = pd.read_csv(fp)
        rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
        df.rename(columns=rename_map, inplace=True)

        cols = ['SDF','Bldg_height','Z_relative','U_over_Uref','X_coords','Y_coords','dir_sin','dir_cos']
        if any(c not in df.columns for c in cols):
            return None
        
        # 1. Input Features (Raw Values)
        stats = {}
        for c in cols:
            vals = df[c].to_numpy()
            stats[c] = {
                'min': np.min(vals), 
                'max': np.max(vals), 
                'mean': np.mean(vals), 
                'std': np.std(vals)
            }

        # 2. Target Variable (delta_u_normalized)
        # Recreate the target logic from train_fno_distributed.py
        target_val = None
        
        mag_cols_dim = ['mag_U_dimensionless','mag_U','mag_dimensionless']
        mag_vals = None
        for c in mag_cols_dim:
            if c in df.columns:
                mag_vals = df[c].to_numpy().astype(float)
                break
        
        if mag_vals is None and all(cc in df.columns for cc in ['Ux_dimensionless','Uy_dimensionless','Uz_dimensionless']):
             uxs = df['Ux_dimensionless'].to_numpy().astype(float)
             uys = df['Uy_dimensionless'].to_numpy().astype(float)
             uzs = df['Uz_dimensionless'].to_numpy().astype(float)
             mag_vals = np.sqrt(uxs**2 + uys**2 + uzs**2)

        if mag_vals is not None:
            u_over_uref = df['U_over_Uref'].to_numpy().astype(float)
            
            # Mask invalid
            valid_mask = np.isfinite(mag_vals)
            
            # Compute delta
            delta = (mag_vals - u_over_uref) / (u_over_uref + 1e-6)
            
            # Filter only valid
            delta_valid = delta[valid_mask]
            
            if len(delta_valid) > 0:
                stats['Target_Delta'] = {
                    'min': np.min(delta_valid),
                    'max': np.max(delta_valid),
                    'mean': np.mean(delta_valid),
                    'std': np.std(delta_valid),
                    'p01': np.percentile(delta_valid, 1),
                    'p99': np.percentile(delta_valid, 99)
                }

        return stats

    except Exception as e:
        return None

def main():
    # Detect data folder similar to training script
    import sys
    # For now assume Windows path since I'm running locally potentially, or check both
    data_folder = "C:/LabShare/Dataset/FormFluxCases/Compressed/Training_Dataset"
    
    # If folder doesn't exist (e.g. running on agent machine without mount), check local 'train_csv'
    if not os.path.exists(data_folder):
        data_folder = 'train_csv'
        
    print(f"Scanning files in {data_folder}...")
    files = sorted(glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True))
    
    if not files:
        print("No files found!")
        return

    # Limit to subset for speed if too many
    # files = files[:500] 
    print(f"Analyzing {len(files)} files...")

    with Pool(min(16, cpu_count())) as pool:
        results = list(tqdm(pool.imap(get_stats_for_file, files), total=len(files)))

    # Aggregation
    agg_stats = collections.defaultdict(list)
    
    for r in results:
        if r is None: continue
        for k, v in r.items():
            agg_stats[k].append(v)

    print("\n" + "="*60)
    print(f"{'FEATURE':<15} | {'Global MIN':<10} | {'Global MAX':<10} | {'Avg MEAN':<10} | {'Avg STD':<10}")
    print("-" * 60)

    for k in sorted(agg_stats.keys()):
        # Min of mins, Max of maxs, Mean of means (approx), Mean of stds (approx)
        all_mins = [x['min'] for x in agg_stats[k]]
        all_maxs = [x['max'] for x in agg_stats[k]]
        all_means = [x['mean'] for x in agg_stats[k]]
        all_stds = [x['std'] for x in agg_stats[k]]
        
        # Approximate global stats
        g_min = np.min(all_mins)
        g_max = np.max(all_maxs)
        g_mean = np.mean(all_means)
        g_std = np.mean(all_stds)
        
        print(f"{k:<15} | {g_min:<10.4f} | {g_max:<10.4f} | {g_mean:<10.4f} | {g_std:<10.4f}")

    print("\n" + "="*60)
    print("TARGET VARIABLE PERCENTILES (Avg of files)")
    if 'Target_Delta' in agg_stats:
        p01s = [x.get('p01', 0) for x in agg_stats['Target_Delta']]
        p99s = [x.get('p99', 0) for x in agg_stats['Target_Delta']]
        print(f"Target Delta p01: {np.mean(p01s):.4f}")
        print(f"Target Delta p99: {np.mean(p99s):.4f}")
    
if __name__ == "__main__":
    main()
