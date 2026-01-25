#!/usr/bin/env python3
"""
Diagnostic script to check grid sizes across all training CSV files.
Run this on PACE to identify files with different dimensions.
"""

import os
import glob
import pandas as pd
from collections import Counter
import tomllib
from tqdm import tqdm

# Load config
config_path = os.path.join(os.path.dirname(__file__), "config.toml")
with open(config_path, 'rb') as f:
    config = tomllib.load(f)

DATA_FOLDER_WIN = config.get('paths', {}).get('data_folder_windows', '')
DATA_FOLDER_LINUX = config.get('paths', {}).get('data_folder_linux', '')

def get_data_folder():
    if os.path.exists(DATA_FOLDER_WIN):
        return DATA_FOLDER_WIN
    elif os.path.exists(DATA_FOLDER_LINUX):
        return DATA_FOLDER_LINUX
    else:
        raise RuntimeError("Data folder not found")

def get_grid_size(fp):
    """Get grid dimensions from a CSV file."""
    try:
        df = pd.read_csv(fp)
        
        # Rename columns for compatibility
        rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
        df.rename(columns=rename_map, inplace=True)
        
        if 'X_coords' not in df.columns or 'Y_coords' not in df.columns:
            return None, "Missing coords"
        
        xs = df['X_coords'].unique()
        ys = df['Y_coords'].unique()
        return (len(ys), len(xs)), None  # (height, width)
    except Exception as e:
        return None, str(e)

def main():
    data_folder = get_data_folder()
    csv_files = sorted(glob.glob(os.path.join(data_folder, '**/*.csv'), recursive=True))
    
    print(f"Scanning {len(csv_files)} CSV files...")
    print("=" * 60)
    
    size_to_files = {}
    errors = []
    
    for fp in tqdm(csv_files, desc="Scanning CSVs", unit="file"):
        size, error = get_grid_size(fp)
        if error:
            errors.append((fp, error))
        else:
            if size not in size_to_files:
                size_to_files[size] = []
            size_to_files[size].append(fp)
    
    # Print summary
    print("\nGrid Size Distribution:")
    print("-" * 40)
    for size, files in sorted(size_to_files.items(), key=lambda x: -len(x[1])):
        print(f"  {size[0]:4d} x {size[1]:4d}  :  {len(files):4d} files")
    
    print(f"\n{len(errors)} files with errors")
    
    # Show sample files for each non-majority size
    majority_size = max(size_to_files.keys(), key=lambda x: len(size_to_files[x]))
    
    print(f"\nMajority size: {majority_size[0]} x {majority_size[1]} ({len(size_to_files[majority_size])} files)")
    
    non_majority = {k: v for k, v in size_to_files.items() if k != majority_size}
    if non_majority:
        print("\n⚠️  Non-majority sizes (these might cause batching issues):")
        for size, files in non_majority.items():
            print(f"\n  Size {size[0]} x {size[1]} ({len(files)} files):")
            for f in files[:5]:  # Show first 5
                print(f"    - {os.path.basename(f)}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")


if __name__ == "__main__":
    main()
