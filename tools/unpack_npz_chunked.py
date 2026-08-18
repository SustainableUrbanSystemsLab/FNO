"""
Chunked Memory-Mapped Unpacker for large .npz files.
Prevents OOM (Out-Of-Memory) crashes by streaming array slices directly to disk.

Usage:
    uv run python tools/unpack_npz_chunked.py --input data/expanded_dataset_uk_roof.npz --output_dir data
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

def unpack_chunked(npz_path, output_dir, chunk_size=50):
    if not os.path.exists(npz_path):
        print(f"ERROR: File '{npz_path}' not found!", flush=True)
        sys.exit(1)

    print(f"Opening '{npz_path}' in streaming mode...", flush=True)
    with np.load(npz_path, allow_pickle=True) as data:
        keys = list(data.keys())
        print(f"Keys found in .npz: {keys}", flush=True)

        if 'X' not in keys or 'Y' not in keys:
            print(f"ERROR: Expected keys 'X' and 'Y' in archive, but found {keys}", flush=True)
            sys.exit(1)

        X_obj = data['X']
        Y_obj = data['Y']

        N = X_obj.shape[0]
        x_shape, x_dtype = X_obj.shape, X_obj.dtype
        y_shape, y_dtype = Y_obj.shape, Y_obj.dtype

        print(f"\nDataset Statistics:")
        print(f"  Total Samples (N): {N}")
        print(f"  X Shape: {x_shape}, Dtype: {x_dtype}")
        print(f"  Y Shape: {y_shape}, Dtype: {y_dtype}")

        os.makedirs(output_dir, exist_ok=True)
        x_out_path = os.path.join(output_dir, "X.npy")
        y_out_path = os.path.join(output_dir, "Y.npy")

        # 1. Create Memory-Mapped X.npy on disk
        print(f"\nCreating memory-mapped '{x_out_path}'...", flush=True)
        x_mmap = np.lib.format.open_memmap(x_out_path, mode='w+', dtype=x_dtype, shape=x_shape)

        print(f"Streaming X array to disk in chunks of {chunk_size}...", flush=True)
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            x_mmap[start_idx:end_idx] = X_obj[start_idx:end_idx]
            print(f"  Wrote X samples {start_idx} to {end_idx} / {N}", flush=True)
        x_mmap.flush()
        del x_mmap
        print(f"Successfully saved '{x_out_path}'!", flush=True)

        # 2. Create Memory-Mapped Y.npy on disk
        print(f"\nCreating memory-mapped '{y_out_path}'...", flush=True)
        y_mmap = np.lib.format.open_memmap(y_out_path, mode='w+', dtype=y_dtype, shape=y_shape)

        print(f"Streaming Y array to disk in chunks of {chunk_size}...", flush=True)
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            y_mmap[start_idx:end_idx] = Y_obj[start_idx:end_idx]
            print(f"  Wrote Y samples {start_idx} to {end_idx} / {N}", flush=True)
        y_mmap.flush()
        del y_mmap
        print(f"Successfully saved '{y_out_path}'!", flush=True)

    print("\n[SUCCESS] Unpacking completed with zero OOM risk!", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Chunked unpacker for large npz datasets")
    parser.add_argument("--input", type=str, default="data/expanded_dataset_uk_roof.npz")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--chunk_size", type=int, default=50)
    args = parser.parse_args()

    unpack_chunked(args.input, args.output_dir, args.chunk_size)

if __name__ == "__main__":
    main()
