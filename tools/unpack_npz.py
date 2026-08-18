"""
Unpack .npz dataset into X.npy and Y.npy with progress status prints.

Usage:
    uv run python tools/unpack_npz.py --input data/expanded_dataset_uk_roof.npz --output_dir data
"""
import os
import sys
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Unpack .npz dataset to X.npy and Y.npy")
    parser.add_argument("--input", type=str, default="data/expanded_dataset_uk_roof.npz", help="Path to input .npz file")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for X.npy and Y.npy")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file '{args.input}' not found!", flush=True)
        sys.exit(1)

    print(f"Loading '{args.input}'... (this may take a few seconds if file is large)", flush=True)
    with np.load(args.input, allow_pickle=True) as data:
        keys = list(data.keys())
        print(f"Found keys in .npz archive: {keys}", flush=True)

        x_key = 'X' if 'X' in keys else keys[0]
        y_key = 'Y' if 'Y' in keys else (keys[1] if len(keys) > 1 else None)

        print(f"Reading '{x_key}' array into memory...", flush=True)
        X = data[x_key]
        print(f"  X shape: {X.shape}, dtype: {X.dtype}, size: {X.nbytes / (1024**2):.1f} MB", flush=True)

        if y_key:
            print(f"Reading '{y_key}' array into memory...", flush=True)
            Y = data[y_key]
            print(f"  Y shape: {Y.shape}, dtype: {Y.dtype}, size: {Y.nbytes / (1024**2):.1f} MB", flush=True)
        else:
            print("WARNING: No second array key found in .npz!", flush=True)
            Y = None

    os.makedirs(args.output_dir, exist_ok=True)
    x_out_path = os.path.join(args.output_dir, "X.npy")
    y_out_path = os.path.join(args.output_dir, "Y.npy")

    print(f"Saving '{x_out_path}'...", flush=True)
    np.save(x_out_path, X)
    print(f"  Successfully saved {x_out_path}", flush=True)

    if Y is not None:
        print(f"Saving '{y_out_path}'...", flush=True)
        np.save(y_out_path, Y)
        print(f"  Successfully saved {y_out_path}", flush=True)

    print("\nAll done! Ready for training.", flush=True)

if __name__ == "__main__":
    main()
