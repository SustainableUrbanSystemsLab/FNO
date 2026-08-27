"""
Inspect Y.npy to verify training target format.
Uses random sampling to handle large arrays quickly.

Usage:
    uv run python tools/inspect_ynpy.py
"""
import os, sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

DATA_PATHS = [
    "/home/hice1/athach7/scratch/Training_Dataset", # SCRATCH (Priority)
    "/storage/ice1/2/4/athach7/Training_Dataset", # STORAGE
    "./train_csv",
]

DATA = None
for p in DATA_PATHS:
    if os.path.exists(p):
        DATA = p; break

if DATA is None:
    print("ERROR: Could not find any data folders"); sys.exit(1)

print(f"Inspecting data in: {DATA}")

x_npy = os.path.join(DATA, "X.npy")
y_npy = os.path.join(DATA, "Y.npy")

if os.path.exists(x_npy) and os.path.exists(y_npy):
    print("Found mmap'd X.npy and Y.npy")
    X = np.load(x_npy, mmap_mode='r')
    Y = np.load(y_npy, mmap_mode='r')
    N = X.shape[0]
    print(f"\n=== Shapes ===\n  X: {X.shape}\n  Y: {Y.shape}")
    
    rng = np.random.default_rng(42)
    idx = rng.choice(N, min(50, N), replace=False)
    x_sample, y_sample = np.array(X[idx]), np.array(Y[idx])
else:
    print("Searching for individual .npz files...")
    import glob
    npz_files = sorted(glob.glob(os.path.join(DATA, "**/*.npz"), recursive=True))
    if not npz_files:
        print(f"ERROR: No .npy or .npz files found in {DATA}"); sys.exit(1)
    
    print(f"Found {len(npz_files)} .npz files")
    N = len(npz_files)
    rng = np.random.default_rng(42)
    idx = rng.choice(N, min(50, N), replace=False)
    
    Xs, Ys = [], []
    for i in idx:
        with np.load(npz_files[i]) as data:
            Xs.append(data['X']); Ys.append(data['Y'])
    x_sample, y_sample = np.stack(Xs), np.stack(Ys)

print(f"\n=== X Channels (normalized) ===")
ch_names = ['SDF', 'Bldg_height', 'Z_relative', 'U_over_Uref', 'X_local', 'Y_local', 'dir_sin', 'dir_cos']
for i, name in enumerate(ch_names):
    c = x_sample[:, i]
    print(f"  ch{i} {name:15s}:  min={c.min():7.3f}  max={c.max():7.3f}  mean={c.mean():7.3f}  std={c.std():.3f}")

print(f"\n=== Y (Training Target) ===")
y_ch_names = ["U (delta_u)", "k (TKE)", "U_roof (delta_u_roof)", "k_roof (TKE_roof)"]
if y_sample.ndim == 4:
    num_y_ch = y_sample.shape[1]
    print(f"  Found {num_y_ch} target channel(s) in Y.npy:")
    for c in range(num_y_ch):
        c_name = y_ch_names[c] if c < len(y_ch_names) else f"Channel {c}"
        vals = y_sample[:, c]
        print(f"    ch{c} {c_name:22s}: min={vals.min():7.4f}  max={vals.max():7.4f}  mean={vals.mean():7.4f}  std={vals.std():.4f}")
else:
    print(f"  min  = {y_sample.min():.4f}")
    print(f"  max  = {y_sample.max():.4f}")
    print(f"  mean = {y_sample.mean():.4f}")
    print(f"  std  = {y_sample.std():.4f}")

y_ch0 = y_sample[:, 0] if y_sample.ndim == 4 else y_sample
y_flat = y_ch0.flatten()
y_mean = y_flat.mean()
y_min  = y_flat.min()
y_max  = y_flat.max()

print(f"\n=== FORMAT DIAGNOSIS (Target Ch0: U) ===")
if y_min >= 0 and y_max > 1.5:
    print("  *** PROBLEM: Y looks like raw mag_U (range 0 to 2+). ***")
    print("  Expected: delta_u (range ~-1 to +5, mean near 0).")
    print("  FIX: Y.npy needs to be regenerated with delta_u = (mag - U_ref) / U_ref")
elif -0.3 < y_mean < 0.5 and y_min < -0.05:
    print("  OK: Y is in delta_u format (mean near 0, negative values present).")
else:
    print(f"  UNCERTAIN: mean={y_mean:.3f}, min={y_min:.3f}, max={y_max:.3f}")
    print("  Expected delta_u: mean ~0.0, min ~-1.0, max ~5.0")

# Wake distribution
sample_n = len(idx)
near_zero   = float((np.abs(y_flat) < 0.05).mean())
mild_wake   = float((y_flat < -0.2).mean())
deep_wake   = float((y_flat < -0.5).mean())
accel       = float((y_flat > 0.5).mean())

print(f"\n=== WAKE DISTRIBUTION (from {sample_n} samples) ===")
print(f"  Near-zero  |delta| < 0.05:  {near_zero:.1%}  <- drives model to predict 0")
print(f"  Mild wake  delta   < -0.20: {mild_wake:.1%}")
print(f"  Deep wake  delta   < -0.50: {deep_wake:.1%}")
print(f"  Accel zone delta   > +0.50: {accel:.1%}")

if near_zero > 0.60:
    needed = round(near_zero / max(mild_wake, 1e-4), 1)
    print(f"\n  *** WARNING: {near_zero:.0%} of pixels are near-zero. ***")
    print(f"  MSE will drive the model to predict 0 everywhere (flat predictions).")
    print(f"  Recommended wake_weight >= {needed:.0f} to counteract class imbalance.")

print(f"\nDone.")
