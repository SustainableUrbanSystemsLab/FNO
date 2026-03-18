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
    "/storage/ice1/2/4/athach7/Training_Dataset",
    "./train_csv",
]

DATA = None
for p in DATA_PATHS:
    if os.path.exists(os.path.join(p, "X.npy")) and os.path.exists(os.path.join(p, "Y.npy")):
        DATA = p
        break

if DATA is None:
    print("ERROR: Could not find X.npy and Y.npy"); sys.exit(1)

print(f"Loading from: {DATA}")
X = np.load(os.path.join(DATA, "X.npy"), mmap_mode='r')
Y = np.load(os.path.join(DATA, "Y.npy"), mmap_mode='r')

N = X.shape[0]
print(f"\n=== Shapes ===")
print(f"  X: {X.shape}  (N, channels, H, W)")
print(f"  Y: {Y.shape}  (N, 1, H, W)")

# Sample a small subset of samples to get stats fast
SAMPLE_N = min(50, N)
rng = np.random.default_rng(42)
idx = rng.choice(N, SAMPLE_N, replace=False)
print(f"\nComputing stats from {SAMPLE_N} random samples (fast)...")

# Load sampled data into RAM
x_sample = np.array(X[idx])   # (50, 8, H, W)
y_sample = np.array(Y[idx])   # (50, 1, H, W)

print(f"\n=== X Channels (normalized) ===")
ch_names = ['SDF', 'Bldg_height', 'Z_relative', 'U_over_Uref', 'X_local', 'Y_local', 'dir_sin', 'dir_cos']
for i, name in enumerate(ch_names):
    c = x_sample[:, i]
    print(f"  ch{i} {name:15s}:  min={c.min():7.3f}  max={c.max():7.3f}  mean={c.mean():7.3f}  std={c.std():.3f}")

print(f"\n=== Y (Training Target) ===")
print(f"  min  = {y_sample.min():.4f}")
print(f"  max  = {y_sample.max():.4f}")
print(f"  mean = {y_sample.mean():.4f}")
print(f"  std  = {y_sample.std():.4f}")

y_flat = y_sample.flatten()
y_mean = y_flat.mean()
y_min  = y_flat.min()
y_max  = y_flat.max()

print(f"\n=== FORMAT DIAGNOSIS ===")
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
near_zero   = float((np.abs(y_flat) < 0.05).mean())
mild_wake   = float((y_flat < -0.2).mean())
deep_wake   = float((y_flat < -0.5).mean())
accel       = float((y_flat > 0.5).mean())

print(f"\n=== WAKE DISTRIBUTION (from {SAMPLE_N} samples) ===")
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
