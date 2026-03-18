"""
Inspect Y.npy to verify training target format.
Run this on the cluster to debug flat predictions.

Usage:
    uv run python tools/inspect_ynpy.py
"""
import os, sys, numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

DATA_PATHS = [
    "/storage/ice1/2/4/athach7/Training_Dataset",
    "./train_csv",
]

for p in DATA_PATHS:
    x_path = os.path.join(p, "X.npy")
    y_path = os.path.join(p, "Y.npy")
    if os.path.exists(x_path) and os.path.exists(y_path):
        DATA = p
        break
else:
    print("ERROR: Could not find X.npy and Y.npy in known locations.")
    sys.exit(1)

print(f"Loading from: {DATA}")
X = np.load(x_path, mmap_mode='r')
Y = np.load(y_path, mmap_mode='r')

print(f"\n=== X.npy (Inputs) ===")
print(f"  Shape:  {X.shape}  (samples, channels, H, W)")
print(f"  Range:  [{X.min():.4f}, {X.max():.4f}]")
print(f"  Channel 0 (SDF):          min={X[:,0].min():.3f}  max={X[:,0].max():.3f}  mean={X[:,0].mean():.3f}")
print(f"  Channel 3 (U_over_Uref):  min={X[:,3].min():.3f}  max={X[:,3].max():.3f}  mean={X[:,3].mean():.3f}")

print(f"\n=== Y.npy (Targets) ===")
print(f"  Shape:  {Y.shape}  (samples, 1, H, W)")
print(f"  Min:    {Y.min():.4f}")
print(f"  Max:    {Y.max():.4f}")
print(f"  Mean:   {Y.mean():.4f}")
print(f"  Std:    {Y.std():.4f}")

# Check format
y_mean = float(Y.mean())
y_min  = float(Y.min())
y_max  = float(Y.max())

print(f"\n=== FORMAT DIAGNOSIS ===")
if y_min >= 0 and y_max > 1.0:
    print("  *** PROBLEM: Y looks like raw mag_U (0 to 2+), NOT delta_u! ***")
    print("  the model is trained on mag_U but inference treats output as delta_u.")
    print("  This WILL cause flat predictions. You need to regenerate Y.npy.")
elif -0.1 < y_mean < 0.3 and y_min < -0.1:
    print("  OK: Y appears to be in delta_u format (mean near 0, has negatives for wakes)")
else:
    print(f"  UNCERTAIN: Mean={y_mean:.3f}, Min={y_min:.3f}, Max={y_max:.3f}")
    print("  Expected delta_u: mean ~0, min ~-1, max ~5")

# Fraction that are wakes
wake_frac = float((Y < -0.2).mean())
deep_wake_frac = float((Y < -0.5).mean())
near_zero_frac = float((np.abs(Y) < 0.05).mean())
print(f"\n=== WAKE DISTRIBUTION ===")
print(f"  Near-zero (|delta_u| < 0.05): {near_zero_frac:.1%}  <- This is why MSE favors flat predictions")
print(f"  Mild wake  (delta_u < -0.20): {wake_frac:.1%}")
print(f"  Deep wake  (delta_u < -0.50): {deep_wake_frac:.1%}")

if near_zero_frac > 0.6:
    print(f"\n  *** WARNING: {near_zero_frac:.0%} of training pixels are near-zero.")
    print(f"      MSE will drive the model to predict 0 everywhere.")
    print(f"      wake_weight needs to be MUCH higher to counteract this.")
    recommended_wake_weight = round(near_zero_frac / (wake_frac + 1e-6), 1)
    print(f"      Recommended wake_weight >= {recommended_wake_weight:.1f}")
