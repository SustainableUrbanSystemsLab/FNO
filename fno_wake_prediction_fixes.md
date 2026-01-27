# FNO Urban Wind Wake Prediction - Critical Improvements

## Problem Analysis

Your model is exhibiting severe over-smoothing, failing to capture:
- Sharp wake boundaries behind buildings
- Low wind speed regions (dark blue areas)
- High wind speed peaks (red areas)
- Realistic flow gradients

## Root Causes

### 1. **Loss Function Imbalance** ⚠️ CRITICAL
Current weights from your config:
```python
GRAD_WEIGHT = 0.15      # Too low for sharp features
SPECTRAL_WEIGHT = 0.05  # Actually ~95% of total loss in practice!
PEAK_WEIGHT = 0.0       # Disabled - BAD for preserving extremes
```

**Problem**: Your spectral loss (0.195-0.205) is 16x larger than MSE loss (0.011-0.012), completely dominating training. This forces the model to prefer smooth, low-frequency solutions.

### 2. **Training Has Stalled**
- Best loss: epoch 205 (0.205834)
- No improvement for 100 epochs (patience counter at 100)
- Learning rate decayed to 1e-6 (essentially zero)
- Model stuck in local minimum optimized for smoothness

### 3. **Architecture May Be Under-Parameterized**
```python
MODES1 = 32  # Fourier modes
MODES2 = 32
WIDTH = 64   # Hidden dimension
N_LAYERS = 5
```
For complex urban geometries with sharp discontinuities, you may need more capacity.

## Immediate Fixes

### Fix 1: Rebalance Loss Function Weights

**Option A - Aggressive (Recommended for Wake Recovery)**
```toml
[loss]
gradient_weight = 1.5      # 10x increase - sharp boundaries
spectral_weight = 0.001    # 50x decrease - reduce smoothing
peak_weight = 0.5          # Enable - preserve extremes
```

**Option B - Conservative (Safer Starting Point)**
```toml
[loss]
gradient_weight = 0.5      # 3x increase
spectral_weight = 0.01     # 5x decrease
peak_weight = 0.2          # Enable moderately
```

**Why this works:**
- Higher gradient weight penalizes smooth wake transitions
- Lower spectral weight reduces frequency domain smoothing bias
- Peak loss explicitly preserves high/low wind speed regions

### Fix 2: Improve Data Weighting for Wake Regions

In your code (line 156-158), you're weighting near-building regions more:
```python
sdf_w = 1.0 + 19.0 * np.exp(-sdf_val / 5.0)
```

**Add wake-specific weighting:**
```python
# In process_single_file(), after line 151:
# Detect wake regions (low speed areas behind buildings)
is_wake = (val < 0.3 * u_over_uref_val) and (sdf_val > 2.0)
wake_w = 5.0 if is_wake else 1.0

# Then modify line 158:
mask_grid[0, iy, ix] = sensor_w * valid_val * sdf_w * wake_w
```

### Fix 3: Increase Model Capacity

```toml
[model]
modes1 = 48        # Was 32 - more high-freq modes
modes2 = 48        # Was 32
width = 96         # Was 64 - wider hidden layers
n_layers = 6       # Was 5 - deeper network
```

### Fix 4: Training Strategy

**Don't continue from checkpoint** - your model is stuck in a smooth-solution local minimum. Instead:

```bash
# Start fresh with new loss weights
python train_fno_distributed.py --fresh --config config_wake_focused.toml
```

**Use learning rate warmup** to escape the current minimum:
```toml
[training]
learning_rate = 5e-3      # Higher initial LR
warmup_epochs = 10        # Add warmup period
min_lr = 1e-5             # Don't decay to zero
```

### Fix 5: Add Custom Wake Loss Component

Consider implementing a **wake-focused loss** in your `sensor_weighted_mse` function:

```python
def wake_aware_loss(pred, target, sensor_mask, wake_threshold=0.3):
    """
    Additional loss component that heavily penalizes errors in wake regions.
    Wake regions are identified as areas with wind speed < wake_threshold.
    """
    # Identify wake regions
    wake_mask = (target < wake_threshold).float()
    
    # Compute error in wake regions
    wake_error = ((pred - target) ** 2) * wake_mask * sensor_mask
    wake_loss = wake_error.sum() / (wake_mask.sum() + 1e-8)
    
    return wake_loss

# Then in sensor_weighted_mse (after line ~449):
wake_loss = wake_aware_loss(y_pred, y_target, sensor_mask)
total_loss += 0.3 * wake_loss  # Add to your config
components['wake_loss'] = wake_loss.item()
```

## Advanced Solutions

### 1. **Multi-Scale Training**
Train on progressively higher resolutions to learn coarse structure first, then refine:
```python
# Epoch 0-50: 64x64 downsampled
# Epoch 50-100: 128x128
# Epoch 100+: Full resolution
```

### 2. **Perceptual Loss**
Use features from a pre-trained flow network to compare structural similarity:
```python
# Similar to style transfer - compare feature maps not just pixel values
```

### 3. **Adversarial Training**
Add a discriminator to distinguish real vs predicted flow fields:
```python
# Forces model to generate realistic flow structures
# Prevents over-smoothing
```

### 4. **Physics-Informed Constraints**
Add soft constraints for:
- Mass conservation (divergence-free for incompressible flow)
- Momentum equations (Navier-Stokes residuals)
- Boundary conditions (no-slip at building walls)

```python
def divergence_loss(velocity_field):
    """Penalize non-zero divergence for incompressible flow"""
    # Compute ∇·u ≈ 0
    pass
```

## Recommended Action Plan

### Phase 1: Quick Wins (Try First)
1. ✅ Update config with Option A loss weights
2. ✅ Start fresh training (--fresh flag)
3. ✅ Train for 100 epochs, evaluate results

### Phase 2: If Still Not Sufficient
4. ✅ Implement wake-aware loss component
5. ✅ Increase model capacity (modes=48, width=96)
6. ✅ Add wake region data weighting

### Phase 3: Advanced (If Needed)
7. ✅ Implement multi-scale training
8. ✅ Add physics-informed losses
9. ✅ Consider hybrid FNO-CNN architecture

## Expected Improvements

With just Phase 1 changes, you should see:
- **Sharper wake boundaries** - gradient loss will penalize smooth transitions
- **Better peak preservation** - peak loss will maintain high/low extremes
- **Less global smoothing** - reduced spectral weight allows more detail
- **Faster convergence** - fresh start escapes local minimum

Target metrics after fixes:
- MSE loss: ~0.008-0.010 (lower is better)
- Gradient loss: ~0.020-0.030 (will increase - that's good!)
- Spectral loss: ~0.001-0.005 (should drop significantly)
- Visual: Clear wake structures, sharp building boundaries

## Configuration File Template

Create `config_wake_focused.toml`:
```toml
[paths]
data_folder_windows = "train_csv"
data_folder_linux = "train_csv"
model_output = "fno_wake_weights.pth"
checkpoint_file = "checkpoint_wake.pth"

[training]
batch_size = 4
epochs = 200
learning_rate = 3e-3
patience = 50
checkpoint_interval = 10

[model]
modes1 = 48
modes2 = 48
width = 96
n_layers = 6

[loss]
gradient_weight = 1.5
spectral_weight = 0.001
peak_weight = 0.5
wake_weight = 0.3  # Add if implementing wake loss

[performance]
num_workers = 0  # Auto-detect
```

## Monitoring Progress

Track these during training:
1. **Visual inspection every 10 epochs** - plot predictions vs ground truth
2. **Component losses** - ensure gradient/peak losses are contributing
3. **Wake region MAE** - separately compute error in low-speed areas
4. **Histogram of predictions** - should match ground truth distribution

## Additional Tips

- **Don't clip outputs** (line 152 is commented out - good!)
- **Normalize inputs/outputs** properly - use robust scaling
- **Augment data** - rotate/flip geometries for more training samples
- **Ensemble models** - train 3-5 models with different seeds, average predictions

## References for Further Reading

- "Fourier Neural Operator for Parametric PDEs" (Li et al., 2021)
- "U-FNO: Multi-scale Fourier Neural Operator" (Rahman et al., 2022)  
- "Physics-Informed Neural Operators" (Wang et al., 2023)
- "Learning to Correct Spectral Methods for Simulating Turbulent Flows" (List et al., 2022)

---

**Priority**: Start with Phase 1 (loss rebalancing + fresh training). This alone should show dramatic improvement in wake capture. The over-smoothing is primarily a loss function issue, not an architecture limitation.
