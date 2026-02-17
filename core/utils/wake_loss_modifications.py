"""
Modified loss function additions for fno2d_model.py

Add these functions to your fno2d_model.py file to implement wake-aware training.
This will help the model focus on correctly predicting low-speed wake regions.
"""

import torch
import torch.nn.functional as F


def compute_wake_loss(y_pred, y_target, sensor_mask, wake_threshold=0.3):
    """
    Compute additional loss for wake regions (low wind speed areas).
    
    Wake regions are where flow separation occurs behind buildings,
    creating the dark blue low-speed zones in your visualization.
    
    Args:
        y_pred: Model predictions [B, C, H, W]
        y_target: Ground truth [B, C, H, W]
        sensor_mask: Weighting mask [B, C, H, W]
        wake_threshold: Wind speed threshold to identify wakes
    
    Returns:
        wake_loss: Scalar loss value
    """
    # Identify wake regions (normalized velocity < threshold)
    wake_mask = (y_target < wake_threshold).float()
    
    # Compute squared error in wake regions
    wake_error = ((y_pred - y_target) ** 2) * wake_mask * sensor_mask
    
    # Average over wake pixels only
    wake_loss = wake_error.sum() / (wake_mask.sum() + 1e-8)
    
    return wake_loss


def compute_peak_loss(y_pred, y_target, sensor_mask, percentile=90):
    """
    Enhanced peak loss that focuses on BOTH extremes:
    - High wind speeds (acceleration around buildings)
    - Low wind speeds (wake regions)
    
    Args:
        y_pred: Model predictions [B, C, H, W]
        y_target: Ground truth [B, C, H, W]
        sensor_mask: Weighting mask [B, C, H, W]
        percentile: Percentile threshold for identifying peaks
    
    Returns:
        peak_loss: Scalar loss value
    """
    # Flatten for percentile computation
    flat_target = y_target.flatten()
    flat_pred = y_pred.flatten()
    flat_mask = sensor_mask.flatten()
    
    # Get high and low thresholds
    high_threshold = torch.quantile(flat_target, percentile / 100.0)
    low_threshold = torch.quantile(flat_target, (100 - percentile) / 100.0)
    
    # Create masks for extreme regions
    high_mask = (y_target >= high_threshold).float()
    low_mask = (y_target <= low_threshold).float()
    extreme_mask = torch.maximum(high_mask, low_mask)
    
    # Compute error in extreme regions
    extreme_error = ((y_pred - y_target) ** 2) * extreme_mask * sensor_mask
    peak_loss = extreme_error.sum() / (extreme_mask.sum() + 1e-8)
    
    return peak_loss


def sensor_weighted_mse(y_pred, y_target, sensor_mask=None,
                        grad_weight=0.0, spectral_weight=0.0, 
                        peak_weight=0.0, wake_weight=0.0,
                        return_components=False):
    """
    MODIFIED VERSION - Add wake_weight parameter
    
    Multi-component loss for FNO training:
    1. MSE loss (pixel-wise)
    2. Gradient loss (spatial derivatives)
    3. Spectral loss (frequency domain)
    4. Peak loss (extremes preservation)
    5. Wake loss (low-speed regions) - NEW!
    
    Args:
        y_pred: Predictions [B, C, H, W]
        y_target: Ground truth [B, C, H, W]
        sensor_mask: Spatial weighting mask [B, C, H, W]
        grad_weight: Weight for gradient loss
        spectral_weight: Weight for spectral loss
        peak_weight: Weight for peak loss
        wake_weight: Weight for wake loss (NEW)
        return_components: Return dict of individual losses
    
    Returns:
        total_loss: Weighted sum of all components
        components: Dict of individual loss values (if return_components=True)
    """
    
    # Default mask
    if sensor_mask is None:
        sensor_mask = torch.ones_like(y_pred)
    
    # === 1. MSE Loss (Base) ===
    mse_loss = (sensor_mask * (y_pred - y_target) ** 2).sum() / sensor_mask.sum()
    
    # === 2. Gradient Loss (Spatial) ===
    gradient_loss = torch.tensor(0.0, device=y_pred.device)
    if grad_weight > 0:
        # Compute gradients in x and y directions
        pred_grad_x = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
        pred_grad_y = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        
        tgt_grad_x = y_target[:, :, :, 1:] - y_target[:, :, :, :-1]
        tgt_grad_y = y_target[:, :, 1:, :] - y_target[:, :, :-1, :]
        
        # Match shapes for masking
        mask_x = sensor_mask[:, :, :, 1:]
        mask_y = sensor_mask[:, :, 1:, :]
        
        # L2 loss on gradients
        grad_loss_x = (mask_x * (pred_grad_x - tgt_grad_x) ** 2).sum() / mask_x.sum()
        grad_loss_y = (mask_y * (pred_grad_y - tgt_grad_y) ** 2).sum() / mask_y.sum()
        
        gradient_loss = grad_loss_x + grad_loss_y
    
    # === 3. Spectral Loss (Frequency Domain) ===
    spectral_loss = torch.tensor(0.0, device=y_pred.device)
    if spectral_weight > 0:
        # FFT of predictions and targets
        pred_fft = torch.fft.rfft2(y_pred, norm='ortho')
        tgt_fft = torch.fft.rfft2(y_target, norm='ortho')
        
        # L2 loss in frequency domain
        spectral_loss = torch.mean(torch.abs(pred_fft - tgt_fft) ** 2)
    
    # === 4. Peak Loss (Extremes) ===
    peak_loss = torch.tensor(0.0, device=y_pred.device)
    if peak_weight > 0:
        peak_loss = compute_peak_loss(y_pred, y_target, sensor_mask)
    
    # === 5. Wake Loss (Low-Speed Regions) - NEW! ===
    wake_loss = torch.tensor(0.0, device=y_pred.device)
    if wake_weight > 0:
        wake_loss = compute_wake_loss(y_pred, y_target, sensor_mask)
    
    # === Total Loss ===
    total_loss = (mse_loss + 
                  grad_weight * gradient_loss + 
                  spectral_weight * spectral_loss + 
                  peak_weight * peak_loss +
                  wake_weight * wake_loss)
    
    if return_components:
        components = {
            'mse_loss': mse_loss.item(),
            'gradient_loss': gradient_loss.item(),
            'spectral_loss': spectral_loss.item(),
            'peak_loss': peak_loss.item(),
            'wake_loss': wake_loss.item(),
        }
        return total_loss, components
    
    return total_loss


# ============================================================================
# ALTERNATIVE: Enhanced Data Weighting (Add to process_single_file)
# ============================================================================
"""
Add this code to train_fno_distributed.py in the process_single_file function
after line 151 where delta_u_normalized is computed:

```python
# Detect wake regions (low normalized velocity behind buildings)
is_wake = (val < 0.3 * u_over_uref_val) and (sdf_val > 2.0 and sdf_val < 20.0)
is_acceleration = (val > 1.5 * u_over_uref_val) and (sdf_val > 0.5)

# Additional weighting for critical regions
wake_weight = 5.0 if is_wake else 1.0
accel_weight = 3.0 if is_acceleration else 1.0
critical_weight = wake_weight * accel_weight

# Modify mask computation (replace line 158)
mask_grid[0, iy, ix] = sensor_w * valid_val * sdf_w * critical_weight
```

This gives 5x weight to wake regions and 3x weight to acceleration zones,
helping the model focus on the areas you care most about.
"""


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
1. Update fno2d_model.py:
   - Replace the sensor_weighted_mse function with the version above
   - Add compute_wake_loss and compute_peak_loss functions

2. Update config_wake_focused.toml:
   - Set wake_weight = 0.3 in [loss] section

3. Update train_fno_distributed.py:
   - Add WAKE_WEIGHT loading from config (around line 60):
     WAKE_WEIGHT = config.get('loss', {}).get('wake_weight', 0.0)
   
   - Modify loss computation (around line 445):
     loss, components = sensor_weighted_mse(
         pred, yb, sensor_mask=mb,
         grad_weight=GRAD_WEIGHT,
         spectral_weight=SPECTRAL_WEIGHT,
         peak_weight=PEAK_WEIGHT,
         wake_weight=WAKE_WEIGHT,  # ADD THIS
         return_components=True
     )
   
   - Add wake_loss to logging (around line 437):
     running_wake = 0.0
   
   - Track wake loss (around line 459):
     running_wake += components.get('wake_loss', 0.0) * batch_size
   
   - Aggregate wake loss for distributed (around line 474):
     running_tensor = torch.tensor([running, running_mse, running_grad, 
                                    running_spec, running_peak, running_wake], device=device)
   
   - Log wake loss (around line 525):
     logger.log_epoch(epoch, {
         'total_loss': avg_loss,
         'mse_loss': avg_mse,
         'gradient_loss': avg_grad,
         'spectral_loss': avg_spec,
         'peak_loss': avg_peak,
         'wake_loss': avg_wake,  # ADD THIS
         ...
     })

4. Train with new config:
   python train_fno_distributed.py --fresh --config config_wake_focused.toml

5. Monitor training:
   - wake_loss should be ~0.01-0.05
   - gradient_loss should increase (0.05-0.10) - this is good!
   - spectral_loss should decrease (0.001-0.005)
   - mse_loss should decrease (0.005-0.010)
"""
