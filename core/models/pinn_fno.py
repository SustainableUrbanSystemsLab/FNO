"""
PINN-FNO: Physics-Informed Neural Operator
===========================================

Combines the spectral expressiveness of Fourier Neural Operators (FNO)
with physics constraints from fluid mechanics (continuity + simplified
Navier-Stokes momentum) enforced directly in the loss function.

Architecture:
  - FNO backbone: learns the spectral/spatial mapping from geometry+conditions -> wind
  - SpatialAttention: focuses on critical regions (wakes, boundaries)
  - Physics loss: penalizes divergence (continuity constraint)
  - Wake loss: extra focus on low-speed deficit regions

How it differs from HybridFNO:
  - The physics constraint replaces the "smoothness" penalty
  - Continuity (∇·u ≈ 0) is enforced to prevent physically impossible predictions
  - The model learns to be consistent across spatial gradients, not just match point-wise MSE
"""

import torch
import torch.nn as nn
from neuralop.models import FNO

from core.models.fno2d import _per_channel_weighted


class SpatialAttention(nn.Module):
    """Spatial attention to focus on wakes and acceleration zones."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(x)


class PINNFNO(nn.Module):
    """
    Physics-Informed FNO (PINN-FNO).

    The model predicts the velocity correction (delta_u) over a baseline
    wind field, conditioned on building geometry (SDF), height, and wind
    direction. Physical constraints are enforced externally via pinn_loss().

    Input channels (8):
        [SDF, Bldg_height, Z_relative, U_over_Uref, X_local, Y_local, dir_sin, dir_cos]

    Output: delta_u (1 channel) — the wind speed correction over baseline.
    """
    def __init__(self, n_modes=(32, 32), in_channels=8, out_channels=1,
                 hidden_channels=64, n_layers=4):
        super().__init__()

        # 1. FNO Backbone
        self.fno = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            use_channel_mlp=True,
            positional_embedding='grid'
        )

        # 2. Spatial attention for wakes
        self.attention = SpatialAttention(hidden_channels)

        # 3. Geometry-aware refinement
        # +2 for SDF and baseline skip connections
        self.refinement = nn.Sequential(
            nn.Conv2d(hidden_channels + 2, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1)
        )

        # Initialize refinement to be "ready" to learn
        for m in self.refinement:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 8, H, W)
        sdf      = x[:, 0:1, :, :]   # Building geometry
        baseline = x[:, 3:4, :, :]   # Baseline wind speed (U_over_Uref)

        feat = self.fno(x)           # Spectral features
        feat = self.attention(feat)  # Focus on wakes

        combined = torch.cat([feat, sdf, baseline], dim=1)
        delta = self.refinement(combined)  # Predict correction
        return delta


# ======================================================================
# Physics-Informed Loss Functions
# ======================================================================

def continuity_loss(u_field, dx=1.0, dy=1.0):
    """
    Enforce 2D continuity (∇·u = 0) using finite differences.

    For steady incompressible 2D flow:
        ∂u/∂x + ∂v/∂y ≈ 0

    Since we only predict a scalar wind speed magnitude (not vector),
    we approximate this as the spatial divergence of the scalar field.
    This penalizes predictions that are NOT smooth or consistent.

    Args:
        u_field: (B, 1, H, W) predicted wind delta
        dx, dy: grid spacing in x and y (meters, default normalized)
    Returns:
        scalar continuity residual loss
    """
    # [CHANNEL SELECTIVE] Ensure we only process the first channel (mag_U)
    if u_field.shape[1] > 1:
        u_field = u_field[:, 0:1]

    # Central differences for gradient
    du_dx = (u_field[:, :, :, 2:] - u_field[:, :, :, :-2]) / (2 * dx)
    du_dy = (u_field[:, :, 2:, :] - u_field[:, :, :-2, :]) / (2 * dy)

    # Divergence: ∂u/∂x + ∂u/∂y (scalar approximation)
    # Trim to matching size
    min_h = min(du_dx.shape[2], du_dy.shape[2])
    min_w = min(du_dx.shape[3], du_dy.shape[3])
    divergence = du_dx[:, :, :min_h, :min_w] + du_dy[:, :, :min_h, :min_w]

    return torch.mean(divergence ** 2)


def momentum_smoothness_loss(u_field):
    """
    Penalize large Laplacians (∇²u), which correspond to physically
    unrealistic sharp isolated spikes in the velocity field.

    This is a soft viscid constraint: the momentum equation predicts
    that velocity gradients are smoothed by viscosity ν∇²u.
    """
    # [CHANNEL SELECTIVE] Focus only on the primary wind speed channel
    if u_field.shape[1] > 1:
        u_field = u_field[:, 0:1]

    # Laplacian via finite differences
    u = u_field[:, :, 1:-1, 1:-1]          # Interior
    u_xp = u_field[:, :, 1:-1, 2:]          # x+1
    u_xm = u_field[:, :, 1:-1, :-2]         # x-1
    u_yp = u_field[:, :, 2:, 1:-1]          # y+1
    u_ym = u_field[:, :, :-2, 1:-1]         # y-1

    laplacian = u_xp + u_xm + u_yp + u_ym - 4 * u

    # Only penalize very large Laplacians (not gentle variations)
    return torch.mean(torch.clamp(laplacian ** 2 - 0.01, min=0.0))


def wake_physics_loss(pred, target, sdf, wake_threshold=-0.2, sdf_threshold=0.05):
    """
    Physics-aware wake loss:
    - Focuses on regions behind buildings (SDF > threshold, meaning NOT inside a building)
    - Targets deep deficits (delta_u < wake_threshold)
    - Penalizes both missed wakes and false wakes

    Args:
        pred:   (B, 1, H, W) predicted delta_u
        target: (B, 1, H, W) ground truth delta_u
        sdf:    (B, 1, H, W) signed distance field (normalized)
        wake_threshold: delta_u value below which we call it a "wake"
        sdf_threshold: sdf above this = outside building = wake region possible
    """
    # [CHANNEL SELECTIVE] Focus only on primary wind speed channel
    if pred.shape[1] > 1:
        pred = pred[:, 0:1]
        target = target[:, 0:1]

    # Wake mask from ground truth in the fluid domain
    exterior = (sdf > sdf_threshold).float()        # Outside buildings
    gt_wake  = (target < wake_threshold).float()    # Strong wind deficit
    wake_mask = exterior * gt_wake                   # Outside AND in a wake

    if wake_mask.sum() < 1:
        return torch.tensor(0.0, device=pred.device)

    # Weighted squared error: penalize more for missed wakes
    error = (pred - target) ** 2
    wake_error = (error * wake_mask).sum() / (wake_mask.sum() + 1e-8)

    # Additional penalty: False positives (model predicts wake where there is none)
    pred_wake   = (pred < wake_threshold).float()
    false_wake  = pred_wake * (1 - gt_wake) * exterior
    false_error = (false_wake * torch.abs(pred - target)).sum() / (false_wake.sum() + 1.0)

    return wake_error + 0.5 * false_error


def pinn_loss(pred, target, x_input, sensor_mask=None,
              grad_weight=2.0, continuity_weight=0.1,
              momentum_weight=0.05, wake_weight=1.0,
              peak_weight=0.5, channel_weights=None):
    """
    Combined PINN-FNO loss:
      L = MSE + grad_weight * L_grad
            + continuity_weight * L_continuity
            + momentum_weight * L_momentum
            + wake_weight * L_wake
            + peak_weight * L_peak

    Args:
        pred:           (B, C, H, W) model output (delta_u, + k/roof channels if C>1)
        target:         (B, C, H, W) ground truth
        x_input:        (B, 8, H, W) the full input tensor (for SDF, dir extraction)
        sensor_mask:    (B, C, H, W) per-channel/per-pixel weighting
        channel_weights: optional per-channel importance, e.g. [1.0, 0.5, 0.5, 0.25]
                         for [U, k, U_roof, k_roof]. See fno2d._per_channel_weighted.
    Returns:
        total_loss (scalar), components (dict)
    """
    if sensor_mask is None:
        sensor_mask = torch.ones_like(pred)

    # 1. Weighted MSE (per-channel-normalized, see fno2d._per_channel_weighted)
    mse = _per_channel_weighted((pred - target) ** 2, sensor_mask, channel_weights)

    # 2. Gradient fidelity (sharp boundaries)
    p_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    p_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    t_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    t_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    m_dx = sensor_mask[:, :, :, 1:]
    m_dy = sensor_mask[:, :, 1:, :]
    grad = (_per_channel_weighted((p_dx - t_dx) ** 2, m_dx, channel_weights)
            + _per_channel_weighted((p_dy - t_dy) ** 2, m_dy, channel_weights))

    # 3. Physics: Continuity (∇·u ≈ 0)
    cont = continuity_loss(pred)

    # 4. Physics: Momentum smoothness (anti-spike)
    mom = momentum_smoothness_loss(pred)

    # 5. Wake loss (geometry-aware)
    sdf_ch = x_input[:, 0:1, :, :]   # SDF channel
    wake = wake_physics_loss(pred, target, sdf_ch)

    # 6. Peak (extreme velocity) loss
    # [CHANNEL SELECTIVE] Focus only on primary wind speed channel, matching
    # continuity/momentum/wake above. The +-0.5 thresholds are calibrated for
    # delta_u (channel 0); applying them to unrelated channels (e.g. raw k)
    # was a latent bug once out_channels grew beyond 1.
    pred_u, target_u = pred, target
    if pred_u.shape[1] > 1:
        pred_u = pred_u[:, 0:1]
        target_u = target_u[:, 0:1]
    hi_thresh = 0.5
    lo_thresh = -0.5
    peak_mask = ((target_u >= hi_thresh) | (target_u <= lo_thresh)).float()
    peak = ((pred_u - target_u) ** 2 * peak_mask).sum() / (peak_mask.sum() + 1e-8)

    total = (mse
             + grad_weight        * grad
             + continuity_weight  * cont
             + momentum_weight    * mom
             + wake_weight        * wake
             + peak_weight        * peak)

    components = {
        'mse_loss':         mse.item(),
        'gradient_loss':    grad.item(),
        'continuity_loss':  cont.item(),
        'momentum_loss':    mom.item(),
        'wake_loss':        wake.item(),
        'peak_loss':        peak.item(),
    }
    return total, components
