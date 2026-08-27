"""
Per-channel normalization for the TKE (k) target channels.

Y went from 1 channel (mag_U only) to 4 channels: [U, k, U_roof, k_roof].
The U/U_roof channels are converted to delta_u = (mag - U_ref)/U_ref and
clipped to [-2, 10] in NpyDataset._remap_to_fno, which already keeps them on
a bounded, comparable scale. k/k_roof (turbulent kinetic energy) are kept
raw and are on a very different scale, so a shared MSE across all 4 channels
weights them very unevenly. These constants z-score k/k_roof so every
channel contributes comparably to the loss.

Stats computed via scripts/streaming pass over data/expanded_dataset_uk_roof.npz
(8000 samples), restricted to each channel's valid region (u_ref > 1e-6 for
k, building footprint for k_roof) -- see the analysis in experiments_log.md /
the "adding 3 new y values" investigation.
"""

K_MEAN = 0.036822
K_STD = 0.030528
K_ROOF_MEAN = 0.052280
K_ROOF_STD = 0.035350
STATS_SOURCE = (
    "data/expanded_dataset_uk_roof.npz, full 8000-sample streaming pass "
    "(k: n=1,601,993,823 valid px where u_ref>1e-6; "
    "k_roof: n=78,849,623 valid px where bldg_h>0)"
)


def normalize_k_channels(y):
    """In-place z-score normalization of the k / k_roof channels of a
    (..., C, H, W) numpy target array. No-op for channels that don't exist
    (C <= 1) or that aren't k-like (C == 2 or 3 without a roof pair handled
    by the caller -- only indices 1 and 3 are touched)."""
    if y.shape[-3] > 1:
        y[..., 1, :, :] = (y[..., 1, :, :] - K_MEAN) / K_STD
    if y.shape[-3] > 3:
        y[..., 3, :, :] = (y[..., 3, :, :] - K_ROOF_MEAN) / K_ROOF_STD
    return y


def denormalize_k_channels(y):
    """Inverse of normalize_k_channels, for turning model output / targets
    back into physical TKE units at inference time."""
    if y.shape[-3] > 1:
        y[..., 1, :, :] = y[..., 1, :, :] * K_STD + K_MEAN
    if y.shape[-3] > 3:
        y[..., 3, :, :] = y[..., 3, :, :] * K_ROOF_STD + K_ROOF_MEAN
    return y
