import numpy as np
import torch

def infer_grid_from_coords_simple(xs, ys, tol=1e-6):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    kx = np.round(xs / tol).astype(int)
    ky = np.round(ys / tol).astype(int)
    ux = np.unique(kx); uy = np.unique(ky)
    nx = len(ux); ny = len(uy)
    ux_sorted = np.sort(ux); uy_sorted = np.sort(uy)
    key_to_ix = {k:i for i,k in enumerate(ux_sorted)}
    key_to_iy = {k:i for i,k in enumerate(uy_sorted)}
    idx_map = [(key_to_iy[kyv], key_to_ix[kxv]) for kxv, kyv in zip(kx, ky)]
    xs_sorted_vals = np.array([np.mean(xs[kx==k]) for k in ux_sorted])
    ys_sorted_vals = np.array([np.mean(ys[ky==k]) for k in uy_sorted])
    return nx, ny, xs_sorted_vals, ys_sorted_vals, idx_map

def build_input_tensor_from_gh(gh_outputs, H=None, W=None, include_U_ref_channel=False, U_ref_scalar=None, dtype=np.float32, device='cpu'):
    # Backward compatibility: map U_at_z -> U_over_Uref
    if 'U_at_z' in gh_outputs and 'U_over_Uref' not in gh_outputs:
        gh_outputs['U_over_Uref'] = gh_outputs['U_at_z']

    required = ['SDF','Bldg_height','Z_relative','U_over_Uref','X_coords','Y_coords','dir_sin','dir_cos']
    for k in required:
        if k not in gh_outputs:
            raise ValueError(f"Missing GH output '{k}'")

    N = len(gh_outputs['SDF'])
    xs = np.array(gh_outputs['X_coords'], dtype=float)
    ys = np.array(gh_outputs['Y_coords'], dtype=float)
    hs = np.array(gh_outputs['Bldg_height'], dtype=float)

    # 1. Grid Inference
    infer = infer_grid_from_coords_simple(xs, ys)
    if infer is not None:
        nx, ny, xs_vals, ys_vals, idx_map = infer
    else:
        if H is None or W is None: raise ValueError("Grid inference failed")
        nx, ny = W, H
        idx_map = [(i//nx, i%nx) for i in range(N)]

    # 2. Centering Strategy
    # We use building center to make the model map-independent
    bldg_mask = hs > 0
    if bldg_mask.any():
        x_center = np.mean(xs[bldg_mask])
        y_center = np.mean(ys[bldg_mask])
    else:
        x_center, y_center = np.mean(xs), np.mean(ys)

    # 3. Channel Population
    # Channel layout matches the on-the-fly remap in pipelines/train/distributed.py:
    #   Ch0: SDF / 200
    #   Ch1: Bldg_height / 50
    #   Ch2: Z_relative / 10
    #   Ch3: U_over_Uref * 2
    #   Ch4: X_local / 500  (building-centered)
    #   Ch5: Y_local / 500  (building-centered)
    #   Ch6: dir_sin
    #   Ch7: dir_cos
    ch_names = ['SDF','Bldg_height','Z_relative','U_over_Uref','X_local','Y_local','dir_sin','dir_cos']
    channels = [np.zeros((ny, nx), dtype=dtype) for _ in ch_names]

    for pt_idx, (iy, ix) in enumerate(idx_map):
        channels[0][iy, ix] = float(gh_outputs['SDF'][pt_idx])
        channels[1][iy, ix] = hs[pt_idx]
        channels[2][iy, ix] = float(gh_outputs['Z_relative'][pt_idx])
        channels[3][iy, ix] = float(gh_outputs['U_over_Uref'][pt_idx])
        channels[4][iy, ix] = xs[pt_idx] - x_center
        channels[5][iy, ix] = ys[pt_idx] - y_center
        channels[6][iy, ix] = float(gh_outputs['dir_sin'][pt_idx])
        channels[7][iy, ix] = float(gh_outputs['dir_cos'][pt_idx])

    # 4. Physical Normalization (matches distributed.py _remap_to_fno)
    channels[0] /= 200.0   # SDF
    channels[1] /= 50.0    # Bldg_height
    channels[2] /= 10.0    # Z_relative
    channels[3] *= 2.0     # U_over_Uref
    channels[4] /= 500.0   # X_local
    channels[5] /= 500.0   # Y_local
    # ch6/ch7 already in [-1, 1]

    if include_U_ref_channel:
        if U_ref_scalar is None: raise ValueError("U_ref_scalar required")
        channels.append(np.full((ny, nx), float(U_ref_scalar), dtype=dtype))

    channel_stack = np.stack(channels, axis=0)
    X = torch.from_numpy(channel_stack).unsqueeze(0).to(device)
    return X, ch_names + (['U_ref'] if include_U_ref_channel else [])
