import numpy as np
import torch

def infer_grid_from_coords_simple(xs, ys, tol=1e-6):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    kx = np.round(xs / tol).astype(int)
    ky = np.round(ys / tol).astype(int)
    ux = np.unique(kx); uy = np.unique(ky)
    nx = len(ux); ny = len(uy)
    # if nx * ny != len(xs):
    #     return None
    ux_sorted = np.sort(ux); uy_sorted = np.sort(uy)
    key_to_ix = {k:i for i,k in enumerate(ux_sorted)}
    key_to_iy = {k:i for i,k in enumerate(uy_sorted)}
    idx_map = [(key_to_iy[kyv], key_to_ix[kxv]) for kxv, kyv in zip(kx, ky)]
    xs_sorted_vals = np.array([np.mean(xs[kx==k]) for k in ux_sorted])
    ys_sorted_vals = np.array([np.mean(ys[ky==k]) for k in uy_sorted])
    return nx, ny, xs_sorted_vals, ys_sorted_vals, idx_map

def build_input_tensor_from_gh(gh_outputs, H=None, W=None, include_U_ref_channel=False, U_ref_scalar=None, dtype=np.float32, device='cpu'):
    required = ['SDF','Bldg_height','Z_relative','U_at_z','X_coords','Y_coords','dir_sin','dir_cos']
    for k in required:
        if k not in gh_outputs:
            raise ValueError(f"Missing GH output '{k}'")
    N = len(gh_outputs['SDF'])
    if not all(len(gh_outputs[k]) == N for k in required):
        raise ValueError("All lists must have same length")
    xs = gh_outputs['X_coords']; ys = gh_outputs['Y_coords']
    infer = infer_grid_from_coords_simple(xs, ys)
    if infer is not None:
        nx, ny, xs_vals, ys_vals, idx_map = infer
        W_infer = nx; H_infer = ny
    else:
        if H is None or W is None:
            raise ValueError("Grid inference failed. Provide H and W") 
        H_infer = H; W_infer = W
        coords = np.vstack([ys, xs]).T
        order = np.lexsort((coords[:,1], -coords[:,0]))
        idx_map = [None]*N
        for i, oi in enumerate(order):
            iy = i // W_infer
            ix = i % W_infer
            idx_map[oi] = (iy, ix)
    H_grid = H_infer; W_grid = W_infer
    channels = []
    ch_names = ['SDF','Bldg_height','Z_relative','U_at_z','X_coords','Y_coords','dir_sin','dir_cos']
    for _ in ch_names:
        channels.append(np.full((H_grid, W_grid), np.nan, dtype=dtype))
    for pt_idx, (iy, ix) in enumerate(idx_map):
        channels[0][iy, ix] = float(gh_outputs['SDF'][pt_idx])
        channels[1][iy, ix] = float(gh_outputs['Bldg_height'][pt_idx])
        channels[2][iy, ix] = float(gh_outputs['Z_relative'][pt_idx])
        channels[3][iy, ix] = float(gh_outputs['U_at_z'][pt_idx])  # user provides dimensionless U_at_z
        channels[4][iy, ix] = float(gh_outputs['X_coords'][pt_idx])
        channels[5][iy, ix] = float(gh_outputs['Y_coords'][pt_idx])
        channels[6][iy, ix] = float(gh_outputs['dir_sin'][pt_idx])
        channels[7][iy, ix] = float(gh_outputs['dir_cos'][pt_idx])
    if include_U_ref_channel:
        if U_ref_scalar is None:
            raise ValueError("U_ref_scalar required")
        channels.append(np.full((H_grid, W_grid), float(U_ref_scalar), dtype=dtype))
    channel_stack = np.stack([np.nan_to_num(ch, nan=0.0) for ch in channels], axis=0)
    X = torch.from_numpy(channel_stack.astype(dtype)).unsqueeze(0).to(device)
    return X, ch_names + (['U_ref'] if include_U_ref_channel else [])
