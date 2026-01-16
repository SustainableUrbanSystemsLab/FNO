import pandas as pd, numpy as np, torch
from gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple
from fno2d_model import FNO2d

CSV = "gh_outputs_2dec.csv"   # change as needed
MODEL = "fno_mag_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROUND = 2

df = pd.read_csv(CSV)
# Renaming known variations
rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
df.rename(columns=rename_map, inplace=True)

cols = ['SDF','Bldg_height','Z_relative','U_at_z','X_coords','Y_coords','dir_sin','dir_cos']
for c in cols:
    if c not in df.columns:
        raise RuntimeError(f"Missing input column {c} in {CSV}")
gh = {c:df[c].tolist() for c in cols}
X, chs = build_input_tensor_from_gh(gh, H=None, W=None, device=DEVICE)
in_ch = X.shape[1]
model = FNO2d(in_channels=in_ch, out_channels=1, modes1=20, modes2=20, width=64, n_layers=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
model.eval()
with torch.no_grad():
    mag_star = model(X.to(DEVICE)).cpu().numpy()[0,0]  # dimensionless grid (H,W)

# flatten to original points (attempt grid reshape)
# Robustly map predictions back to original points (handles sparse/unsorted)
nx, ny, _, _, idx_map = infer_grid_from_coords_simple(df['X_coords'], df['Y_coords'])
flat = np.array([mag_star[iy, ix] for (iy, ix) in idx_map])

# write dimensionless predictions; if U_ref present also write physical mag
df['mag_U_pred_dimensionless'] = np.round(flat, ROUND)
if 'U_ref' in df.columns:
    Uref = float(df['U_ref'].iloc[0])
    df['mag_U_pred'] = np.round(flat * Uref, ROUND)
out = CSV.replace('.csv', '_mag_pred.csv')
df.to_csv(out, index=False)
print('Saved predictions to', out)
