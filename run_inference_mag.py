import pandas as pd, numpy as np, torch
from gh_to_fno import build_input_tensor_from_gh
from fno2d_model import FNO2d

XLSX = "gh_outputs_2dec.xlsx"   # change as needed
MODEL = "fno_mag_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROUND = 2

df = pd.read_excel(XLSX)
cols = ['SDF','Bldg_height','Z_relative','U_at_z','X_coords','Y_coords','dir_sin','dir_cos']
for c in cols:
    if c not in df.columns:
        raise RuntimeError(f"Missing input column {c} in {XLSX}")
gh = {c:df[c].tolist() for c in cols}
X, chs = build_input_tensor_from_gh(gh, H=None, W=None, device=DEVICE)
in_ch = X.shape[1]
model = FNO2d(in_channels=in_ch, out_channels=1, modes1=20, modes2=20, width=64, n_layers=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
model.eval()
with torch.no_grad():
    mag_star = model(X.to(DEVICE)).cpu().numpy()[0,0]  # dimensionless grid (H,W)

# flatten to original points (attempt grid reshape)
nx = df['X_coords'].nunique(); ny = df['Y_coords'].nunique()
if nx * ny == len(df):
    flat = mag_star.reshape(-1)[:len(df)]
else:
    flat = mag_star.ravel()[:len(df)]

# write dimensionless predictions; if U_ref present also write physical mag
df['mag_U_pred_dimensionless'] = np.round(flat, ROUND)
if 'U_ref' in df.columns:
    Uref = float(df['U_ref'].iloc[0])
    df['mag_U_pred'] = np.round(flat * Uref, ROUND)
out = XLSX.replace('.xlsx', '_mag_pred.xlsx')
df.to_excel(out, index=False)
print('Saved predictions to', out)
