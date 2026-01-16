# Train FNO to predict dimensionless magnitude (mag_U)
import os, glob, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from fno2d_model import FNO2d, sensor_weighted_mse
from gh_to_fno import build_input_tensor_from_gh


DATA_FOLDER = "./train_csv"
MODEL_OUT = "fno_mag_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 2; EPOCHS = 40; LR = 1e-3
MODES1=20; MODES2=20; WIDTH=64; N_LAYERS=4
FORCE_H = None; FORCE_W = None

def infer_grid(xs, ys, tol=1e-6):
    xs=np.array(xs); ys=np.array(ys)
    kx=np.round(xs/tol).astype(int); ky=np.round(ys/tol).astype(int)
    ux=np.unique(kx); uy=np.unique(ky)
    # if len(ux)*len(uy)!=len(xs): return None
    key_x={k:i for i,k in enumerate(np.sort(ux))}
    key_y={k:i for i,k in enumerate(np.sort(uy))}
    idx=[(key_y[kyv], key_x[kxv]) for kxv,kyv in zip(kx,ky)]
    return len(ux), len(uy), idx

files = sorted(glob.glob(os.path.join(DATA_FOLDER,"*.csv")))
if not files: raise RuntimeError("No training files in " + DATA_FOLDER)
Xs=[]; Ys=[]; Masks=[]
print("Preparing dataset from", len(files), "files...")
for fp in files:
    df = pd.read_csv(fp)
    # Renaming known variations
    rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
    df.rename(columns=rename_map, inplace=True)

    cols = ['SDF','Bldg_height','Z_relative','U_at_z','X_coords','Y_coords','dir_sin','dir_cos']
    if any(c not in df.columns for c in cols):
        raise RuntimeError(fp + " missing input columns")
    infer = infer_grid(df['X_coords'].to_numpy(), df['Y_coords'].to_numpy())
    if infer is None:
        if FORCE_H is None or FORCE_W is None:
            raise RuntimeError("Grid inference failed for " + fp)
        nx, ny = FORCE_W, FORCE_H
        idx_map = [(i//nx, i%nx) for i in range(len(df))]
    else:
        nx, ny, idx_map = infer

    gh_out = {c: df[c].tolist() for c in cols}
    X_tensor, chs = build_input_tensor_from_gh(gh_out, H=ny, W=nx, device='cpu')

    # get mag (dimensionless). accept mag_U or mag_U_dimensionless as already dimensionless
    mag_cols_dim = ['mag_U_dimensionless','mag_U','mag_dimensionless']
    mag_vals = None
    for c in mag_cols_dim:
        if c in df.columns:
            mag_vals = df[c].to_numpy().astype(float)
            break
    if mag_vals is None:
        # try compute from dimensionless vector columns
        if all(cc in df.columns for cc in ['Ux_dimensionless','Uy_dimensionless','Uz_dimensionless']):
            uxs = df['Ux_dimensionless'].to_numpy().astype(float)
            uys = df['Uy_dimensionless'].to_numpy().astype(float)
            uzs = df['Uz_dimensionless'].to_numpy().astype(float)
            mag_vals = np.sqrt(uxs**2 + uys**2 + uzs**2)
    if mag_vals is None:
        raise RuntimeError("No dimensionless mag target found in " + fp + ". Provide 'mag_U' (dimensionless) or Ux_dimensionless/Uy_dimensionless/Uz_dimensionless.")

    Y_grid = np.zeros((1, ny, nx), dtype=np.float32) * np.nan
    mask_grid = np.zeros((1, ny, nx), dtype=np.float32)
    for i,(iy,ix) in enumerate(idx_map):
        val = mag_vals[i]
        # finite check
        if not np.isfinite(val):
            val = 0.0
            valid_val = 0.0
        else:
            valid_val = 1.0
        
        Y_grid[0, iy, ix] = float(val)
        sensor_w = float(df['is_sensor'].iloc[i]) if 'is_sensor' in df.columns else 1.0
        mask_grid[0, iy, ix] = sensor_w * valid_val

    Y_grid = np.nan_to_num(Y_grid, nan=0.0)
    Xs.append(X_tensor.squeeze(0))
    Ys.append(torch.from_numpy(Y_grid))
    Masks.append(torch.from_numpy(mask_grid))

X_all = torch.stack(Xs, dim=0).to(DEVICE)
Y_all = torch.stack(Ys, dim=0).to(DEVICE)
M_all = torch.stack(Masks, dim=0).to(DEVICE)
print("Dataset shapes", X_all.shape, Y_all.shape, M_all.shape)

dataset = TensorDataset(X_all, Y_all, M_all)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

in_ch = X_all.shape[1]
model = FNO2d(in_channels=in_ch, out_channels=1, modes1=MODES1, modes2=MODES2, width=WIDTH, n_layers=N_LAYERS).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS+1):
    model.train(); running=0.0
    for xb, yb, mb in loader:
        xb = xb.float(); yb = yb.float(); mb = mb.float()
        pred = model(xb)
        loss = sensor_weighted_mse(pred, yb, sensor_mask=mb)
        opt.zero_grad(); loss.backward(); opt.step()
        running += float(loss.item()) * xb.shape[0]
    print(f"Epoch {epoch}/{EPOCHS} loss {running/len(dataset):.6e}")
    
    # Save epoch checkpoint to epochs/ folder
    os.makedirs("epochs", exist_ok=True)
    epoch_path = os.path.join("epochs", MODEL_OUT + f".epoch{epoch}")
    torch.save(model.state_dict(), epoch_path)

torch.save(model.state_dict(), MODEL_OUT)
print("Training finished. Saved:", MODEL_OUT)
