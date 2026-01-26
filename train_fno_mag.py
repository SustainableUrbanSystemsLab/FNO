# Train FNO to predict dimensionless magnitude (mag_U)
import os, glob, numpy as np, pandas as pd, torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from fno2d_model import FNO2d, sensor_weighted_mse
from gh_to_fno import build_input_tensor_from_gh

# ============ Load Configuration ============
CONFIG_FILE = "config.toml"

def load_config():
    """Load configuration from config.toml file."""
    import tomllib  # Python 3.11+ built-in
    
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            return tomllib.load(f)
    else:
        print(f"Warning: {CONFIG_FILE} not found, using defaults")
        return {}

config = load_config()

# ============ Configuration from file ============
import sys
# Auto-detect platform and use appropriate data folder
if sys.platform == 'win32':
    DATA_FOLDER = config.get('paths', {}).get('data_folder_windows', 'train_csv')
else:
    DATA_FOLDER = config.get('paths', {}).get('data_folder_linux', 'train_csv')

MODEL_OUT = config.get('paths', {}).get('model_output', 'fno_mag_weights.pth')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = config.get('training', {}).get('batch_size', 4)
EPOCHS = config.get('training', {}).get('epochs', 200)
LR = config.get('training', {}).get('learning_rate', 1e-3)
MODES1 = config.get('model', {}).get('modes1', 32)
MODES2 = config.get('model', {}).get('modes2', 32)
WIDTH = config.get('model', {}).get('width', 64)
N_LAYERS = config.get('model', {}).get('n_layers', 5)

# ============ Load Logger ============
from training_logger import TrainingLogger

# ... (Config loading remains the same until GRAD_WEIGHT) ...
GRAD_WEIGHT = config.get('loss', {}).get('gradient_weight', 0.15)
SPECTRAL_WEIGHT = config.get('loss', {}).get('spectral_weight', 0.05)
PEAK_WEIGHT = config.get('loss', {}).get('peak_weight', 0.0) # Load peak weight

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

# Cleanup old files that might be corrupted or locked
if os.path.exists(MODEL_OUT):
    try: os.remove(MODEL_OUT)
    except: pass
if os.path.exists(MODEL_OUT + ".tmp"):
    try: os.remove(MODEL_OUT + ".tmp")
    except: pass

# EMERGENCY CLEANUP: Free up space by deleting old epoch history
import shutil
if os.path.exists("epochs"):
    try: shutil.rmtree("epochs", ignore_errors=True)
    except: pass
if os.path.exists("../epochs"): # Check parent too if being run from subfolder
    try: shutil.rmtree("../epochs", ignore_errors=True)
    except: pass

files = sorted(
    glob.glob(os.path.join(DATA_FOLDER, "**", "*.csv"), recursive=True)
)

if not files: raise RuntimeError("No training files in " + DATA_FOLDER)
Xs=[]; Ys=[]; Masks=[]
print("Preparing dataset from", len(files), "files...")
for fp in tqdm(files, desc="Data Preparation"):
    df = pd.read_csv(fp)
    # Renaming known variations
    rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
    df.rename(columns=rename_map, inplace=True)

    cols = ['SDF','Bldg_height','Z_relative','U_over_Uref','X_coords','Y_coords','dir_sin','dir_cos']
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
        u_over_uref_val = float(df['U_over_Uref'].iloc[i])
        
        # ✅ Precision Enhancement: Interior Punishment
        # We don't just 'mask' buildings (weight=0). We 'punish' the model if it 
        # tries to put wind inside them. This forces the boundary to stay sharp.
        if not np.isfinite(val):
            val = 0.0              # Target Speed = 0
            valid_val = 0.2        # Interior punishment weight (prev: 0.0)
        else:
            valid_val = 1.0
        
        # Target: Deficit relative to local inlet profile
        delta_u_normalized = (val - u_over_uref_val) / (u_over_uref_val + 1e-6)
        delta_u_normalized = np.clip(delta_u_normalized, -1.0, 5.0)
        
        Y_grid[0, iy, ix] = float(delta_u_normalized)
        
        sensor_w = float(df['is_sensor'].iloc[i]) if 'is_sensor' in df.columns else 1.0
        sdf_val = max(float(df['SDF'].iloc[i]), 0.0)
        
        # ✅ Physics weight: Focal Sharpness
        # alpha=19.0, L=5.0m (Narrow range focus + 20x surface weight)
        sdf_w = 1.0 + 19.0 * np.exp(-sdf_val / 5.0)
            
        mask_grid[0, iy, ix] = sensor_w * valid_val * sdf_w

    Y_grid = np.nan_to_num(Y_grid, nan=0.0)
    Xs.append(X_tensor.squeeze(0))
    Ys.append(torch.from_numpy(Y_grid))
    Masks.append(torch.from_numpy(mask_grid))

# Pad tensors to same size (max H, max W) to allow stacking
max_h = max(t.shape[1] for t in Xs)
max_w = max(t.shape[2] for t in Xs)
print(f"Max grid size found: {max_h}x{max_w}. Padding smaller grids...")

import torch.nn.functional as F
def pad_to_max(t_list, h, w):
    padded = []
    for t in t_list:
        # t shape: (C, H_curr, W_curr)
        # Pad right and bottom
        pad_h = h - t.shape[1]
        pad_w = w - t.shape[2]
        # F.pad expects (left, right, top, bottom)
        # We want (0, pad_w, 0, pad_h)
        p = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded.append(p)
    return torch.stack(padded, dim=0)

X_all = pad_to_max(Xs, max_h, max_w).to(DEVICE)
Y_all = pad_to_max(Ys, max_h, max_w).to(DEVICE)
M_all = pad_to_max(Masks, max_h, max_w).to(DEVICE)
print("Dataset shapes", X_all.shape, Y_all.shape, M_all.shape)

# Grouping Strategy:
# We sort files so that all 8 wind directions for "Case 0" are consecutive, then "Case 1", etc.
# We set BATCH=8 and shuffle=False.
# This ensures each training step sees ALL directions for ONE building geometry.
# This forces the model to learn how the fixed geometry interacts with changing wind.

dataset = TensorDataset(X_all, Y_all, M_all)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)

in_ch = X_all.shape[1]
model = FNO2d(in_channels=in_ch, out_channels=1, modes1=MODES1, modes2=MODES2, width=WIDTH, n_layers=N_LAYERS).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5)

# Initialize Logger
logger = TrainingLogger(output_dir="training_logs", experiment_name=None)
# Combine config for logging
full_config = config.copy()
full_config['training'] = {'batch_size': BATCH, 'epochs': EPOCHS, 'lr': LR}
full_config['model'] = {'modes1': MODES1, 'modes2': MODES2, 'width': WIDTH, 'n_layers': N_LAYERS}
full_config['loss'] = {'grad_weight': GRAD_WEIGHT, 'spectral_weight': SPECTRAL_WEIGHT, 'peak_weight': PEAK_WEIGHT}

logger.start_training(full_config, model=model)

# Early Stopping parameters (from config if available)
PATIENCE = config.get('training', {}).get('patience', 50)
best_loss = float('inf')
patience_counter = 0

import time

def save_feature_importance(model, feature_names, epoch_num=None):
    try:
        # print(f"\n--- Feature Importance (based on in_proj weights) ---")
        w = model.in_proj.weight.detach().cpu().numpy()
        importance = np.linalg.norm(w.squeeze(), axis=0)
        importance_pct = 100.0 * importance / importance.sum()
        
        feature_names = list(feature_names)
        indices = np.argsort(importance)[::-1]
        
        # Save to experiment dir instead of root
        filename = os.path.join(logger.experiment_dir, "feature_importance.txt")
        with open(filename, "a") as f: # Append mode to accumulate history
            header = f"\nEpoch {epoch_num} Importance" if epoch_num else "\nFinal Importance"
            f.write(header + "\n" + "-"*30 + "\n")
            
            for i in indices:
                name = feature_names[i] if i < len(feature_names) else f"Ch_{i}"
                msg = f"{name:15s}: {importance_pct[i]:.2f}%"
                # print(msg)
                f.write(msg + "\n")
        # print(f"Appended to {filename}")
    except Exception as e:
        print(f"Failed to calculate feature importance: {e}")

for epoch in range(1, EPOCHS+1):
    model.train(); running=0.0
    
    # Track components sum
    running_comp = {'mse_loss': 0.0, 'gradient_loss': 0.0, 'spectral_loss': 0.0, 'peak_loss': 0.0}
    
    start_time = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    
    for xb, yb, mb in pbar:
        xb = xb.float(); yb = yb.float(); mb = mb.float()
        pred = model(xb)
        
        # Pass PEAK_WEIGHT here
        loss, components = sensor_weighted_mse(pred, yb, sensor_mask=mb, 
                                             grad_weight=GRAD_WEIGHT, 
                                             spectral_weight=SPECTRAL_WEIGHT, 
                                             peak_weight=PEAK_WEIGHT,
                                             return_components=True)
                                             
        opt.zero_grad(); loss.backward(); opt.step()
        
        batch_size = xb.shape[0]
        running += float(loss.item()) * batch_size
        
        # Accumulate components
        for k, v in components.items():
            if k in running_comp:
                running_comp[k] += v * batch_size
                
        pbar.set_postfix({"loss": f"{loss.item():.4e}"})
        
    scheduler.step()
    epoch_time = time.time() - start_time
    
    avg_loss = running/len(dataset)
    
    # Calculate average components
    dataset_len = len(dataset)
    avg_components = {k: v / dataset_len for k, v in running_comp.items()}
    
    # Prepare metrics for logger
    metrics = {
        'total_loss': avg_loss,
        'mse_loss': avg_components['mse_loss'],
        'gradient_loss': avg_components['gradient_loss'],
        'spectral_loss': avg_components['spectral_loss'],
        'peak_loss': avg_components['peak_loss'],
        'learning_rate': opt.param_groups[0]['lr'],
        'epoch_time': epoch_time,
        'best_loss': best_loss, 
        'patience': patience_counter
    }
    
    logger.log_epoch(epoch, metrics)
    print(f"Epoch {epoch}/{EPOCHS} loss {avg_loss:.6e} (Peak: {avg_components['peak_loss']:.6e})")
    
    # Save epoch checkpoint history (every 10 epochs)
    if epoch % 100 == 0:
        os.makedirs("epochs", exist_ok=True)
        epoch_path = os.path.join("epochs", MODEL_OUT + f".epoch{epoch}")
        # Atomic save for epoch checkpoints
        temp_epoch = epoch_path + ".tmp"
        torch.save(model.state_dict(), temp_epoch)
        if os.path.exists(epoch_path): os.remove(epoch_path)
        os.rename(temp_epoch, epoch_path)
        
        # Save feature importance
        save_feature_importance(model, chs, epoch_num=epoch)

    # Check for improvement (Early Stopping)
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # Save BEST model to main file
        # Atomic save: save to temp and rename to prevent corruption
        temp_out = MODEL_OUT + ".tmp"
        torch.save(model.state_dict(), temp_out)
        if os.path.exists(MODEL_OUT): os.remove(MODEL_OUT)
        os.rename(temp_out, MODEL_OUT)
        print(f"  > New best loss! Saved {MODEL_OUT}")
    else:
        patience_counter += 1
        print(f"  > No improvement. Patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            save_feature_importance(model, chs, epoch_num=epoch) # Save at early stop too
            break

# Finish Logger
logger.finish_training()

print(f"Training finished. Best loss: {best_loss:.6e}")
print("Saved best model to:", MODEL_OUT)
# Also save feature importance to main output
save_feature_importance(model, chs)
