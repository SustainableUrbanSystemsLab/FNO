# Train FNO to predict dimensionless magnitude (mag_U)
import os, glob, numpy as np, pandas as pd, torch, sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.models.fno2d import FNO2d, sensor_weighted_mse
from core.utils.gh_to_fno import build_input_tensor_from_gh

# ============ Load Configuration ============
CONFIG_FILE = "config.toml"

def load_config():
    """Load configuration from config.toml file."""
    import tomllib  # Python 3.11+ built-in
    
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', CONFIG_FILE))
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
    # Check for PACE vs ICE paths
    pace_path = config.get('paths', {}).get('data_folder_linux', None)
    ice_path = config.get('paths', {}).get('data_folder_ice', None)
    
    if pace_path and os.path.exists(pace_path):
        DATA_FOLDER = pace_path
        print(f"Environment: PACE Cluster detected ({DATA_FOLDER})")
    elif ice_path and os.path.exists(ice_path):
        DATA_FOLDER = ice_path
        print(f"Environment: ICE Cluster detected ({DATA_FOLDER})")
    else:
        # Fallback to current directory or default
        DATA_FOLDER = 'train_csv'
        print(f"Environment: Linux (Generic). using local {DATA_FOLDER}")

MODEL_OUT = config.get('paths', {}).get('model_output', 'fno_mag_weights.pth')
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # defined in distributed setup now
BATCH = config.get('training', {}).get('batch_size', 4)
EPOCHS = config.get('training', {}).get('epochs', 200)
LR = config.get('training', {}).get('learning_rate', 1e-3)
MODES1 = config.get('model', {}).get('modes1', 32)
MODES2 = config.get('model', {}).get('modes2', 32)
WIDTH = config.get('model', {}).get('width', 64)
N_LAYERS = config.get('model', {}).get('n_layers', 5)

# ============ Distributed Setup ============
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        print(f"Distributed: True | Rank {rank}/{world_size}")
        return rank, world_size, local_rank, True
    return 0, 1, 0, False

RANK, WORLD_SIZE, LOCAL_RANK, IS_DISTRIBUTED = setup_distributed()
DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"

# ============ Load Logger ============
from core.utils.training_logger import TrainingLogger

# ... (Config loading remains the same until GRAD_WEIGHT) ...
# FIX: reduced max weights; physics terms now ramp from 0 over WARMUP_EPOCHS
# to prevent the model collapsing to a flat constant field early in training.
GRAD_WEIGHT     = config.get('loss', {}).get('gradient_weight', 0.5)   
SPECTRAL_WEIGHT = config.get('loss', {}).get('spectral_weight', 0.05)
PEAK_WEIGHT     = config.get('loss', {}).get('peak_weight', 0.3)        
WAKE_WEIGHT     = config.get('loss', {}).get('wake_weight', 0.3)        
WARMUP_EPOCHS   = config.get('loss', {}).get('warmup_epochs', 50)

def get_loss_weights(epoch):
    # Linearly ramp physics weights from 0 to their max over WARMUP_EPOCHS.
    # Pure MSE for the first few epochs gives the model a stable foundation
    # before physics penalties are introduced.
    t = min(epoch / max(WARMUP_EPOCHS, 1), 1.0)
    return dict(
        grad_weight=GRAD_WEIGHT * t,
        spectral_weight=SPECTRAL_WEIGHT * t,
        peak_weight=PEAK_WEIGHT * t,
        wake_weight=WAKE_WEIGHT * t,
    )

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

# Cleanup logic removed to prevent data loss on resume.
# Previous logic at lines 94-110 was deleting checkpoints at startup.
# We now preserve 'MODEL_OUT' and 'epochs/' to allow resumption.

# ============ Data Loading ============
import hashlib
import pickle
from multiprocessing import Pool, cpu_count

CACHE_FILE = "dataset_cache_fno_mag.pkl"
NUM_WORKERS = config.get('performance', {}).get('num_workers', 0)
if NUM_WORKERS == 0:
    NUM_WORKERS = max(1, cpu_count() - 2)

def load_single_csv(fp):
    """Load and process a single CSV file."""
    try:
        df = pd.read_csv(fp)
        # Renaming known variations
        rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
        df.rename(columns=rename_map, inplace=True)

        cols = ['SDF','Bldg_height','Z_relative','U_over_Uref','X_coords','Y_coords','dir_sin','dir_cos']
        if any(c not in df.columns for c in cols):
            return None, f"Missing input columns in {fp}"
            
        infer = infer_grid(df['X_coords'].to_numpy(), df['Y_coords'].to_numpy())
        if infer is None:
            if FORCE_H is None or FORCE_W is None:
                return None, f"Grid inference failed for {fp}"
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
            return None, f"No dimensionless mag target found in {fp}"

        Y_grid = np.zeros((1, ny, nx), dtype=np.float32) * np.nan
        mask_grid = np.zeros((1, ny, nx), dtype=np.float32)
        
        for i,(iy,ix) in enumerate(idx_map):
            val = mag_vals[i]
            u_ref = max(float(df['U_over_Uref'].iloc[i]), 0.01)

            # FIX: convert to Delta_U per data spec: (Mag_U - U_ref) / U_ref
            # This centres the target around 0 (wake deficit is negative, speedup positive)
            # so the model has a meaningful zero baseline to learn from.
            if not np.isfinite(val):
                delta_u = -1.0         # stagnation point inside solid
                valid_val = 0.2        # down-weight interior cells
            else:
                delta_u = (val - u_ref) / u_ref
                valid_val = 1.0

            target_val = np.clip(delta_u, -1.5, 2.0)
            Y_grid[0, iy, ix] = float(target_val)
            
            sensor_w = float(df['is_sensor'].iloc[i]) if 'is_sensor' in df.columns else 1.0
            sdf_val = max(float(df['SDF'].iloc[i]), 0.0)
            
            # FIX: reduced mask ceiling from 20x to 5x
            sdf_w = 1.0 + 4.0 * np.exp(-sdf_val / 5.0)
                
            mask_grid[0, iy, ix] = sensor_w * valid_val * sdf_w

        Y_grid = np.nan_to_num(Y_grid, nan=0.0)
        return (X_tensor.squeeze(0), torch.from_numpy(Y_grid), torch.from_numpy(mask_grid), chs), None
        
    except Exception as e:
        return None, f"Error processing {fp}: {e}"

def get_cache_hash(files):
    """Generate a hash based on file list and modifications."""
    hash_parts = ["fno_mag_direct_pred_v1"] # Version tag to force invalidation on code change
    hash_input = "".join(sorted([os.path.basename(f) for f in files[::10]])) # Sample files
    return hashlib.md5((str(len(files)) + hash_input).encode()).hexdigest()

def load_or_prepare_dataset(files, rank, is_main):
    """Load from cache or prepare dataset with multiprocessing."""
    # Support for Pickle-Only Transfer (ICE Cluster)
    if len(files) == 0:
        # If no CSVs are found, check if a cache file exists blindly
        potential_caches = glob.glob(f"{CACHE_FILE}.*")
        if potential_caches:
            cache_path = potential_caches[0]
            if is_main:
                print(f"Warning: No CSV files found, but detected cache: {cache_path}")
                print("Attempting to load dataset from cache (Pickle-Only Mode)...")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                if is_main: print(f"Failed to load existing cache {cache_path}: {e}")
        
        # If we get here, truly no data
        return [], [], [], None

    cache_hash = get_cache_hash(files)
    cache_path = f"{CACHE_FILE}.{cache_hash}"
    
    if os.path.exists(cache_path):
        if is_main: print(f"Loading cached dataset from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            if is_main: print(f"Cache load failed ({e}), rebuilding...")
    
    if is_main: print(f"Preparing dataset from {len(files)} files using {NUM_WORKERS} workers...")
    
    # Use torch.multiprocessing if available for tensor sharing, else standard
    # Standard is safer for data loading logic usually
    with Pool(NUM_WORKERS) as pool:
        if is_main:
            results = list(tqdm(pool.imap(load_single_csv, files), total=len(files), desc="Data Preparation"))
        else:
            results = list(pool.imap(load_single_csv, files))
            
    Xs, Ys, Masks = [], [], []
    chs = None
    errors = []
    
    for res, err in results:
        if err:
            errors.append(err)
            continue
        X, Y, M, c = res
        Xs.append(X)
        Ys.append(Y)
        Masks.append(M)
        if chs is None: chs = c
        
    if is_main:
        print(f"Loaded {len(Xs)} samples. {len(errors)} errors.")
        if errors and len(errors) < 10: print("Errors:", errors)
        
        # Save cache
        print(f"Saving cache to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump((Xs, Ys, Masks, chs), f)
            
    return Xs, Ys, Masks, chs
files = sorted(
    glob.glob(os.path.join(DATA_FOLDER, "**", "*.csv"), recursive=True)
)
if not files: raise RuntimeError("No training files in " + DATA_FOLDER)

Xs, Ys, Masks, chs = load_or_prepare_dataset(files, RANK, RANK==0)

if len(Xs) == 0:
    raise RuntimeError("No valid data loaded after filtering errors.")

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

# NOTE: DDP & DataLoader with pin_memory require CPU tensors initially.
# We move them to device ONLY inside the loop or let DataLoader handle pinning.
X_all = pad_to_max(Xs, max_h, max_w) # Keep on CPU
Y_all = pad_to_max(Ys, max_h, max_w) # Keep on CPU
M_all = pad_to_max(Masks, max_h, max_w) # Keep on CPU
print("Dataset shapes (CPU)", X_all.shape, Y_all.shape, M_all.shape)

# Grouping Strategy:
# We sort files so that all 8 wind directions for "Case 0" are consecutive, then "Case 1", etc.
# We set BATCH=8 and shuffle=False.
# This ensures each training step sees ALL directions for ONE building geometry.
# This forces the model to learn how the fixed geometry interacts with changing wind.



dataset = TensorDataset(X_all, Y_all, M_all)

if IS_DISTRIBUTED:
    sampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler, pin_memory=True)
else:
    sampler = None
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

in_ch = X_all.shape[1]
model = FNO2d(in_channels=in_ch, out_channels=1, modes1=MODES1, modes2=MODES2, width=WIDTH, n_layers=N_LAYERS).to(DEVICE)

if IS_DISTRIBUTED:
    model = DDP(model, device_ids=[LOCAL_RANK])
elif torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = DDP(model, device_ids=[LOCAL_RANK])
elif torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = nn.DataParallel(model)

opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

# Resume from checkpoint if available
CHECKPOINT_FILE = "checkpoint.pth"
start_epoch = 1

if os.path.exists(CHECKPOINT_FILE):
    print(f"Resuming from internal checkpoint: {CHECKPOINT_FILE}")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    
    # Handle Full Checkpoint (dict) vs Weights Only (older files)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        # Load optimizer/scheduler if available
        if 'optimizer_state_dict' in checkpoint:
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resuming from Epoch {checkpoint['epoch']} (Best Loss: {best_loss:.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("Resuming from weights-only checkpoint.")

elif os.path.exists(MODEL_OUT):
    print(f"Resuming from best model: {MODEL_OUT}")
    # Map location is critical on DDP to avoid device mismatch
    state_dict = torch.load(MODEL_OUT, map_location=DEVICE)
    model.load_state_dict(state_dict)



# Initialize Logger (Only on Rank 0)
if RANK == 0:
    logger = TrainingLogger(output_dir="training_logs", experiment_name=None)
    # Combine config for logging
    full_config = config.copy()
    full_config['training'] = {'batch_size': BATCH, 'epochs': EPOCHS, 'lr': LR}
    full_config['model'] = {'modes1': MODES1, 'modes2': MODES2, 'width': WIDTH, 'n_layers': N_LAYERS}
    full_config['loss'] = {'grad_weight': GRAD_WEIGHT, 'spectral_weight': SPECTRAL_WEIGHT, 'peak_weight': PEAK_WEIGHT}
    
    logger.start_training(full_config, model=model)
else:
    logger = None

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

for epoch in range(start_epoch, EPOCHS+1):
    model.train(); running=0.0
    
    if IS_DISTRIBUTED:
        sampler.set_epoch(epoch)
    
    # Track components sum
    running_comp = {'mse_loss': 0.0, 'gradient_loss': 0.0, 'spectral_loss': 0.0, 'peak_loss': 0.0, 'neg_loss': 0.0}
    
    start_time = time.time()
    
    # Only show progress bar on Rank 0
    if RANK == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    else:
        pbar = loader
    
    for xb, yb, mb in pbar:
        # Move batch to GPU explicitly
        xb, yb, mb = xb.to(DEVICE).float(), yb.to(DEVICE).float(), mb.to(DEVICE).float()
        pred = model(xb)
        
        # FIX: use epoch-dependent warmup weights
        w = get_loss_weights(epoch)
        loss, components = sensor_weighted_mse(pred, yb, sensor_mask=mb, 
                                             grad_weight=w['grad_weight'], 
                                             spectral_weight=w['spectral_weight'], 
                                             peak_weight=w['peak_weight'],
                                             wake_weight=w['wake_weight'],
                                             return_components=True)
                                             
        opt.zero_grad(); loss.backward(); opt.step()
        
        batch_size = xb.shape[0]
        running += float(loss.item()) * batch_size
        
        # Accumulate components
        for k, v in components.items():
            if k in running_comp:
                running_comp[k] += v * batch_size
                
        if RANK == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4e}"})
        
    scheduler.step()
    epoch_time = time.time() - start_time
    
    avg_loss = running/len(dataset)
    
    # Calculate average components
    dataset_len = len(dataset)
    avg_components = {k: v / dataset_len for k, v in running_comp.items()}
    
    # Prepare metrics for logger


    # Load Checkpoint Interval
    CHECKPOINT_INTERVAL = config.get('training', {}).get('checkpoint_interval', 10)

    # Save Rolling Checkpoint (Every N epochs, overwrite)
    # Save Rolling Checkpoint (Every N epochs, overwrite)
    if epoch % CHECKPOINT_INTERVAL == 0:
        # Atomic overwriting of rolling checkpoint
        if RANK == 0:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }
            temp_ckpt = CHECKPOINT_FILE + ".tmp"
            torch.save(checkpoint_state, temp_ckpt)
            if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)
            os.rename(temp_ckpt, CHECKPOINT_FILE)
            print(f"  > Saved rolling full-state checkpoint to {CHECKPOINT_FILE}")
            
            # Save feature importance
            save_feature_importance(model, chs, epoch_num=epoch)
            
    # Check for improvement (Early Stopping)
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # Save BEST model to main file
        # Atomic save: save to temp and rename to prevent corruption
        if RANK == 0:
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
            early_stop = True # Flag to break after logging
        else:
            early_stop = False

    # Prepare metrics for logger (updated with new patience)
    metrics = {
        'total_loss': avg_loss,
        'mse_loss': avg_components['mse_loss'],
        'gradient_loss': avg_components['gradient_loss'],
        'spectral_loss': avg_components['spectral_loss'],
        'peak_loss': avg_components['peak_loss'],
        'wake_loss': avg_components.get('wake_loss', 0.0),
        'learning_rate': opt.param_groups[0]['lr'],
        'epoch_time_sec': epoch_time,
        'best_loss': best_loss, 
        'patience_counter': patience_counter
    }
    
    if RANK == 0:
        logger.log_epoch(epoch, metrics)
        print(f"Epoch {epoch}/{EPOCHS} loss {avg_loss:.6e} (Peak: {avg_components['peak_loss']:.6e})")

    if 'early_stop' in locals() and early_stop:
        break

# Finish Logger
if RANK == 0:
    logger.finish_training()

print(f"Training finished. Best loss: {best_loss:.6e}")
print("Saved best model to:", MODEL_OUT)
# Also save feature importance to main output
save_feature_importance(model, chs)
