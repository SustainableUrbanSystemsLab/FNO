# Multi-GPU Training for FNO (using DistributedDataParallel)
# Usage: torchrun --nproc_per_node=2 train_fno_distributed.py
#    or: python train_fno_distributed.py (falls back to single GPU)

import os, glob, numpy as np, pandas as pd, torch, hashlib, pickle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from fno2d_model import FNO2d, sensor_weighted_mse
from gh_to_fno import build_input_tensor_from_gh

# ============ Configuration ============
DATA_FOLDER = r"C:\LabShare\Dataset\FormFluxCases\Compressed\Training_Dataset"
MODEL_OUT = "fno_mag_weights.pth"
CACHE_FILE = "dataset_cache.pkl"
BATCH = 4  # Per-GPU batch size
EPOCHS = 200
LR = 1e-3
MODES1 = 32; MODES2 = 32; WIDTH = 64; N_LAYERS = 5
FORCE_H = None; FORCE_W = None
NUM_WORKERS = cpu_count()  # Use all available cores for data loading

# ============ Distributed Setup ============
def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size, True
    else:
        # Fallback to single GPU
        return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

# ============ Grid Inference ============
def infer_grid(xs, ys, tol=1e-6):
    xs = np.array(xs); ys = np.array(ys)
    kx = np.round(xs/tol).astype(int); ky = np.round(ys/tol).astype(int)
    ux = np.unique(kx); uy = np.unique(ky)
    key_x = {k:i for i,k in enumerate(np.sort(ux))}
    key_y = {k:i for i,k in enumerate(np.sort(uy))}
    idx = [(key_y[kyv], key_x[kxv]) for kxv,kyv in zip(kx,ky)]
    return len(ux), len(uy), idx

# ============ Single File Processing (for multiprocessing) ============
def process_single_file(fp):
    """Process a single CSV file and return tensors. Used by multiprocessing pool."""
    try:
        df = pd.read_csv(fp)
        rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
        df.rename(columns=rename_map, inplace=True)

        cols = ['SDF','Bldg_height','Z_relative','U_at_z','X_coords','Y_coords','dir_sin','dir_cos']
        if any(c not in df.columns for c in cols):
            return None, f"{fp} missing input columns"
        
        infer = infer_grid(df['X_coords'].to_numpy(), df['Y_coords'].to_numpy())
        if infer is None:
            return None, f"Grid inference failed for {fp}"
        nx, ny, idx_map = infer

        gh_out = {c: df[c].tolist() for c in cols}
        X_tensor, chs = build_input_tensor_from_gh(gh_out, H=ny, W=nx, device='cpu')

        mag_cols_dim = ['mag_U_dimensionless','mag_U','mag_dimensionless']
        mag_vals = None
        for c in mag_cols_dim:
            if c in df.columns:
                mag_vals = df[c].to_numpy().astype(float)
                break
        if mag_vals is None:
            if all(cc in df.columns for cc in ['Ux_dimensionless','Uy_dimensionless','Uz_dimensionless']):
                uxs = df['Ux_dimensionless'].to_numpy().astype(float)
                uys = df['Uy_dimensionless'].to_numpy().astype(float)
                uzs = df['Uz_dimensionless'].to_numpy().astype(float)
                mag_vals = np.sqrt(uxs**2 + uys**2 + uzs**2)
        if mag_vals is None:
            return None, f"No dimensionless mag target found in {fp}"

        Y_grid = np.zeros((1, ny, nx), dtype=np.float32) * np.nan
        mask_grid = np.zeros((1, ny, nx), dtype=np.float32)
        
        for i, (iy, ix) in enumerate(idx_map):
            val = mag_vals[i]
            u_at_z_val = float(df['U_at_z'].iloc[i])
            
            if not np.isfinite(val):
                val = 0.0
                valid_val = 0.2
            else:
                valid_val = 1.0
            
            delta_u_normalized = (val - u_at_z_val) / (u_at_z_val + 1e-6)
            delta_u_normalized = np.clip(delta_u_normalized, -1.0, 0.5)
            Y_grid[0, iy, ix] = float(delta_u_normalized)
            
            sensor_w = float(df['is_sensor'].iloc[i]) if 'is_sensor' in df.columns else 1.0
            sdf_val = max(float(df['SDF'].iloc[i]), 0.0)
            sdf_w = 1.0 + 19.0 * np.exp(-sdf_val / 5.0)
            mask_grid[0, iy, ix] = sensor_w * valid_val * sdf_w

        Y_grid = np.nan_to_num(Y_grid, nan=0.0)
        return (X_tensor.squeeze(0), torch.from_numpy(Y_grid), torch.from_numpy(mask_grid), chs), None
    except Exception as e:
        return None, f"Error processing {fp}: {e}"

def get_cache_hash(files):
    """Generate a hash based on file list and modification times."""
    hash_input = "".join([f"{f}_{os.path.getmtime(f)}" for f in files[:100]])  # Sample first 100 for speed
    return hashlib.md5(hash_input.encode()).hexdigest()

def load_or_prepare_dataset(files, rank, is_main):
    """Load from cache or prepare dataset with multiprocessing."""
    cache_hash = get_cache_hash(files)
    cache_path = f"{CACHE_FILE}.{cache_hash}"
    
    # Try to load from cache
    if os.path.exists(cache_path):
        if is_main:
            print(f"Loading cached dataset from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Prepare dataset with multiprocessing
    if is_main:
        print(f"Preparing dataset from {len(files)} files using {NUM_WORKERS} workers...")
    
    with Pool(NUM_WORKERS) as pool:
        if is_main:
            results = list(tqdm(pool.imap(process_single_file, files), total=len(files), desc="Data Preparation"))
        else:
            results = list(pool.imap(process_single_file, files))
    
    Xs, Ys, Masks = [], [], []
    chs = None
    for result, error in results:
        if error:
            if is_main:
                print(f"Warning: {error}")
            continue
        X, Y, M, c = result
        Xs.append(X)
        Ys.append(Y)
        Masks.append(M)
        if chs is None:
            chs = c
    
    # Save to cache (only on main process)
    if is_main:
        print(f"Saving cache to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump((Xs, Ys, Masks, chs), f)
    
    return Xs, Ys, Masks, chs

# ============ Main Training ============
def main():
    local_rank, rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process(rank):
        print(f"Training with {world_size} GPU(s)")
        print(f"Distributed: {is_distributed}")
    
    # Cleanup old files (only on main process)
    if is_main_process(rank):
        if os.path.exists(MODEL_OUT):
            try: os.remove(MODEL_OUT)
            except: pass
        if os.path.exists(MODEL_OUT + ".tmp"):
            try: os.remove(MODEL_OUT + ".tmp")
            except: pass
        import shutil
        if os.path.exists("epochs"):
            try: shutil.rmtree("epochs", ignore_errors=True)
            except: pass

    # Load files
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "**", "*.csv"), recursive=True))
    if not files:
        raise RuntimeError("No training files in " + DATA_FOLDER)
    
    # Load or prepare dataset (with caching and multiprocessing)
    Xs, Ys, Masks, chs = load_or_prepare_dataset(files, rank, is_main_process(rank))

    # Pad tensors
    max_h = max(t.shape[1] for t in Xs)
    max_w = max(t.shape[2] for t in Xs)
    if is_main_process(rank):
        print(f"Max grid size: {max_h}x{max_w}. Padding...")

    import torch.nn.functional as F
    def pad_to_max(t_list, h, w):
        padded = []
        for t in t_list:
            pad_h = h - t.shape[1]
            pad_w = w - t.shape[2]
            p = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
            padded.append(p)
        return torch.stack(padded, dim=0)

    X_all = pad_to_max(Xs, max_h, max_w)
    Y_all = pad_to_max(Ys, max_h, max_w)
    M_all = pad_to_max(Masks, max_h, max_w)
    
    if is_main_process(rank):
        print("Dataset shapes", X_all.shape, Y_all.shape, M_all.shape)

    # Create dataset and sampler
    dataset = TensorDataset(X_all, Y_all, M_all)
    
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)

    # Create model
    in_ch = X_all.shape[1]
    model = FNO2d(in_channels=in_ch, out_channels=1, modes1=MODES1, modes2=MODES2, 
                  width=WIDTH, n_layers=N_LAYERS).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5)

    # Training loop
    PATIENCE = 50
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        if is_distributed:
            sampler.set_epoch(epoch)
        
        model.train()
        running = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False) if is_main_process(rank) else loader
        for xb, yb, mb in pbar:
            xb = xb.float().to(device)
            yb = yb.float().to(device)
            mb = mb.float().to(device)
            
            pred = model(xb)
            loss = sensor_weighted_mse(pred, yb, sensor_mask=mb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.shape[0]
            
            if is_main_process(rank) and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({"loss": f"{loss.item():.4e}"})
        
        scheduler.step()
        
        # Aggregate loss across GPUs
        if is_distributed:
            running_tensor = torch.tensor([running], device=device)
            dist.all_reduce(running_tensor, op=dist.ReduceOp.SUM)
            running = running_tensor.item()
        
        avg_loss = running / len(dataset)
        
        if is_main_process(rank):
            print(f"Epoch {epoch}/{EPOCHS} loss {avg_loss:.6e}")
            
            # Save checkpoints
            if epoch % 100 == 0:
                os.makedirs("epochs", exist_ok=True)
                epoch_path = os.path.join("epochs", MODEL_OUT + f".epoch{epoch}")
                state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save(state_dict, epoch_path)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                temp_out = MODEL_OUT + ".tmp"
                torch.save(state_dict, temp_out)
                if os.path.exists(MODEL_OUT):
                    os.remove(MODEL_OUT)
                os.rename(temp_out, MODEL_OUT)
                print(f"  > New best loss! Saved {MODEL_OUT}")
            else:
                patience_counter += 1
                print(f"  > No improvement. Patience {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break

    if is_main_process(rank):
        print(f"Training finished. Best loss: {best_loss:.6e}")
        print("Saved best model to:", MODEL_OUT)

    cleanup_distributed()

if __name__ == "__main__":
    main()
