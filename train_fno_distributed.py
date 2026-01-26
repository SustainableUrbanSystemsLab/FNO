# Multi-GPU Training for FNO (using DistributedDataParallel)
# Usage: torchrun --nproc_per_node=2 train_fno_distributed.py
#    or: python train_fno_distributed.py (falls back to single GPU)

import os, glob, numpy as np, pandas as pd, torch, hashlib, pickle, sys, argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from fno2d_model import FNO2d, sensor_weighted_mse
from gh_to_fno import build_input_tensor_from_gh
from training_logger import TrainingLogger

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
# Auto-detect platform and use appropriate data folder
if sys.platform == 'win32':
    DATA_FOLDER = config.get('paths', {}).get('data_folder_windows', 'train_csv')
else:
    DATA_FOLDER = config.get('paths', {}).get('data_folder_linux', 'train_csv')

MODEL_OUT = config.get('paths', {}).get('model_output', 'fno_mag_weights.pth')
CHECKPOINT_PATH = config.get('paths', {}).get('checkpoint_file', 'checkpoint_latest.pth')
CACHE_FILE = "dataset_cache.pkl"
BATCH = config.get('training', {}).get('batch_size', 4)
EPOCHS = config.get('training', {}).get('epochs', 200)
LR = config.get('training', {}).get('learning_rate', 1e-3)
PATIENCE = config.get('training', {}).get('patience', 50)
CHECKPOINT_INTERVAL = config.get('training', {}).get('checkpoint_interval', 10)
MODES1 = config.get('model', {}).get('modes1', 32)
MODES2 = config.get('model', {}).get('modes2', 32)
WIDTH = config.get('model', {}).get('width', 64)
N_LAYERS = config.get('model', {}).get('n_layers', 5)
GRAD_WEIGHT = config.get('loss', {}).get('gradient_weight', 0.15)
SPECTRAL_WEIGHT = config.get('loss', {}).get('spectral_weight', 0.05)
FORCE_H = None; FORCE_W = None

# Worker count from config (0 = auto-detect)
_configured_workers = config.get('performance', {}).get('num_workers', 0)
if _configured_workers > 0:
    NUM_WORKERS = _configured_workers
elif sys.platform == 'win32':
    NUM_WORKERS = min(8, max(1, cpu_count() // 2))
else:
    NUM_WORKERS = max(1, cpu_count() // 2)

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

        cols = ['SDF','Bldg_height','Z_relative','U_over_Uref','X_coords','Y_coords','dir_sin','dir_cos']
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
            u_over_uref_val = float(df['U_over_Uref'].iloc[i])
            
            if not np.isfinite(val):
                val = 0.0
                valid_val = 0.2
            else:
                valid_val = 1.0
            
            delta_u_normalized = (val - u_over_uref_val) / (u_over_uref_val + 1e-6)
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
    """Generate a hash based on file count, folder count, and sample of modification times."""
    # Include total count and folder structure
    folders = set(os.path.dirname(f) for f in files)
    hash_parts = [
        f"files:{len(files)}",
        f"folders:{len(folders)}",
    ]
    # Sample files for modification times (every 10th file for speed)
    for f in files[::10]:
        hash_parts.append(f"{os.path.basename(f)}_{os.path.getmtime(f):.0f}")
    hash_input = "|".join(hash_parts)
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FNO Training')
    parser.add_argument('--fresh', action='store_true', help='Start fresh training (ignore checkpoint)')
    args = parser.parse_args()
    
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
        print("=" * 50)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 50)
        print(f"  Total samples: {len(Xs)}")
        print(f"  Input shape:   {X_all.shape} (N, C, H, W)")
        print(f"  Target shape:  {Y_all.shape}")
        print(f"  Mask shape:    {M_all.shape}")
        print(f"  Input channels: {chs}")
        print("=" * 50)

    # Create dataset and sampler
    dataset = TensorDataset(X_all, Y_all, M_all)
    
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)

    if is_main_process(rank):
        print("CREATING MODEL...")
        print(f"  FNO2d: modes=({MODES1},{MODES2}), width={WIDTH}, layers={N_LAYERS}")
        
    # Create model
    in_ch = X_all.shape[1]
    model = FNO2d(in_channels=in_ch, out_channels=1, modes1=MODES1, modes2=MODES2, 
                  width=WIDTH, n_layers=N_LAYERS).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    if is_main_process(rank):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        print("=" * 50)
    
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)  # Smooth LR decay
    
    # Checkpoint resume logic (auto-detect)
    start_epoch = 1
    best_loss = float('inf')
    patience_counter = 0
    
    # Auto-resume if checkpoint exists (unless --fresh flag is passed)
    if os.path.exists(CHECKPOINT_PATH) and not args.fresh:
        if is_main_process(rank):
            print("=" * 50)
            print("CHECKPOINT DETECTED")
            print("=" * 50)
            print(f"  Loading from: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if is_distributed:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        patience_counter = checkpoint['patience_counter']
        if is_main_process(rank):
            print(f"  Last completed epoch: {checkpoint['epoch']}")
            print(f"  Best loss so far: {best_loss:.6e}")
            print(f"  Training will continue from epoch {start_epoch}")
            print(f"  (Use 'python train_fno_distributed.py --fresh' to start from scratch)")
            print("=" * 50)
    else:
        if is_main_process(rank):
            print("=" * 50)
            print("STARTING FRESH TRAINING")
            print("=" * 50)

    if is_main_process(rank):
        print(f"  Epochs: {EPOCHS}, Batch size: {BATCH}, LR: {LR}")
        print(f"  Patience: {PATIENCE}, Device: {device}")
        print("=" * 50)
        
        # Initialize training logger for publication metrics
        logger = TrainingLogger(output_dir="training_logs")
        logger.start_training({
            'batch_size': BATCH,
            'epochs': EPOCHS,
            'learning_rate': LR,
            'patience': PATIENCE,
            'modes1': MODES1,
            'modes2': MODES2,
            'width': WIDTH,
            'n_layers': N_LAYERS,
            'gradient_weight': GRAD_WEIGHT,
            'spectral_weight': SPECTRAL_WEIGHT,
            'dataset_size': len(dataset),
            'distributed': is_distributed,
            'world_size': world_size,
        }, model=model.module if is_distributed else model)
    else:
        logger = None

    # Training loop

    for epoch in range(start_epoch, EPOCHS + 1):
        if is_distributed:
            sampler.set_epoch(epoch)
        
        model.train()
        running = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False) if is_main_process(rank) else loader
        
        # Accumulators for loss components
        running_mse = 0.0
        running_grad = 0.0
        running_spec = 0.0
        
        for xb, yb, mb in pbar:
            xb = xb.float().to(device)
            yb = yb.float().to(device)
            mb = mb.float().to(device)
            
            pred = model(xb)
            loss, components = sensor_weighted_mse(pred, yb, sensor_mask=mb, grad_weight=GRAD_WEIGHT, spectral_weight=SPECTRAL_WEIGHT, return_components=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            batch_size = xb.shape[0]
            running += float(loss.item()) * batch_size
            running_mse += components['mse_loss'] * batch_size
            running_grad += components['gradient_loss'] * batch_size
            running_spec += components['spectral_loss'] * batch_size
            
            if is_main_process(rank) and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({"loss": f"{loss.item():.4e}"})
        
        scheduler.step()
        n_samples = len(dataset)
        
        # Aggregate loss across GPUs
        if is_distributed:
            # Sync GPUs before collective operation to prevent timeout
            torch.cuda.synchronize()
            dist.barrier()
            
            # Aggregate all loss components across GPUs
            running_tensor = torch.tensor([running, running_mse, running_grad, running_spec], device=device)
            dist.all_reduce(running_tensor, op=dist.ReduceOp.SUM)
            running = running_tensor[0].item()
            running_mse = running_tensor[1].item()
            running_grad = running_tensor[2].item()
            running_spec = running_tensor[3].item()
        
        avg_loss = running / n_samples
        avg_mse = running_mse / n_samples
        avg_grad = running_grad / n_samples
        avg_spec = running_spec / n_samples
        
        if is_main_process(rank):
            print(f"Epoch {epoch}/{EPOCHS} loss {avg_loss:.6e}")
            
            # Save resumable checkpoint every N epochs
            if epoch % CHECKPOINT_INTERVAL == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model': model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'patience_counter': patience_counter,
                }
                torch.save(checkpoint, CHECKPOINT_PATH)
                print(f"  > Checkpoint saved to {CHECKPOINT_PATH}")

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
            
            # Log epoch metrics for publication
            if logger:
                logger.log_epoch(epoch, {
                    'total_loss': avg_loss,
                    'mse_loss': avg_mse,
                    'gradient_loss': avg_grad,
                    'spectral_loss': avg_spec,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'best_loss': best_loss,
                    'patience': patience_counter,
                })

    if is_main_process(rank):
        print(f"Training finished. Best loss: {best_loss:.6e}")
        print("Saved best model to:", MODEL_OUT)
        
        # Finalize training logger
        if logger:
            logger.finish_training({'best_loss': best_loss})
            
            # Generate publication-ready plots
            try:
                from generate_plots import generate_publication_plots
                generate_publication_plots(logger.metrics_csv)
            except Exception as e:
                print(f"[Plots] Could not generate plots: {e}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
