"""
Distributed Training for Hybrid FNO Model
=========================================
Features:
- PyTorch Distributed Data Parallel (DDP)
- Physics-Informed Loss (Divergence / Continuity)
- Hashed Dataset Caching
- Slurm/ICE Cluster Integration
- Multiprocessing Data Preparation
"""

import os, glob, numpy as np, pandas as pd, torch, hashlib, pickle, sys, argparse, time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import tomllib

# Local imports
from fno_hybrid_model import HybridFNO, physics_informed_loss
from gh_to_fno import build_input_tensor_from_gh
from training_logger import TrainingLogger
from neuralop.losses import LpLoss, H1Loss

# ============ Load Configuration ============
def load_config(config_file):
    """Load configuration from toml file."""
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            return tomllib.load(f)
    print(f"Warning: {config_file} not found, using defaults")
    return {}

# ============ Distributed Setup ============
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size, True
    return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

# ============ Data Loading Boilerplate (from train_fno_distributed.py) ============
def infer_grid(xs, ys, tol=1e-6):
    xs = np.array(xs); ys = np.array(ys)
    kx = np.round(xs/tol).astype(int); ky = np.round(ys/tol).astype(int)
    ux = np.unique(kx); uy = np.unique(ky)
    key_x = {k:i for i,k in enumerate(np.sort(ux))}
    key_y = {k:i for i,k in enumerate(np.sort(uy))}
    idx = [(key_y[kyv], key_x[kxv]) for kxv,kyv in zip(kx,ky)]
    return len(ux), len(uy), idx

def process_single_file(fp):
    try:
        df = pd.read_csv(fp)
        rename_map = {'X': 'X_coords', 'Y': 'Y_coords', 'x': 'X_coords', 'y': 'Y_coords'}
        df.rename(columns=rename_map, inplace=True)
        cols = ['SDF','Bldg_height','Z_relative','U_over_Uref','X_coords','Y_coords','dir_sin','dir_cos']
        if any(c not in df.columns for c in cols): return None, f"{fp} missing input columns"
        infer = infer_grid(df['X_coords'].to_numpy(), df['Y_coords'].to_numpy())
        nx, ny, idx_map = infer
        gh_out = {c: df[c].tolist() for c in cols}
        X_tensor, chs = build_input_tensor_from_gh(gh_out, H=ny, W=nx, device='cpu')
        
        mag_cols = ['mag_U_dimensionless','mag_U','mag_dimensionless']
        mag_vals = None
        for c in mag_cols:
            if c in df.columns: mag_vals = df[c].to_numpy().astype(float); break
        if mag_vals is None: return None, f"No mag target found in {fp}"

        Y_grid = np.zeros((1, ny, nx), dtype=np.float32)
        mask_grid = np.zeros((1, ny, nx), dtype=np.float32)
        for i, (iy, ix) in enumerate(idx_map):
            val = mag_vals[i]
            u_over_uref = float(df['U_over_Uref'].iloc[i])
            delta_u = (val - u_over_uref) / (u_over_uref + 1e-6)
            delta_u = np.clip(delta_u, -2.0, 5.0) 
            Y_grid[0, iy, ix] = float(delta_u)
            sdf_val = max(float(df['SDF'].iloc[i]), 0.0)
            sdf_w = 1.0 + 19.0 * np.exp(-sdf_val / 5.0)
            mask_grid[0, iy, ix] = sdf_w
        return (X_tensor.squeeze(0), torch.from_numpy(Y_grid), torch.from_numpy(mask_grid), chs), None
    except Exception as e: return None, f"Error processing {fp}: {e}"

def get_cache_hash(files):
    hash_parts = [f"files:{len(files)}", "hybrid_v1"]
    for f in files[::10]: hash_parts.append(f"{os.path.basename(f)}_{os.path.getmtime(f):.0f}")
    return hashlib.md5("|".join(hash_parts).encode()).hexdigest()

def load_or_prepare_dataset(files, rank, is_main, num_workers):
    cache_hash = get_cache_hash(files)
    cache_path = f"dataset_cache_hybrid_{cache_hash}.pkl"
    if os.path.exists(cache_path):
        if is_main: print(f"Loading cached dataset from {cache_path}...")
        with open(cache_path, 'rb') as f: return pickle.load(f)
    if is_main: print(f"Preparing dataset using {num_workers} workers...")
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_file, files), total=len(files))) if is_main else list(pool.imap(process_single_file, files))
    Xs, Ys, Masks, chs = [], [], [], None
    for res, err in results:
        if err: continue
        X, Y, M, c = res
        Xs.append(X); Ys.append(Y); Masks.append(M)
        if chs is None: chs = c
    if is_main:
        with open(cache_path, 'wb') as f: pickle.dump((Xs, Ys, Masks, chs), f)
    return Xs, Ys, Masks, chs

# ============ Main Training ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.toml')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--reset-patience', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    local_rank, rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Path detection
    if sys.platform == 'win32':
        DATA_FOLDER = config.get('paths', {}).get('data_folder_windows', 'train_csv')
    else:
        linux_path = config.get('paths', {}).get('data_folder_linux', '')
        ice_path = config.get('paths', {}).get('data_folder_ice', '')
        
        # Priority: ICE > Linux > local 'train_csv'
        if ice_path and os.path.exists(ice_path) and glob.glob(os.path.join(ice_path, "*.csv")):
            DATA_FOLDER = ice_path
        elif linux_path and os.path.exists(linux_path) and glob.glob(os.path.join(linux_path, "*.csv")):
            DATA_FOLDER = linux_path
        else:
            DATA_FOLDER = 'train_csv' # Final fallback to local folder
            
    if is_main_process(rank):
        print(f"Using Data Folder: {DATA_FOLDER}")

    MODEL_OUT = "hybrid_fno_weights.pth"
    CHECKPOINT_PATH = "checkpoint_hybrid.pth"
    BATCH = config.get('training', {}).get('batch_size', 4)
    EPOCHS = config.get('training', {}).get('epochs', 200)
    LR = config.get('training', {}).get('learning_rate', 5e-4)
    
    # Data Prep
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "**/*.csv"), recursive=True))
    if not files:
        raise RuntimeError(f"No CSV files found in {DATA_FOLDER}. Please check your config.toml paths.")
        
    num_workers = max(1, cpu_count() // 2)
    Xs, Ys, Masks, chs = load_or_prepare_dataset(files, rank, is_main_process(rank), num_workers)
    
    if not Xs:
        raise RuntimeError(f"Failed to load any valid samples from {len(files)} files.")
    
    # Padding
    max_h = max(t.shape[1] for t in Xs); max_w = max(t.shape[2] for t in Xs)
    def pad(t_list):
        return torch.stack([torch.nn.functional.pad(t, (0, max_w-t.shape[2], 0, max_h-t.shape[1])) for t in t_list])
    X_all, Y_all, M_all = pad(Xs), pad(Ys), pad(Masks)
    
    dataset = TensorDataset(X_all, Y_all, M_all)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if is_distributed else None
    loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler, shuffle=(sampler is None))

    # Model
    model = HybridFNO(in_channels=X_all.shape[1], hidden_channels=64).to(device)
    if is_distributed: model = DDP(model, device_ids=[local_rank])
    
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    l2_loss = LpLoss(d=2, p=2)
    h1_loss = H1Loss(d=2)

    if is_main_process(rank):
        print(f"Hybrid FNO Training Started on {world_size} GPUs")
        logger = TrainingLogger(output_dir="training_logs")
    
    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        if is_distributed: sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            pred = model(xb)
            
            # Hybrid Loss
            loss_data = l2_loss(pred * mb, yb * mb)
            loss_grad = h1_loss(pred * mb, yb * mb)
            loss_phys = physics_informed_loss(pred, yb, xb, device)
            
            loss = loss_data + 0.3 * loss_grad + 0.1 * loss_phys
            
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item() * xb.shape[0]

        # Aggregate & Log
        if is_distributed:
            loss_tensor = torch.tensor(running_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            running_loss = loss_tensor.item()
        
        avg_loss = running_loss / len(dataset)
        if is_main_process(rank):
            print(f"Epoch {epoch}/{EPOCHS} Loss: {avg_loss:.6e}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.module.state_dict() if is_distributed else model.state_dict(), MODEL_OUT)

    cleanup_distributed()

if __name__ == "__main__":
    main()
