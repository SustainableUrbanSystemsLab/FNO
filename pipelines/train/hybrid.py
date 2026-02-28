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
import os, sys, torch, numpy as np, pandas as pd, tomllib, argparse, glob, hashlib, pickle, traceback, time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Local imports
from core.models.fno2d import FNO2d, sensor_weighted_mse
from core.models.hybrid import HybridFNO
from core.utils.gh_to_fno import build_input_tensor_from_gh
from core.utils.training_logger import TrainingLogger
from neuralop.losses import LpLoss, H1Loss

# ============ Load Configuration ============
def load_config(config_file):
    """Load configuration from toml file."""
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', config_file))
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

class HybridNumpyDataset(torch.utils.data.Dataset):
    """
    Lazy-loading dataset for large mmap'd Numpy arrays.
    Prevents OOM by only loading batches into RAM.
    """
    def __init__(self, x_path, y_path, sdf_scaling):
        self.x_mmap = np.load(x_path, mmap_mode='c')
        self.y_mmap = np.load(y_path, mmap_mode='c')
        self.sdf_scaling = sdf_scaling
        
    def __len__(self):
        return self.x_mmap.shape[0]
        
    def __getitem__(self, idx):
        # We use .copy() to get an owned array that Pytorch can convert to a tensor properly
        x = torch.from_numpy(self.x_mmap[idx].copy()).float()
        y = torch.from_numpy(self.y_mmap[idx].copy()).float()
        
        # Compute mask on-the-fly (Fast for a single sample)
        # SDF is channel 0
        sdf_meters = x[0:1, :, :] * self.sdf_scaling
        mask = 1.0 + 19.0 * torch.exp(-torch.clamp(sdf_meters, min=0.0) / 5.0)
        
        return x, y, mask

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

    try:
        # 1. Path detection
        if sys.platform == 'win32':
            DATA_FOLDER = config.get('paths', {}).get('data_folder_windows', 'train_csv')
        else:
            linux_path = config.get('paths', {}).get('data_folder_linux', '')
            ice_path = config.get('paths', {}).get('data_folder_ice', '')
            
            if is_main_process(rank):
                print(f"[Diag] Raw Config - ICE: {ice_path}, Linux: {linux_path}", file=sys.stderr)
            
            if ice_path: ice_path = os.path.expanduser(ice_path)
            if linux_path: linux_path = os.path.expanduser(linux_path)
            
            def check_path(p, name):
                if not p: return False
                exists = os.path.exists(p)
                isdir = os.path.isdir(p)
                has_npy = False
                if exists and isdir:
                    has_npy = os.path.exists(os.path.join(p, "X.npy")) and os.path.exists(os.path.join(p, "Y.npy"))
                
                if is_main_process(rank):
                    msg = f"[Diag] {name}: {p} | Exists: {exists}, IsDir: {isdir}, HasNPY: {has_npy}"
                    print(msg, flush=True)
                    print(msg, file=sys.stderr, flush=True)
                return exists and isdir and has_npy

            if check_path(ice_path, "ICE Path"):
                DATA_FOLDER = ice_path
            elif check_path(linux_path, "Linux Path"):
                DATA_FOLDER = linux_path
            else:
                guesses = ["/storage/ice1/2/4/scratch/Training_Dataset", "/storage/ice1/2/4/athach7/Training_Dataset"]
                for guess in guesses:
                    if check_path(guess, "Hardcoded Guess"):
                        DATA_FOLDER = guess
                        break
                else:
                    DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'train_csv')
                    if is_main_process(rank):
                        print(f"[Diag] Falling back to local: {DATA_FOLDER}", file=sys.stderr, flush=True)
                
        if is_main_process(rank):
            print(f"Using Data Folder: {os.path.abspath(DATA_FOLDER)}", flush=True)

        # 2. Config & Training Params
        MODEL_OUT = "hybrid_fno_weights.pth"
        BATCH = config.get('training', {}).get('batch_size', 4)
        EPOCHS = config.get('training', {}).get('epochs', 1000)
        LR = config.get('training', {}).get('learning_rate', 5e-4)

        MODES1 = config.get('model', {}).get('modes1', 32)
        MODES2 = config.get('model', {}).get('modes2', 32)
        WIDTH = config.get('model', {}).get('width', 64)
        
        GRAD_WEIGHT = config.get('loss', {}).get('gradient_weight', 0.15)
        SPECTRAL_WEIGHT = config.get('loss', {}).get('spectral_weight', 0.05)
        PEAK_WEIGHT = config.get('loss', {}).get('peak_weight', 0.0)
        WAKE_WEIGHT = config.get('loss', {}).get('wake_weight', 0.0)
        CHECKPOINT_INTERVAL = config.get('training', {}).get('checkpoint_interval', 10)
        EPOCHS_DIR = "epochs"

        if is_main_process(rank):
            os.makedirs(EPOCHS_DIR, exist_ok=True)
            os.makedirs("training_logs", exist_ok=True)

        # 3. Data Prep (Hybrid Lazy Loading)
        x_npy = os.path.join(DATA_FOLDER, "X.npy")
        y_npy = os.path.join(DATA_FOLDER, "Y.npy")
        
        if os.path.exists(x_npy) and os.path.exists(y_npy):
            if is_main_process(rank):
                print(f"Loading via HybridNumpyDataset (mmap)...", flush=True)
            temp_x = np.load(x_npy, mmap_mode='r')
            sdf_max = temp_x[0, 0].max()
            sdf_scaling = 200.0 if sdf_max <= 5.0 else 1.0
            dataset = HybridNumpyDataset(x_npy, y_npy, sdf_scaling)
        else:
            if is_main_process(rank):
                print(f"Falling back to CSV loading (Memory Intensive)...", flush=True)
            files = sorted(glob.glob(os.path.join(DATA_FOLDER, "**/*.csv"), recursive=True))
            if not files: raise RuntimeError(f"No Data found in {DATA_FOLDER}")
            num_workers = max(1, cpu_count() // 2)
            Xs, Ys, Masks, _ = load_or_prepare_dataset(files, rank, is_main_process(rank), num_workers)
            max_h = max(t.shape[1] for t in Xs); max_w = max(t.shape[2] for t in Xs)
            def pad(t_list):
                return torch.stack([torch.nn.functional.pad(t, (0, max_w-t.shape[2], 0, max_h-t.shape[1])) for t in t_list])
            dataset = TensorDataset(pad(Xs), pad(Ys), pad(Masks))
        
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if is_distributed else None
        loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler, shuffle=(sampler is None), num_workers=2 if not is_distributed else 1)

        # 4. Model & Optimization
        sample_x, _, _ = dataset[0]
        model = HybridFNO(in_channels=sample_x.shape[0], 
                          n_modes=(MODES1, MODES2),
                          hidden_channels=WIDTH).to(device)
        
        if os.path.exists(MODEL_OUT) and not args.fresh:
            if is_main_process(rank): print(f"Resuming weights: {MODEL_OUT}", flush=True)
            model.load_state_dict(torch.load(MODEL_OUT, map_location=device, weights_only=False), strict=False)

        if is_distributed: model = DDP(model, device_ids=[local_rank])
        
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        # Add scheduler for consistency with standard training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
        
        l2_loss = LpLoss(d=2, p=2); h1_loss = H1Loss(d=2)
        if is_main_process(rank): logger = TrainingLogger(output_dir="training_logs")
        
        best_loss = float('inf')
        for epoch in range(1, EPOCHS + 1):
            if is_distributed: sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            for xb, yb, mb in loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb)
                
                loss, components = sensor_weighted_mse(
                    pred, yb, sensor_mask=mb,
                    grad_weight=GRAD_WEIGHT,
                    spectral_weight=SPECTRAL_WEIGHT,
                    peak_weight=PEAK_WEIGHT,
                    wake_weight=WAKE_WEIGHT,
                    return_components=True
                )
                
                opt.zero_grad(); loss.backward(); opt.step()
                running_loss += loss.item() * xb.shape[0]

            if is_distributed:
                loss_tensor = torch.tensor(running_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                running_loss = loss_tensor.item()
            
            scheduler.step()
            avg_loss = running_loss / len(dataset)
            if is_main_process(rank):
                print(f"Epoch {epoch}/{EPOCHS} Loss: {avg_loss:.6e}", flush=True)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.module.state_dict() if is_distributed else model.state_dict(), MODEL_OUT)
                    print(f"   * Best model saved (Loss: {best_loss:.6e})", flush=True)

                if epoch % CHECKPOINT_INTERVAL == 0:
                    ckpt_path = os.path.join(EPOCHS_DIR, f"hybrid_epoch_{epoch}.pth")
                    torch.save(model.module.state_dict() if is_distributed else model.state_dict(), ckpt_path)
                    print(f"  > Periodic checkpoint saved: {ckpt_path}", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR on Rank {rank}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        if is_distributed: cleanup_distributed()
        sys.exit(1)

    cleanup_distributed()

if __name__ == "__main__":
    main()
