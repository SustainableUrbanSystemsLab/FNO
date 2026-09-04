# Multi-GPU Training for FNO (using DistributedDataParallel)
# Usage: torchrun --nproc_per_node=2 train_fno_distributed.py
#    or: python train_fno_distributed.py (falls back to single GPU)

import os, glob, numpy as np, torch, sys, argparse, time
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from multiprocessing import cpu_count

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.models.fno2d import FNO2d, sensor_weighted_mse
from core.utils.training_logger import TrainingLogger
from core.utils.target_norm import normalize_k_channels

# ============ Load Configuration ============
import argparse

# Pre-parse just the --config argument (before main argparse in main())
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--config', type=str, default='config.toml', help='Config file to use')
_pre_args, _ = _pre_parser.parse_known_args()
CONFIG_FILE = _pre_args.config

from core.utils.config_loader import load_config

config = load_config()

# ============ Configuration from file ============
# Auto-detect platform and use appropriate data folder
if sys.platform == 'win32':
    DATA_FOLDER = config.get('paths', {}).get('data_folder_windows', 'train_csv')
else:
    # Smart detection for cluster environment
    linux_path = config.get('paths', {}).get('data_folder_linux', 'train_csv')
    ice_path = config.get('paths', {}).get('data_folder_ice', '')

    # Prefer ICE path if on Linux and it exists
    if ice_path and os.path.exists(ice_path):
        DATA_FOLDER = ice_path
        print(f"Detected ICE Cluster path: {DATA_FOLDER}")
    elif os.path.exists(linux_path):
        DATA_FOLDER = linux_path
        print(f"Detected Linux/PACE path: {DATA_FOLDER}")
    else:
        # Fallback
        DATA_FOLDER = linux_path
        print(f"Warning: Neither ICE nor Linux paths found. Defaulting to: {DATA_FOLDER}")

MODEL_OUT = config.get('paths', {}).get('model_output', 'fno_mag_weights.pth')
CHECKPOINT_PATH = config.get('paths', {}).get('checkpoint_file', 'epochs/checkpoint_latest.pth')
EPOCHS_DIR = "epochs"
BATCH = config.get('training', {}).get('batch_size', 4)
EPOCHS = config.get('training', {}).get('epochs', 200)
LR = config.get('training', {}).get('learning_rate', 1e-3)
PATIENCE = config.get('training', {}).get('patience', 50)
CHECKPOINT_INTERVAL = config.get('training', {}).get('checkpoint_interval', 10)
MODES1 = config.get('model', {}).get('modes1', 32)
MODES2 = config.get('model', {}).get('modes2', 32)
WIDTH = config.get('model', {}).get('width', 64)
N_LAYERS = config.get('model', {}).get('n_layers', 5)
# FIX: physics terms now ramp from 0 over WARMUP_EPOCHS
# to prevent the model collapsing to a flat constant field early in training.
GRAD_WEIGHT     = config.get('loss', {}).get('gradient_weight', 1.5)
SPECTRAL_WEIGHT = config.get('loss', {}).get('spectral_weight', 0.001)
PEAK_WEIGHT     = config.get('loss', {}).get('peak_weight', 0.5)
WAKE_WEIGHT     = config.get('loss', {}).get('wake_weight', 1.0)
WARMUP_EPOCHS   = config.get('loss', {}).get('warmup_epochs', 10)
CHANNEL_WEIGHTS = config.get('loss', {}).get('channel_weights', None)

def get_loss_weights(epoch):
    # Linearly ramp physics weights from 0 to their max over WARMUP_EPOCHS.
    t = min(epoch / max(WARMUP_EPOCHS, 1), 1.0)
    return dict(
        grad_weight=GRAD_WEIGHT * t,
        spectral_weight=SPECTRAL_WEIGHT * t,
        peak_weight=PEAK_WEIGHT * t,
        wake_weight=WAKE_WEIGHT * t,
    )

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

# ============ Dataset for monolithic X.npy / Y.npy ============
#
# The X.npy/Y.npy on ICE were created by the Conditional Transformer's
# convert_to_npz.py and have THIS channel layout:
#   Raw[0]: X_coords         (raw meters, -485 to 521)
#   Raw[1]: Y_coords         (raw meters, -514 to 491)
#   Raw[2]: Z_relative       (raw, 0-1.8)
#   Raw[3]: SDF              (raw, can be negative inside buildings)
#   Raw[4]: Bldg_height      (raw meters, 0-152)
#   Raw[5]: U_over_Uref      (raw, ~0-0.26)
#   Raw[6]: dir_sin          (-1 to 1)
#   Raw[7]: dir_cos          (-1 to 1)
#   Y[0]:   mag_U            (raw wind magnitude)
#
# The FNO expects (from build_input_tensor_from_gh):
#   FNO[0]: SDF / 200          (0-1)
#   FNO[1]: Bldg_height / 50   (0-1)
#   FNO[2]: Z_relative / 10    (0-1)
#   FNO[3]: U_over_Uref * 2    (0.4-2.0)
#   FNO[4]: X_local / 500      (-1 to 1, building-centered)
#   FNO[5]: Y_local / 500      (-1 to 1, building-centered)
#   FNO[6]: dir_sin            (-1 to 1)
#   FNO[7]: dir_cos            (-1 to 1)
#   Y ch0:  delta_u       = (mag_U      - U_ref) / U_ref, clipped [-2, 10]
#   Y ch1:  k             (raw, non-negative)
#   Y ch2:  delta_u_roof  = (mag_U_roof - U_ref) / U_ref, clipped [-2, 10]
#   Y ch3:  k_roof        (raw, non-negative)
#
# This dataset remaps on-the-fly.

# Output channel metadata (matching Conditional Transformer repo)
CHANNEL_NAMES  = ["U", "k", "U_roof", "k_roof"]
CHANNEL_VMAXES = [2.0, 0.5, 2.0, 0.5]
# "pedestrian" channels are only physically meaningful outside building
# footprints; "roof" channels only exist on building footprints. Matches the
# masking convention documented in tools/infer_csv.py.
CHANNEL_ROLES  = ["pedestrian", "pedestrian", "roof", "roof"]


def build_channel_mask(xb, out_ch):
    """Build a (B, out_ch, H, W) loss mask matching each output channel's
    valid physical region, derived from the FNO-format input tensor `xb`.

    Bug this fixes: every training loop previously built a 1-channel mask
    (`torch.ones_like(xb[:, 0:1])`) and passed it into a loss whose numerator
    sums squared error over ALL output channels but whose denominator summed
    only that 1-channel mask -- silently inflating mse_loss by ~out_ch once
    Y went from 1 to 4 channels, and skipping any building/roof masking.
    Returning a mask shaped exactly like y_pred/y_target fixes both at once.

    For out_ch <= 1 this returns the original full-domain ones mask,
    preserving single-channel training behavior exactly.
    """
    if out_ch <= 1:
        return torch.ones_like(xb[:, 0:1, :, :])

    bldg = (xb[:, 1:2, :, :] > 0).float()   # FNO Ch1 = Bldg_height / 50
    dom = (xb[:, 3:4, :, :] > 0).float()    # FNO Ch3 = U_over_Uref * 2: zero outside the circular CFD domain
    open_ground = dom * (1.0 - bldg)        # the corners outside the circle carry no CFD data
    role_mask = {"pedestrian": open_ground, "roof": bldg}

    channels = [
        role_mask[CHANNEL_ROLES[i]] if i < len(CHANNEL_ROLES) else torch.ones_like(bldg)
        for i in range(out_ch)
    ]
    return torch.cat(channels, dim=1)


# Transformer -> FNO channel index mapping
_RAW_X_COORDS  = 0
_RAW_Y_COORDS  = 1
_RAW_Z_REL     = 2
_RAW_SDF       = 3
_RAW_BLDG_H    = 4
_RAW_U_REF     = 5
_RAW_DIR_SIN   = 6
_RAW_DIR_COS   = 7

class NpyDataset(Dataset):
    """Dataset that loads Transformer-format X.npy/Y.npy and remaps to FNO format.

    Performs on-the-fly:
      1. Channel reordering (Transformer order -> FNO order)
      2. Physical normalization (SDF/200, Height/50, etc.)
      3. Building-centered coordinate computation
      4. Target conversion: mag_U -> delta_u = (mag - U_ref) / U_ref
    """
    def __init__(self, X_path, Y_path, augment=False, subset=None):
        if not os.path.exists(X_path):
            raise FileNotFoundError(f"X.npy not found at {X_path}")
        if not os.path.exists(Y_path):
            raise FileNotFoundError(f"Y.npy not found at {Y_path}")

        # Memory-map for low RAM usage
        self.X = np.load(X_path, mmap_mode='r')  # (N, 8, H, W)
        self.Y = np.load(Y_path, mmap_mode='r')  # (N, C_y, H, W)
        self.augment = augment
        # data-scaling study: optional .npy of row indices into this split (benchmark/make_subsets.py)
        self.idx = np.load(subset).astype(np.int64) if subset else None
        if self.idx is not None:
            assert self.idx.max() < self.X.shape[0], "subset indices outside the split"
            print(f"[NpyDataset] subset {subset}: {self.idx.size} of {self.X.shape[0]} samples")

        assert self.X.shape[0] == self.Y.shape[0], \
            f"X/Y sample count mismatch: {self.X.shape[0]} vs {self.Y.shape[0]}"

        # Auto-detect format: if Ch0 range >> 1, it's Transformer format (raw coords)
        ch0_max = np.abs(self.X[0, 0]).max()
        self.needs_remap = ch0_max > 10.0  # SDF/200 would be 0-1; raw X_coords is ~500
        if self.needs_remap:
            print("[NpyDataset] Detected Transformer-format data. Will remap to FNO format on-the-fly.")
        else:
            print("[NpyDataset] Data appears to already be in FNO format (Ch0 range < 10).")

    def __len__(self):
        return int(self.idx.size) if self.idx is not None else self.X.shape[0]

    def _remap_to_fno(self, raw_x, raw_y):
        """Convert Transformer channel layout + raw mag_U to FNO format."""
        H, W = raw_x.shape[1], raw_x.shape[2]
        out_x = np.zeros_like(raw_x)  # (8, H, W)

        # --- Channel remapping with physical normalization ---
        sdf_raw = raw_x[_RAW_SDF]          # Can be negative inside buildings
        bldg_h  = raw_x[_RAW_BLDG_H]
        z_rel   = raw_x[_RAW_Z_REL]
        u_ref   = raw_x[_RAW_U_REF]
        x_coord = raw_x[_RAW_X_COORDS]
        y_coord = raw_x[_RAW_Y_COORDS]

        # FNO Ch0: SDF / 200
        out_x[0] = sdf_raw / 200.0

        # FNO Ch1: Bldg_height / 50
        out_x[1] = bldg_h / 50.0

        # FNO Ch2: Z_relative / 10
        out_x[2] = z_rel / 10.0

        # FNO Ch3: U_over_Uref * 2
        out_x[3] = u_ref * 2.0

        # FNO Ch4,5: Building-centered coordinates / 500
        bldg_mask = bldg_h > 0
        if bldg_mask.any():
            x_center = x_coord[bldg_mask].mean()
            y_center = y_coord[bldg_mask].mean()
        else:
            x_center = x_coord.mean()
            y_center = y_coord.mean()
        out_x[4] = (x_coord - x_center) / 500.0
        out_x[5] = (y_coord - y_center) / 500.0

        # FNO Ch6,7: dir_sin, dir_cos (already correct)
        out_x[6] = raw_x[_RAW_DIR_SIN]
        out_x[7] = raw_x[_RAW_DIR_COS]

        # --- Target remapping (4-channel: mag_U, k, mag_U_roof, k_roof) ---
        out_y = np.zeros_like(raw_y)  # (C_y, H, W)
        has_data = u_ref > 1e-6

        # Ch0: mag_U -> delta_u = (mag_U - U_ref) / U_ref
        mag_u = raw_y[0]
        out_y[0, has_data] = (mag_u[has_data] - u_ref[has_data]) / (u_ref[has_data] + 1e-6)
        out_y[0] = np.clip(out_y[0], -2.0, 10.0)

        # Ch1: k (TKE) — clamp non-negative, then z-score normalize.
        # Raw TKE is on a very different scale than the clipped [-2, 10]
        # delta_u channels; left un-normalized it made mse_loss weight the
        # 4 output channels very unevenly. See core/utils/target_norm.py.
        if raw_y.shape[0] > 1:
            out_y[1] = np.maximum(raw_y[1], 0.0)

        # Ch2: mag_U_roof -> delta_u_roof (same transform as ch0)
        if raw_y.shape[0] > 2:
            mag_u_roof = raw_y[2]
            out_y[2, has_data] = (mag_u_roof[has_data] - u_ref[has_data]) / (u_ref[has_data] + 1e-6)
            out_y[2] = np.clip(out_y[2], -2.0, 10.0)

        # Ch3: k_roof (TKE roof) — clamp non-negative, then z-score normalize
        if raw_y.shape[0] > 3:
            out_y[3] = np.maximum(raw_y[3], 0.0)

        normalize_k_channels(out_y)

        return out_x, out_y

    def __getitem__(self, idx):
        if self.idx is not None:
            idx = int(self.idx[idx])
        x = np.array(self.X[idx], copy=True, dtype=np.float32)  # (8, H, W)
        y = np.array(self.Y[idx], copy=True, dtype=np.float32)  # (C_y, H, W)

        # Remap from Transformer format to FNO format if needed
        if self.needs_remap:
            x, y = self._remap_to_fno(x, y)

        # Optional augmentation: random horizontal flip
        if self.augment and np.random.random() > 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, :, ::-1].copy()
            # Negate X_local (ch4) and dir_sin (ch6) for horizontal flip
            x[4] = -x[4]
            x[6] = -x[6]

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(x), torch.from_numpy(y)

# ============ Main Training ============
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FNO Training')
    parser.add_argument('--fresh', action='store_true', help='Start fresh training (ignore checkpoint)')
    parser.add_argument('--config', type=str, default='config.toml', help='Config file to use')
    parser.add_argument('--val_dir', type=str, default=None, help='Directory containing validation X.npy and Y.npy')
    parser.add_argument('--reset-patience', action='store_true', help='Reset best_loss and patience (use when changing loss function)')
    parser.add_argument('--train-subset', type=str, default=None, help='.npy of row indices into the train split (data-scaling study)')
    parser.add_argument('--epochs', type=int, default=None, help='override training.epochs (scaled with the subset size to keep the step budget)')
    parser.add_argument('--patience', type=int, default=None, help='override training.patience (epochs)')
    args = parser.parse_args()
    global EPOCHS, PATIENCE
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.patience is not None:
        PATIENCE = args.patience

    local_rank, rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process(rank):
        print(f"Training with {world_size} GPU(s)")
        print(f"Distributed: {is_distributed}")

    # Cleanup old files (only on main process, and only when explicitly starting
    # fresh -- otherwise this would delete the checkpoint the resume path below
    # is about to load)
    if is_main_process(rank) and args.fresh:
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

    # Create output directories
    os.makedirs(EPOCHS_DIR, exist_ok=True)

    # Load dataset from monolithic X.npy / Y.npy
    x_path = os.path.join(DATA_FOLDER, 'X.npy')
    y_path = os.path.join(DATA_FOLDER, 'Y.npy')

    if is_main_process(rank):
        print(f"Loading dataset from {DATA_FOLDER}...")
        print(f"  X path: {x_path}")
        print(f"  Y path: {y_path}")

    if args.val_dir and os.path.exists(os.path.join(args.val_dir, 'X.npy')):
        train_dataset = NpyDataset(x_path, y_path, augment=True, subset=args.train_subset)
        val_dataset = NpyDataset(os.path.join(args.val_dir, 'X.npy'), os.path.join(args.val_dir, 'Y.npy'), augment=False)
        total_samples = len(train_dataset) + len(val_dataset)
        train_dataset_full = train_dataset
        if is_main_process(rank): print(f"Using explicitly specified val_dir: {args.val_dir}", flush=True)
    else:
        train_dataset_full = NpyDataset(x_path, y_path, augment=True)
        val_dataset_full = NpyDataset(x_path, y_path, augment=False)

        # Train/Val split (configurable from config.toml)
        VAL_SPLIT = config.get('training', {}).get('val_split', 0.1)
        total_samples = len(train_dataset_full)
        train_size = int((1.0 - VAL_SPLIT) * total_samples)
        val_size = total_samples - train_size

        if is_main_process(rank):
            print(f"Applying {((1.0 - VAL_SPLIT)*100):.0f}/{VAL_SPLIT*100:.0f} random train/val split natively (from config). Train: {train_size}, Val: {val_size}")

        # Fixed seed for deterministic split across nodes
        indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42)).tolist()
        train_idx, val_idx = indices[:train_size], indices[train_size:]

        train_dataset = Subset(train_dataset_full, train_idx)
        val_dataset = Subset(val_dataset_full, val_idx)

    # sample indexing to get input/output channel counts
    sample_x, sample_y = train_dataset[0]
    in_ch = sample_x.shape[0]  # Should be 8
    out_ch = sample_y.shape[0] # Dynamic (1, 2, or 4)

    if is_main_process(rank):
        print("=" * 50)
        print("DATASET INITIALIZATION COMPLETE")
        print("=" * 50)
        print(f"  Total samples: {total_samples}")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples:   {len(val_dataset)}")
        print(f"  Input channels: {in_ch}")
        print(f"  Output channels: {out_ch}")
        print(f"  Remap active: {train_dataset_full.needs_remap}")

        # Sanity check: run one sample through __getitem__ to show POST-REMAP ranges
        print("  --- Post-remap sample check (sample 0) ---")
        try:
            sample_x, sample_y = train_dataset_full[0]
            ch_names = ['SDF/200', 'Height/50', 'Z_rel/10', 'U_ref*2', 'X_loc/500', 'Y_loc/500', 'dir_sin', 'dir_cos']
            for ch_idx, ch_name in enumerate(ch_names):
                ch = sample_x[ch_idx]
                print(f"    Ch{ch_idx} ({ch_name}): [{ch.min():.4f}, {ch.max():.4f}] mean={ch.mean():.4f}")
            print(f"    Y (delta_u): [{sample_y.min():.4f}, {sample_y.max():.4f}] mean={sample_y.mean():.4f}")
        except Exception as e:
            print(f"  !!! ERROR in sample check: {e}")
            import traceback
            traceback.print_exc()
        print("=" * 50)

    # Create DataLoaders
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH, sampler=train_sampler, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH, sampler=val_sampler, num_workers=NUM_WORKERS)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    if is_main_process(rank):
        print("CREATING MODEL...")
        print(f"  FNO2d: modes=({MODES1},{MODES2}), width={WIDTH}, layers={N_LAYERS}")

    # Create model
    model = FNO2d(in_channels=in_ch, out_channels=out_ch, modes1=MODES1, modes2=MODES2,
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

        # Override if reset requested
        if args.reset_patience:
            best_loss = float('inf')
            patience_counter = 0
            if is_main_process(rank):
                print("  [INFO] Patience and best_loss RESET via --reset-patience")

            # FORCE NEW HYPERPARAMETERS
            # 1. Update Optimizer LR
            for param_group in opt.param_groups:
                param_group['lr'] = LR
            if is_main_process(rank):
                print(f"  [INFO] Learning rate updated to {LR}")

            # 2. Reset Scheduler (to respect new epochs and LR)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
            if is_main_process(rank):
                print(f"  [INFO] Scheduler reset with T_max={EPOCHS}")


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
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = TrainingLogger(output_dir="training_logs", experiment_name=f"STANDARD_{_ts}")
        logger.start_training({
            'dataset_size': total_samples,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'batch_size': BATCH,
            'epochs': EPOCHS,
            'learning_rate': LR,
            'patience': PATIENCE,
            'modes1': MODES1,
            'modes2': MODES2,
            'width': WIDTH,
            'n_layers': N_LAYERS,
        }, model=model.module if is_distributed else model)
    else:
        logger = None

    # Training loop

    # Determine end epoch
    if args.reset_patience:
        # If resetting, we treat EPOCHS as "additional epochs" or "new phase duration"
        target_end_epoch = start_epoch + EPOCHS - 1
        if is_main_process(rank):
            print(f"  [INFO] Training for {EPOCHS} additional epochs (End epoch: {target_end_epoch})")
    else:
        # Standard behavior: EPOCHS is the total absolute epoch count
        target_end_epoch = EPOCHS
        if start_epoch > target_end_epoch:
            if is_main_process(rank):
                print(f"  [WARNING] Start epoch {start_epoch} > Total epochs {EPOCHS}. Training may exit immediately.")
                print(f"  [TIP] Use --reset-patience to treat 'epochs' as a new phase duration.")

    if is_main_process(rank):
        sys.stdout.flush()  # Ensure all printout appears in log before training starts

    # History tracking for plot_comparison_curves.py
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, target_end_epoch + 1):
        epoch_start = time.time()
        if is_distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False) if is_main_process(rank) else train_loader

        # Accumulators for loss components
        running_mse = 0.0
        running_grad = 0.0
        running_spec = 0.0
        running_peak = 0.0
        running_wake = 0.0

        for batch in pbar:
            # NpyDataset returns (x_tensor, y_tensor) directly
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)

            mb = build_channel_mask(xb, out_ch)

            pred = model(xb)

            # Use epoch-dependent warmup weights
            w = get_loss_weights(epoch)
            loss, components = sensor_weighted_mse(pred, yb, sensor_mask=mb,
                                                grad_weight=w['grad_weight'],
                                                spectral_weight=w['spectral_weight'],
                                                peak_weight=w['peak_weight'],
                                                wake_weight=w['wake_weight'],
                                                channel_weights=CHANNEL_WEIGHTS,
                                                return_components=True)
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_size = xb.shape[0]
            running += float(loss.item()) * batch_size
            running_mse += components['mse_loss'] * batch_size
            running_grad += components['gradient_loss'] * batch_size
            running_spec += components['spectral_loss'] * batch_size
            running_peak += components.get('peak_loss', 0.0) * batch_size
            running_wake += components.get('wake_loss', 0.0) * batch_size

            if is_main_process(rank) and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({"loss": f"{loss.item():.4e}"})

        # --- Validation Pass ---
        model.eval()
        val_running = 0.0
        val_se, val_n = 0.0, 0.0  # masked squared error of the scored quantity (U / U_ref)
        with torch.no_grad():
            for batch in val_loader:
                xb, yb = batch
                xb = xb.to(device); yb = yb.to(device)
                mb = build_channel_mask(xb, out_ch)
                pred = model(xb)
                # benchmark-aligned selection metric: RMSE of U on open-ground domain
                # cells; delta_u -> U/U_ref via the constant inlet ratio 0.26
                m0 = (mb[:, 0] > 0) & (xb[:, 3] > 0)
                err_u = 0.26 * (pred[:, 0] - yb[:, 0])
                val_se += float((err_u[m0] ** 2).sum().item())
                val_n += float(m0.sum().item())
                # Validation always uses the fully ramped weights so the val loss is
                # comparable across epochs; with the training ramp, the epoch-1 loss
                # was the smallest by construction and stayed "best" forever.
                w = get_loss_weights(WARMUP_EPOCHS)
                v_loss = sensor_weighted_mse(pred, yb, sensor_mask=mb,
                                            grad_weight=w['grad_weight'],
                                            spectral_weight=w['spectral_weight'],
                                            peak_weight=w['peak_weight'],
                                            wake_weight=w['wake_weight'],
                                            channel_weights=CHANNEL_WEIGHTS)
                val_running += float(v_loss.item()) * xb.shape[0]

        scheduler.step()
        n_train = len(train_dataset)
        n_val = len(val_dataset)

        # Aggregate loss across GPUs
        if is_distributed:
            # Sync GPUs before collective operation to prevent timeout
            torch.cuda.synchronize()
            dist.barrier()

            # Aggregate all loss components across GPUs
            running_tensor = torch.tensor([running, running_mse, running_grad, running_spec, running_peak, running_wake, val_running, val_se, val_n], device=device)
            dist.all_reduce(running_tensor, op=dist.ReduceOp.SUM)
            running = running_tensor[0].item()
            running_mse = running_tensor[1].item()
            running_grad = running_tensor[2].item()
            running_spec = running_tensor[3].item()
            running_peak = running_tensor[4].item()
            running_wake = running_tensor[5].item()
            val_running = running_tensor[6].item()
            val_se = running_tensor[7].item()
            val_n = running_tensor[8].item()

        avg_loss = running / n_train
        avg_mse = running_mse / n_train
        avg_grad = running_grad / n_train
        avg_spec = running_spec / n_train
        avg_peak = running_peak / n_train
        avg_wake = running_wake / n_train
        avg_val_loss = val_running / n_val
        avg_val_rmse_u = (val_se / max(val_n, 1.0)) ** 0.5

        if is_main_process(rank):
            epoch_duration = time.time() - epoch_start
            print(f"Epoch {epoch}/{target_end_epoch} loss {avg_loss:.6e} val_loss {avg_val_loss:.6e} val_rmse_u {avg_val_rmse_u:.5f} ({epoch_duration:.2f}s)")

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
                print(f"  > Latest checkpoint saved to {CHECKPOINT_PATH}")

            # Early stopping
            # CRITICAL FIX: All ranks must track patience and decide to stop together.
            # Since avg_loss is synchronized (all_reduce), everyone sees the same value.

            if is_main_process(rank):
                train_losses.append(avg_loss)
                val_losses.append(avg_val_loss)

            # select and early-stop on the scored quantity, not the 4-channel composite
            # in which mag_U is only a few percent of the value
            if avg_val_rmse_u < best_loss:
                best_loss = avg_val_rmse_u
                patience_counter = 0

                # Only Rank 0 saves the model
                if is_main_process(rank):
                    state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                    # Payload including training history for tools/plot_all_histories.py
                    payload = {
                        'model_state_dict': state_dict,
                        'history': {
                            'train_loss': train_losses,
                            'val_loss': val_losses,
                            'epoch': epoch
                        },
                        'config': {
                            'modes': (MODES1, MODES2),
                            'width': WIDTH,
                            'n_layers': N_LAYERS
                        }
                    }
                    temp_out = MODEL_OUT + ".tmp"
                    torch.save(payload, temp_out)
                    if os.path.exists(MODEL_OUT):
                        try: os.remove(MODEL_OUT)
                        except: pass
                    os.rename(temp_out, MODEL_OUT)
                    print(f"  > New best loss! Saved {MODEL_OUT} (Epoch {epoch})")

            else:
                patience_counter += 1
                if is_main_process(rank):
                    print(f"  > No improvement. Patience {patience_counter}/{PATIENCE}")

                if patience_counter >= PATIENCE:
                    if is_main_process(rank):
                        print(f"Early stopping at epoch {epoch}")
                    break

            # Log epoch metrics for publication (only Rank 0)
            if is_main_process(rank) and logger:
                logger.log_epoch(epoch, {
                    'total_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'val_rmse_u': avg_val_rmse_u,
                    'mse_loss': avg_mse,
                    'gradient_loss': avg_grad,
                    'spectral_loss': avg_spec,
                    'peak_loss': avg_peak,
                    'wake_loss': avg_wake,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'epoch_time': epoch_duration,
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
                from core.utils.generate_plots import generate_publication_plots
                generate_publication_plots(logger.metrics_csv)
            except Exception as e:
                print(f"[Plots] Could not generate plots: {e}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
