# Multi-GPU Training for FNO (using DistributedDataParallel)
# Usage: torchrun --nproc_per_node=2 train_fno_distributed.py
#    or: python train_fno_distributed.py (falls back to single GPU)

import os, glob, numpy as np, torch, sys, argparse, time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from multiprocessing import cpu_count

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.models.fno2d import FNO2d, sensor_weighted_mse
from core.utils.training_logger import TrainingLogger

# ============ Load Configuration ============
import argparse

# Pre-parse just the --config argument (before main argparse in main())
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--config', type=str, default='config.toml', help='Config file to use')
_pre_args, _ = _pre_parser.parse_known_args()
CONFIG_FILE = _pre_args.config

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
GRAD_WEIGHT     = config.get('loss', {}).get('gradient_weight', 0.5) 
SPECTRAL_WEIGHT = config.get('loss', {}).get('spectral_weight', 0.05)
PEAK_WEIGHT     = config.get('loss', {}).get('peak_weight', 0.3)
WAKE_WEIGHT     = config.get('loss', {}).get('wake_weight', 0.3) 
WARMUP_EPOCHS   = config.get('loss', {}).get('warmup_epochs', 50)

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
# Channel layout (from build_input_tensor_from_gh / regenerate_dataset.py):
#   X[0]: SDF / 200          (0-1, 0=building boundary)
#   X[1]: Bldg_height / 50   (0-1)
#   X[2]: Z_relative / 10    (0-1)
#   X[3]: U_over_Uref * 2    (0.4-2.0)
#   X[4]: X_local / 500      (-1 to 1, building-centered)
#   X[5]: Y_local / 500      (-1 to 1, building-centered)
#   X[6]: dir_sin            (-1 to 1)
#   X[7]: dir_cos            (-1 to 1)
# Target Y: delta_u = (mag_U - U_ref) / U_ref, clipped [-2, 10]

class NpyDataset(Dataset):
    """Dataset that loads from monolithic X.npy and Y.npy arrays.
    
    These files are produced by tools/regenerate_dataset.py which uses
    build_input_tensor_from_gh for physical normalization and computes
    delta_u = (mag - U_ref) / U_ref as the target.
    """
    def __init__(self, X_path, Y_path, augment=False):
        if not os.path.exists(X_path):
            raise FileNotFoundError(f"X.npy not found at {X_path}")
        if not os.path.exists(Y_path):
            raise FileNotFoundError(f"Y.npy not found at {Y_path}")
        
        # Memory-map for low RAM usage
        self.X = np.load(X_path, mmap_mode='r')  # (N, 8, H, W)
        self.Y = np.load(Y_path, mmap_mode='r')  # (N, 1, H, W)
        self.augment = augment
        
        assert self.X.shape[0] == self.Y.shape[0], \
            f"X/Y sample count mismatch: {self.X.shape[0]} vs {self.Y.shape[0]}"
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = np.array(self.X[idx], copy=True, dtype=np.float32)  # (8, H, W)
        y = np.array(self.Y[idx], copy=True, dtype=np.float32)  # (1, H, W)
        
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
    parser.add_argument('--reset-patience', action='store_true', help='Reset best_loss and patience (use when changing loss function)')
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
 
    # Create output directories
    os.makedirs(EPOCHS_DIR, exist_ok=True)
    
    # Load dataset from monolithic X.npy / Y.npy
    x_path = os.path.join(DATA_FOLDER, 'X.npy')
    y_path = os.path.join(DATA_FOLDER, 'Y.npy')
    
    if is_main_process(rank):
        print(f"Loading dataset from {DATA_FOLDER}...")
        print(f"  X path: {x_path}")
        print(f"  Y path: {y_path}")
    
    train_dataset = NpyDataset(x_path, y_path, augment=True)
    in_ch = train_dataset.X.shape[1]  # Should be 8
    
    if is_main_process(rank):
        print("=" * 50)
        print("DATASET INITIALIZATION COMPLETE")
        print("=" * 50)
        print(f"  Total samples: {len(train_dataset)}")
        print(f"  X shape: {train_dataset.X.shape}")
        print(f"  Y shape: {train_dataset.Y.shape}")
        print(f"  Input channels: {in_ch}")
        # Print channel ranges for sanity check
        for ch_idx, ch_name in enumerate(['SDF/200', 'Height/50', 'Z_rel/10', 'U_ref*2', 'X_loc/500', 'Y_loc/500', 'dir_sin', 'dir_cos']):
            ch_data = train_dataset.X[:, ch_idx]
            print(f"    Ch{ch_idx} ({ch_name}): [{ch_data.min():.3f}, {ch_data.max():.3f}]")
        y_data = train_dataset.Y[:, 0]
        print(f"    Y (delta_u): [{y_data.min():.3f}, {y_data.max():.3f}] mean={y_data.mean():.3f}")
        print("=" * 50)

    # Create DataLoader
    if is_distributed:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(train_dataset, batch_size=BATCH, sampler=sampler, num_workers=NUM_WORKERS)
    else:
        loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)

    if is_main_process(rank):
        print("CREATING MODEL...")
        print(f"  FNO2d: modes=({MODES1},{MODES2}), width={WIDTH}, layers={N_LAYERS}")
        
    # Create model
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
        logger = TrainingLogger(output_dir="training_logs", is_main=is_main_process(rank))
        logger.start_training({
            'batch_size': BATCH,
            'epochs': EPOCHS,
            'learning_rate': LR,
            'patience': PATIENCE,
            'modes1': MODES1,
            'modes2': MODES2,
            'width': WIDTH,
            'n_layers': N_LAYERS,
            'dataset_size': len(train_dataset),
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

    for epoch in range(start_epoch, target_end_epoch + 1):
        epoch_start = time.time()
        if is_distributed:
            sampler.set_epoch(epoch)
        
        model.train()
        running = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False) if is_main_process(rank) else loader
        
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
            
            # Build mask from SDF channel (Channel 0, physically normalized: SDF/200)
            # SDF=0 means building boundary, SDF>0 means outside building
            # After /200 normalization, SDF>0 still correctly identifies exterior
            sdf = xb[:, 0:1, :, :]
            mb = torch.where(sdf > 0, torch.ones_like(sdf), torch.full_like(sdf, 0.2))
            
            pred = model(xb)
            
            # Use epoch-dependent warmup weights
            w = get_loss_weights(epoch)
            loss, components = sensor_weighted_mse(pred, yb, sensor_mask=mb, 
                                                grad_weight=w['grad_weight'], 
                                                spectral_weight=w['spectral_weight'],
                                                peak_weight=w['peak_weight'],
                                                wake_weight=w['wake_weight'],
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
        
        scheduler.step()
        n_samples = len(train_dataset)
        
        # Aggregate loss across GPUs
        if is_distributed:
            # Sync GPUs before collective operation to prevent timeout
            torch.cuda.synchronize()
            dist.barrier()
            
            # Aggregate all loss components across GPUs
            running_tensor = torch.tensor([running, running_mse, running_grad, running_spec, running_peak, running_wake], device=device)
            dist.all_reduce(running_tensor, op=dist.ReduceOp.SUM)
            running = running_tensor[0].item()
            running_mse = running_tensor[1].item()
            running_grad = running_tensor[2].item()
            running_spec = running_tensor[3].item()
            running_peak = running_tensor[4].item()
            running_wake = running_tensor[5].item()
        
        avg_loss = running / n_samples
        avg_mse = running_mse / n_samples
        avg_grad = running_grad / n_samples
        avg_spec = running_spec / n_samples
        avg_peak = running_peak / n_samples
        avg_wake = running_wake / n_samples
        
        if is_main_process(rank):
            epoch_duration = time.time() - epoch_start
            print(f"Epoch {epoch}/{target_end_epoch} loss {avg_loss:.6e} ({epoch_duration:.2f}s)")
            
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
                
                # Also save epoch-specific checkpoint for safety
                epoch_ckpt = os.path.join(EPOCHS_DIR, f"checkpoint_epoch_{epoch}.pth")
                torch.save(checkpoint, epoch_ckpt)
                print(f"  > Epoch checkpoint saved to {epoch_ckpt}")

            # Early stopping
            # CRITICAL FIX: All ranks must track patience and decide to stop together.
            # Since avg_loss is synchronized (all_reduce), everyone sees the same value.
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Only Rank 0 saves the model
                if is_main_process(rank):
                    state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                    temp_out = MODEL_OUT + ".tmp"
                    torch.save(state_dict, temp_out)
                    if os.path.exists(MODEL_OUT):
                        try: os.remove(MODEL_OUT)
                        except: pass
                    os.rename(temp_out, MODEL_OUT)
                    print(f"  > New best loss! Saved {MODEL_OUT}")
                    
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
