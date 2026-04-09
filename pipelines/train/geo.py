"""
Distributed Training for Geometry-Aware FNO (GeoFNO) Model
=========================================
Features:
- PyTorch Distributed Data Parallel (DDP)
- Physics-Informed Loss (Divergence / Continuity)
- Hashed Dataset Caching
- Slurm/ICE Cluster Integration
- Multiprocessing Data Preparation
"""
import os, sys, torch, numpy as np, pandas as pd, tomllib, argparse, glob, hashlib, pickle, traceback, time
from datetime import datetime
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
from core.models.geo_fno import GeoFNO
from core.utils.training_logger import TrainingLogger
from neuralop.losses import LpLoss, H1Loss

# Share dataset logic
from pipelines.train.distributed import NpyDataset

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
        MODEL_OUT = "geo_fno_weights.pth"
        BATCH = config.get('training', {}).get('batch_size', 4)
        EPOCHS = config.get('training', {}).get('epochs', 1000)
        LR = config.get('training', {}).get('learning_rate', 5e-4)

        MODES1 = config.get('model', {}).get('modes1', 32)
        MODES2 = config.get('model', {}).get('modes2', 32)
        WIDTH = config.get('model', {}).get('width', 64)
        
        GRAD_WEIGHT = config.get('loss', {}).get('gradient_weight', 0.5)
        SPECTRAL_WEIGHT = config.get('loss', {}).get('spectral_weight', 0.05)
        PEAK_WEIGHT = config.get('loss', {}).get('peak_weight', 0.3)
        WAKE_WEIGHT = config.get('loss', {}).get('wake_weight', 0.3)
        WARMUP_EPOCHS = config.get('loss', {}).get('warmup_epochs', 50)
        CHECKPOINT_INTERVAL = config.get('training', {}).get('checkpoint_interval', 10)

        def get_loss_weights(epoch):
            # Linearly ramp physics weights from 0 to their max over WARMUP_EPOCHS.
            # Pure MSE for first 50 epochs ensures stable foundation.
            t = min(epoch / max(WARMUP_EPOCHS, 1), 1.0)
            return dict(
                grad_weight=GRAD_WEIGHT * t,
                spectral_weight=SPECTRAL_WEIGHT * t,
                peak_weight=PEAK_WEIGHT * t,
                wake_weight=WAKE_WEIGHT * t,
            )
        EPOCHS_DIR = "epochs"

        if is_main_process(rank):
            os.makedirs(EPOCHS_DIR, exist_ok=True)
            os.makedirs("training_logs", exist_ok=True)

        # 3. Data Prep
        x_path = os.path.join(DATA_FOLDER, 'X.npy')
        y_path = os.path.join(DATA_FOLDER, 'Y.npy')
        
        if is_main_process(rank):
            print(f"Loading dataset from {DATA_FOLDER}...")
            print(f"  X path: {x_path}")
            print(f"  Y path: {y_path}")
            
        dataset = NpyDataset(x_path, y_path, augment=True)
        
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if is_distributed else None
        loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler, shuffle=(sampler is None), num_workers=2 if not is_distributed else 1)

        # 4. Model & Optimization
        sample_x, _ = dataset[0]
        model = GeoFNO(in_channels=sample_x.shape[0], 
                          n_modes=(MODES1, MODES2),
                          hidden_channels=WIDTH).to(device)
        
        if os.path.exists(MODEL_OUT) and not args.fresh:
            if is_main_process(rank): print(f"Resuming weights: {MODEL_OUT}", flush=True)
            try:
                saved = torch.load(MODEL_OUT, map_location=device, weights_only=False)
                result = model.load_state_dict(saved, strict=True)
                if is_main_process(rank): print("  Weights loaded successfully (strict=True)", flush=True)
            except RuntimeError as e:
                if is_main_process(rank):
                    print(f"  WARNING: Weight mismatch detected — starting FRESH (use --fresh to suppress this).", flush=True)
                    print(f"  Reason: {e}", file=sys.stderr, flush=True)
                # Architecture changed: don't load incompatible weights at all

        if is_distributed: model = DDP(model, device_ids=[local_rank])
        
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        # Add scheduler for consistency with standard training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
        
        l2_loss = LpLoss(d=2, p=2); h1_loss = H1Loss(d=2)
        if is_main_process(rank): 
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger = TrainingLogger(output_dir="training_logs", experiment_name=f"GEO_{_ts}")
            logger.start_training({
                'batch_size': BATCH,
                'epochs': EPOCHS,
                'learning_rate': LR,
                'modes': (MODES1, MODES2),
                'width': WIDTH,
            }, model=model.module if is_distributed else model)
        
        best_loss = float('inf')
        train_losses = []
        val_losses = []
        for epoch in range(1, EPOCHS + 1):
            if is_distributed: sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            for batch in loader:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                
                # Build mask from SDF channel (Channel 0, physically normalized: SDF/200)
                sdf = xb[:, 0:1, :, :]
                mb = torch.where(sdf > 0, torch.ones_like(sdf), torch.full_like(sdf, 0.2))
                
                pred = model(xb)
                
                # FIX: use epoch-dependent warmup weights
                w = get_loss_weights(epoch)
                loss, components = sensor_weighted_mse(
                    pred, yb, sensor_mask=mb,
                    grad_weight=w['grad_weight'],
                    spectral_weight=w['spectral_weight'],
                    peak_weight=w['peak_weight'],
                    wake_weight=w['wake_weight'],
                    wake_threshold=-0.5, # Deep wake deficits
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
                
                # Log epoch metrics for plotting
                logger.log_epoch(epoch, {
                    'total_loss': avg_loss,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'best_loss': best_loss,
                })
                train_losses.append(avg_loss)
                val_losses.append(avg_loss) # Tracking training for now

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    
                    # Payload including training history for tools/plot_comparison_curves.py
                    state_dict = model.module.state_dict() if is_distributed else model.state_dict()
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
                            'n_layers': config.get('model', {}).get('n_layers', 4)
                        }
                    }
                    
                    temp_out = MODEL_OUT + ".tmp"
                    torch.save(payload, temp_out)
                    if os.path.exists(MODEL_OUT): os.remove(MODEL_OUT)
                    os.rename(temp_out, MODEL_OUT)
                    print(f"   * Best model saved (Loss: {best_loss:.6e})", flush=True)

                if epoch % CHECKPOINT_INTERVAL == 0:
                    ckpt_path = os.path.join(EPOCHS_DIR, "rolling_geo_checkpoint.pth")
                    temp_ckpt = ckpt_path + ".tmp"
                    torch.save(model.module.state_dict() if is_distributed else model.state_dict(), temp_ckpt)
                    if os.path.exists(ckpt_path): os.remove(ckpt_path)
                    os.rename(temp_ckpt, ckpt_path)
                    print(f"  > Periodic checkpoint overwritten: {ckpt_path}", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR on Rank {rank}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        if is_distributed: cleanup_distributed()
        sys.exit(1)

    cleanup_distributed()

if __name__ == "__main__":
    main()
