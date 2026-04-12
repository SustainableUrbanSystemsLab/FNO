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
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset, Subset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Local imports
from core.models.fno2d import FNO2d, sensor_weighted_mse
from core.models.hybrid import HybridFNO
from core.utils.training_logger import TrainingLogger
from neuralop.losses import LpLoss, H1Loss

# Share dataset logic
from pipelines.train.distributed import NpyDataset

# ============ Load Configuration ============
from core.utils.config_loader import load_config

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
    parser.add_argument('--val_dir', type=str, default=None, help='Directory containing validation X.npy/Y.npy')
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
                _user = os.environ.get('USER', 'unknown')
                guesses = [
                    f"/home/hice1/{_user}/scratch/Training_Dataset",
                    f"/storage/ice1/2/4/{_user}/Training_Dataset",
                    "/storage/ice1/2/4/scratch/Training_Dataset",
                ]
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
        N_LAYERS = config.get('model', {}).get('n_layers', 4)
        
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
            
<<<<<<< HEAD
        train_dataset_full = NpyDataset(x_path, y_path, augment=True)
        val_dataset_full = NpyDataset(x_path, y_path, augment=False)
        
        # Train/Val split (configurable from config.toml)
        VAL_SPLIT = config.get('training', {}).get('val_split', 0.1)
        total_samples = len(train_dataset_full)
        train_size = int((1.0 - VAL_SPLIT) * total_samples)
        val_size = total_samples - train_size
        
        # Fixed seed for deterministic split across nodes
        indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42)).tolist()
        train_idx, val_idx = indices[:train_size], indices[train_size:]
        
        train_dataset = Subset(train_dataset_full, train_idx)
        val_dataset = Subset(val_dataset_full, val_idx)
        
        if is_distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            loader = DataLoader(train_dataset, batch_size=BATCH, sampler=train_sampler, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=BATCH, sampler=val_sampler, num_workers=2)
        else:
            train_sampler = None
            loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=2)

        # 4. Model & Optimization
        sample_x, _ = train_dataset_full[0]
=======
        if args.val_dir and os.path.exists(os.path.join(args.val_dir, 'X.npy')):
            train_dataset = NpyDataset(x_path, y_path, augment=True)
            val_dataset = NpyDataset(os.path.join(args.val_dir, 'X.npy'), os.path.join(args.val_dir, 'Y.npy'), augment=False)
            if is_main_process(rank): print(f"Using explicitly specified val_dir: {args.val_dir}", flush=True)
        else:
            full_dataset = NpyDataset(x_path, y_path, augment=True)
            val_dataset_full = NpyDataset(x_path, y_path, augment=False)
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            indices = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(42)).tolist()
            from torch.utils.data import Subset
            train_dataset = Subset(full_dataset, indices[:train_size])
            val_dataset = Subset(val_dataset_full, indices[train_size:])
            if is_main_process(rank): print(f"Using 90/10 random train/val split natively. Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
        
        loader = DataLoader(train_dataset, batch_size=BATCH, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=2 if not is_distributed else 1)
        val_loader = DataLoader(val_dataset, batch_size=BATCH, sampler=val_sampler, shuffle=False, num_workers=2 if not is_distributed else 1)

        # 4. Model & Optimization
        sample_x, _ = train_dataset[0]
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e
        model = HybridFNO(in_channels=sample_x.shape[0], 
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
            logger = TrainingLogger(output_dir="training_logs", experiment_name=f"HYBRID_{_ts}")
        
        best_loss = float('inf')
        train_losses = []
        val_losses = []
        for epoch in range(1, EPOCHS + 1):
            if is_distributed: train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            running_mse = 0.0
            running_grad = 0.0
            running_spec = 0.0
            running_peak = 0.0
            running_wake = 0.0
            epoch_start = time.time()
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
                n = xb.shape[0]
                n = xb.shape[0]
                running_loss += loss.item() * n
                running_mse  += components['mse_loss'] * n
                running_grad += components['gradient_loss'] * n
                running_spec += components['spectral_loss'] * n
                running_peak += components.get('peak_loss', 0.0) * n
                running_wake += components.get('wake_loss', 0.0) * n
            
            # --- Validation Pass ---
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    xb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
                    sdf = xb[:, 0:1, :, :]
                    mb = torch.where(sdf > 0, torch.ones_like(sdf), torch.full_like(sdf, 0.2))
                    pred = model(xb)
                    w = get_loss_weights(epoch)
                    v_loss, _ = sensor_weighted_mse(pred, yb, sensor_mask=mb, 
                                                grad_weight=w['grad_weight'], 
                                                spectral_weight=w['spectral_weight'], 
                                                peak_weight=w['peak_weight'],
                                                wake_weight=w['wake_weight'],
                                                return_components=True)
                    val_running += v_loss.item() * xb.shape[0]

            if is_distributed:
                torch.cuda.synchronize()
                dist.barrier()
<<<<<<< HEAD
=======
            
            # --- EVALUATION PASS ---
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    xb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
                    sdf = xb[:, 0:1, :, :]
                    mb = torch.where(sdf > 0, torch.ones_like(sdf), torch.full_like(sdf, 0.2))
                    pred = model(xb)
                    w = get_loss_weights(epoch)
                    v_loss = sensor_weighted_mse(pred, yb, sensor_mask=mb,
                                        grad_weight=w['grad_weight'], spectral_weight=w['spectral_weight'],
                                        peak_weight=w['peak_weight'], wake_weight=w['wake_weight'], wake_threshold=-0.5)
                    val_running += float(v_loss.item()) * xb.shape[0]

            if is_distributed:
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e
                running_tensor = torch.tensor([running_loss, running_mse, running_grad, running_spec, running_peak, running_wake, val_running], device=device)
                dist.all_reduce(running_tensor, op=dist.ReduceOp.SUM)
                running_loss = running_tensor[0].item()
                running_mse  = running_tensor[1].item()
                running_grad = running_tensor[2].item()
                running_spec = running_tensor[3].item()
                running_peak = running_tensor[4].item()
                running_wake = running_tensor[5].item()
                val_running  = running_tensor[6].item()

            scheduler.step()
<<<<<<< HEAD
            n_train = len(train_dataset)
            n_val = len(val_dataset)
            avg_loss = running_loss / n_train
            avg_mse  = running_mse  / n_train
            avg_grad = running_grad / n_train
            avg_spec = running_spec / n_train
            avg_peak = running_peak / n_train
            avg_wake = running_wake / n_train
            avg_val_loss = val_running / n_val
            epoch_duration = time.time() - epoch_start

            if is_main_process(rank):
                print(f"Epoch {epoch}/{EPOCHS} Loss: {avg_loss:.6e} Val Loss: {avg_val_loss:.6e}", flush=True)
=======
            n_samples = len(train_dataset)
            avg_loss = running_loss / n_samples
            avg_mse  = running_mse  / n_samples
            avg_grad = running_grad / n_samples
            avg_spec = running_spec / n_samples
            avg_peak = running_peak / n_samples
            avg_wake = running_wake / n_samples
            avg_val_loss = val_running / len(val_dataset)
            epoch_duration = time.time() - epoch_start

            if is_main_process(rank):
                print(f"Epoch {epoch}/{EPOCHS} Loss: {avg_loss:.6e} | Val Loss: {avg_val_loss:.6e}", flush=True)
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e

                # Log epoch metrics
                logger.log_epoch(epoch, {
                    'total_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'mse_loss':      avg_mse,
                    'gradient_loss': avg_grad,
                    'spectral_loss': avg_spec,
                    'peak_loss':     avg_peak,
                    'wake_loss':     avg_wake,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'epoch_time_sec': epoch_duration,
                    'best_loss': best_loss,
                })
                train_losses.append(avg_loss)
<<<<<<< HEAD
                val_losses.append(avg_val_loss)
                
=======
                val_losses.append(avg_val_loss) 

>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    
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
                            'n_layers': N_LAYERS
                        }
                    }
                    
                    temp_out = MODEL_OUT + ".tmp"
                    torch.save(payload, temp_out)
                    if os.path.exists(MODEL_OUT): os.remove(MODEL_OUT)
                    os.rename(temp_out, MODEL_OUT)
                    print(f"   * Best model saved (Loss: {best_loss:.6e})", flush=True)

                if epoch % CHECKPOINT_INTERVAL == 0:
                    ckpt_path = os.path.join(EPOCHS_DIR, "rolling_hybrid_checkpoint.pth")
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

    if is_main_process(rank):
        print(f"Training finished. Best loss: {best_loss:.6e}")
        # Finalize training logger
        if 'logger' in locals():
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
