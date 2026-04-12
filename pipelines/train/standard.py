# Train FNO to predict dimensionless magnitude (mag_U)
import os, glob, numpy as np, pandas as pd, torch, sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, Subset

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.models.fno2d import FNO2d, sensor_weighted_mse
from pipelines.train.distributed import NpyDataset

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

# ============ Data Loading ============
x_path = os.path.join(DATA_FOLDER, 'X.npy')
y_path = os.path.join(DATA_FOLDER, 'Y.npy')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--val_dir', type=str, default=None)
parser.add_argument('--fresh', action='store_true')
_args, _ = parser.parse_known_args()

if _args.val_dir and os.path.exists(os.path.join(_args.val_dir, 'X.npy')):
    train_dataset = NpyDataset(x_path, y_path, augment=True)
    val_dataset = NpyDataset(os.path.join(_args.val_dir, 'X.npy'), os.path.join(_args.val_dir, 'Y.npy'), augment=False)
    if RANK == 0: print(f"Using explicitly specified val_dir: {_args.val_dir}", flush=True)
else:
    full_dataset = NpyDataset(x_path, y_path, augment=True)
    val_dataset_full = NpyDataset(x_path, y_path, augment=False)
    VAL_SPLIT = config.get('training', {}).get('val_split', 0.1)
    total_samples = len(full_dataset)
    train_size = int((1.0 - VAL_SPLIT) * total_samples)
    val_size = total_samples - train_size
    indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42)).tolist()
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, indices[:train_size])
    val_dataset = Subset(val_dataset_full, indices[train_size:])
    if RANK == 0: print(f"Using {((1.0 - VAL_SPLIT)*100):.0f}/{VAL_SPLIT*100:.0f} random train/val split natively. Train: {train_size}, Val: {val_size}", flush=True)

if IS_DISTRIBUTED:
    train_sampler = DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
    loader = DataLoader(train_dataset, batch_size=BATCH, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, sampler=val_sampler, num_workers=2, pin_memory=True)
else:
    train_sampler = None
    loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=2)

    sample_x, _ = train_dataset[0]
in_ch = sample_x.shape[0]
chs = ['SDF', 'Height', 'Z_rel', 'U_ref', 'X_loc', 'Y_loc', 'sin', 'cos']

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

    # Only show progress bar on Rank 0
    if RANK == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    else:
        pbar = loader

    if IS_DISTRIBUTED:
        train_sampler.set_epoch(epoch)

    # Accumulators for loss components
    running_mse = 0.0
    running_grad = 0.0
    running_spec = 0.0
    running_peak = 0.0
    running_wake = 0.0

    start_time = time.time()

    for batch in pbar:
        xb, yb = batch
        xb, yb = xb.to(DEVICE).float(), yb.to(DEVICE).float()

        # Build mask from SDF channel
        sdf = xb[:, 0:1, :, :]
        mb = torch.where(sdf > 0, torch.ones_like(sdf), torch.full_like(sdf, 0.2))

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
        running_mse += components['mse_loss'] * batch_size
        running_grad += components['gradient_loss'] * batch_size
        running_spec += components['spectral_loss'] * batch_size
        running_peak += components.get('peak_loss', 0.0) * batch_size
        running_wake += components.get('wake_loss', 0.0) * batch_size

        if RANK == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4e}"})
<<<<<<< HEAD

    # --- Validation Pass ---
=======

    if IS_DISTRIBUTED:
        dist.barrier()

    # --- EVALUATION PASS ---
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for batch in val_loader:
            xb, yb = batch
            xb, yb = xb.to(DEVICE).float(), yb.to(DEVICE).float()
            sdf = xb[:, 0:1, :, :]
            mb = torch.where(sdf > 0, torch.ones_like(sdf), torch.full_like(sdf, 0.2))
            pred = model(xb)
            w = get_loss_weights(epoch)
<<<<<<< HEAD
            v_loss = sensor_weighted_mse(pred, yb, sensor_mask=mb,
                                        grad_weight=w['grad_weight'],
                                        spectral_weight=w['spectral_weight'],
                                        peak_weight=w['peak_weight'],
                                        wake_weight=w['wake_weight'])
            val_running += float(v_loss.item()) * xb.shape[0]
=======
            v_loss = sensor_weighted_mse(pred, yb, sensor_mask=mb,
                                        grad_weight=w['grad_weight'], spectral_weight=w['spectral_weight'],
                                        peak_weight=w['peak_weight'], wake_weight=w['wake_weight'], wake_threshold=-0.5)
            val_running += float(v_loss.item()) * xb.shape[0]

    if IS_DISTRIBUTED:
        running_tensor = torch.tensor([running, val_running], device=DEVICE)
        dist.all_reduce(running_tensor, op=dist.ReduceOp.SUM)
        running = running_tensor[0].item()
        val_running = running_tensor[1].item()
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e

    scheduler.step()
    epoch_time = time.time() - start_time

<<<<<<< HEAD
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # Aggregate losses across GPUs
    if IS_DISTRIBUTED:
        torch.cuda.synchronize()
        dist.barrier()
        running_tensor = torch.tensor([running, running_mse, running_grad, running_spec, running_peak, running_wake, val_running], device=DEVICE)
        dist.all_reduce(running_tensor, op=dist.ReduceOp.SUM)
        running = running_tensor[0].item()
        running_mse = running_tensor[1].item()
        running_grad = running_tensor[2].item()
        running_spec = running_tensor[3].item()
        running_peak = running_tensor[4].item()
        running_wake = running_tensor[5].item()
        val_running = running_tensor[6].item()

    avg_loss = running / n_train
    avg_mse = running_mse / n_train
    avg_grad = running_grad / n_train
    avg_spec = running_spec / n_train
    avg_peak = running_peak / n_train
    avg_wake = running_wake / n_train
    avg_val_loss = val_running / n_val
=======
    avg_loss = running / len(train_dataset)
    avg_val_loss = val_running / len(val_dataset)

    # Calculate average components
    dataset_len = len(train_dataset)
    avg_components = {k: v / dataset_len for k, v in running_comp.items()}
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e

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
<<<<<<< HEAD
=======
    # Check for improvement (Early Stopping based on Validation)
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
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
        print(f"  > No improvement. Patience {patience_counter}/{PATIENCE} (Val Loss: {avg_val_loss:.6e})")
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            save_feature_importance(model, chs, epoch_num=epoch) # Save at early stop too
            early_stop = True # Flag to break after logging
        else:
            early_stop = False

    # Prepare metrics for logger (updated with new patience)
    metrics = {
        'total_loss': avg_loss,
        'val_loss': avg_val_loss,
<<<<<<< HEAD
        'mse_loss': avg_mse,
        'gradient_loss': avg_grad,
        'spectral_loss': avg_spec,
        'peak_loss': avg_peak,
        'wake_loss': avg_wake,
=======
        'mse_loss': avg_components['mse_loss'],
        'gradient_loss': avg_components['gradient_loss'],
        'spectral_loss': avg_components['spectral_loss'],
        'peak_loss': avg_components['peak_loss'],
        'wake_loss': avg_components.get('wake_loss', 0.0),
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e
        'learning_rate': opt.param_groups[0]['lr'],
        'epoch_time_sec': epoch_time,
        'best_loss': best_loss,
        'patience_counter': patience_counter
    }

    if RANK == 0:
        logger.log_epoch(epoch, metrics)
<<<<<<< HEAD
        print(f"Epoch {epoch}/{EPOCHS} loss {avg_loss:.6e} val_loss {avg_val_loss:.6e} (Peak: {avg_peak:.6e})")
=======
        print(f"Epoch {epoch}/{EPOCHS} Train Loss {avg_loss:.6e} | Val Loss {avg_val_loss:.6e}")
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e

    if 'early_stop' in locals() and early_stop:
        break

# Finish Logger
if RANK == 0:
    logger.finish_training()

print(f"Training finished. Best loss: {best_loss:.6e}")
print("Saved best model to:", MODEL_OUT)
# Also save feature importance to main output
save_feature_importance(model, chs)
