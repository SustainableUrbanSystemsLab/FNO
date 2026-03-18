"""
PINN-FNO Training Pipeline
===========================
Trains the Physics-Informed FNO on the wind field dataset.

Usage:
    bash slurm/deploy_ice.sh --script pipelines/train/pinn.py --gpu h100 --ngpus 2 --fresh
"""

import os, sys, torch, numpy as np, pandas as pd, tomllib, argparse, glob, hashlib, pickle, traceback, time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.models.pinn_fno import PINNFNO, pinn_loss
from core.utils.gh_to_fno import build_input_tensor_from_gh
from core.utils.training_logger import TrainingLogger


# ============ Config ============
def load_config(config_file):
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', config_file))
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            return tomllib.load(f)
    return {}


# ============ Distributed ============
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank        = int(os.environ['RANK'])
        world_size  = int(os.environ['WORLD_SIZE'])
        local_rank  = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size, True
    return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main(rank):
    return rank == 0


# ============ Dataset ============
class PINNNumpyDataset(torch.utils.data.Dataset):
    """Lazy-loading mmap dataset for PINN training."""
    def __init__(self, x_path, y_path, sdf_scaling=200.0):
        self.X = np.load(x_path, mmap_mode='c')
        self.Y = np.load(y_path, mmap_mode='c')
        self.sdf_scaling = sdf_scaling

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.from_numpy(self.Y[idx].copy()).float()
        sdf_meters = x[0:1, :, :] * self.sdf_scaling
        mask = 1.0 + 19.0 * torch.exp(-torch.clamp(sdf_meters, min=0.0) / 5.0)
        return x, y, mask

class PINNNPZDataset(torch.utils.data.Dataset):
    """Dataset for a list of .npz files."""
    def __init__(self, files, sdf_scaling=200.0):
        self.files = files
        self.sdf_scaling = sdf_scaling
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            x = torch.from_numpy(data['X']).float()
            y = torch.from_numpy(data['Y']).float()
            if x.ndim == 4: x = x.squeeze(0)
            if y.ndim == 4: y = y.squeeze(0)
            sdf_meters = x[0:1, :, :] * self.sdf_scaling
            mask = 1.0 + 19.0 * torch.exp(-torch.clamp(sdf_meters, min=0.0) / 5.0)
            return x, y, mask


# ============ Main ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.toml')
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    local_rank, rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    try:
        # --- Paths ---
        if sys.platform == 'win32':
            DATA_FOLDER = config.get('paths', {}).get('data_folder_windows', 'train_csv')
        else:
            ice_path   = os.path.expanduser(config.get('paths', {}).get('data_folder_ice', ''))
            linux_path = os.path.expanduser(config.get('paths', {}).get('data_folder_linux', 'train_csv'))
            if ice_path and os.path.exists(os.path.join(ice_path, 'X.npy')):
                DATA_FOLDER = ice_path
                if is_main(rank): print(f"Using ICE path: {DATA_FOLDER}", flush=True)
            elif os.path.exists(os.path.join(linux_path, 'X.npy')):
                DATA_FOLDER = linux_path
            else:
                DATA_FOLDER = linux_path

        MODEL_OUT   = 'pinn_fno_weights.pth'
        EPOCHS_DIR  = 'epochs'
        if is_main(rank):
            os.makedirs(EPOCHS_DIR, exist_ok=True)
            os.makedirs('training_logs', exist_ok=True)

        # --- Hyperparams ---
        BATCH    = config.get('training', {}).get('batch_size', 4)
        EPOCHS   = config.get('training', {}).get('epochs', 1000)
        LR       = config.get('training', {}).get('learning_rate', 1e-4)
        PATIENCE = config.get('training', {}).get('patience', 100)
        CKPT_INT = config.get('training', {}).get('checkpoint_interval', 10)

        MODES1   = config.get('model', {}).get('modes1', 32)
        MODES2   = config.get('model', {}).get('modes2', 32)
        WIDTH    = config.get('model', {}).get('width', 64)
        N_LAYERS = config.get('model', {}).get('n_layers', 4)

        # PINN-specific weights (can be added to config.toml later)
        GRAD_W   = config.get('loss', {}).get('gradient_weight', 2.0)
        WAKE_W   = config.get('loss', {}).get('wake_weight', 1.0)
        PEAK_W   = config.get('loss', {}).get('peak_weight', 0.5)
        CONT_W   = config.get('loss', {}).get('continuity_weight', 0.1)
        MOM_W    = config.get('loss', {}).get('momentum_weight', 0.05)

        if is_main(rank):
            print("=" * 60, flush=True)
            print("PINN-FNO Training", flush=True)
            print(f"  Modes: {MODES1}x{MODES2}, Width: {WIDTH}, Layers: {N_LAYERS}", flush=True)
            print(f"  Physics: continuity_w={CONT_W}, momentum_w={MOM_W}", flush=True)
            print(f"  Wake weight: {WAKE_W}, Gradient weight: {GRAD_W}", flush=True)
            print("=" * 60, flush=True)

        # --- Dataset ---
        x_npy = os.path.join(DATA_FOLDER, 'X.npy')
        y_npy = os.path.join(DATA_FOLDER, 'Y.npy')

        if os.path.exists(x_npy) and os.path.exists(y_npy):
            if is_main(rank): print(f"Loading via PINNNumpyDataset (mmap)...", flush=True)
            temp_x   = np.load(x_npy, mmap_mode='r')
            sdf_max  = float(temp_x[0, 0].max())
            sdf_scale = 200.0 if sdf_max <= 5.0 else 1.0
            dataset  = PINNNumpyDataset(x_npy, y_npy, sdf_scale)
        else:
            npz_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "**/*.npz"), recursive=True))
            if not npz_files:
                raise RuntimeError(f"No .npy or .npz data found in {DATA_FOLDER}")
            if is_main(rank): print(f"Loading {len(npz_files)} .npz files...", flush=True)
            # Sample one to get scaling
            with np.load(npz_files[0]) as data:
                sdf_max = float(data['X'][0, 0].max())
            sdf_scale = 200.0 if sdf_max <= 5.0 else 1.0
            dataset = PINNNPZDataset(npz_files, sdf_scale)

        if is_main(rank):
            print(f"Dataset size: {len(dataset)} samples", flush=True)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if is_distributed else None
        loader  = DataLoader(dataset, batch_size=BATCH, sampler=sampler,
                             shuffle=(sampler is None), num_workers=2, pin_memory=True)

        # --- Model ---
        sample_x, _, _ = dataset[0]
        model = PINNFNO(
            in_channels=sample_x.shape[0],
            n_modes=(MODES1, MODES2),
            hidden_channels=WIDTH,
            n_layers=N_LAYERS
        ).to(device)

        if os.path.exists(MODEL_OUT) and not args.fresh:
            if is_main(rank): print(f"Resuming from {MODEL_OUT}", flush=True)
            state_dict = torch.load(MODEL_OUT, map_location=device, weights_only=False)
            model.load_state_dict(state_dict, strict=False)

        if is_distributed:
            model = DDP(model, device_ids=[local_rank])

        opt       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

        if is_main(rank):
            logger = TrainingLogger(output_dir='training_logs')

        # --- Training Loop ---
        best_loss      = float('inf')
        patience_count = 0

        for epoch in range(1, EPOCHS + 1):
            if is_distributed: sampler.set_epoch(epoch)
            model.train()
            running = 0.0
            comp_accum = {k: 0.0 for k in ['mse_loss','gradient_loss','continuity_loss',
                                            'momentum_loss','wake_loss','peak_loss']}
            epoch_start = time.time()

            for xb, yb, mb in loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb)

                loss, comps = pinn_loss(
                    pred, yb, x_input=xb, sensor_mask=mb,
                    grad_weight=GRAD_W,
                    continuity_weight=CONT_W,
                    momentum_weight=MOM_W,
                    wake_weight=WAKE_W,
                    peak_weight=PEAK_W
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                bs = xb.shape[0]
                running += loss.item() * bs
                for k in comp_accum:
                    comp_accum[k] += comps.get(k, 0.0) * bs

            # Sync across GPUs
            if is_distributed:
                t = torch.tensor(running, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                running = t.item()

            scheduler.step()
            n = len(dataset)
            avg_loss = running / n

            if is_main(rank):
                dur = time.time() - epoch_start
                cont_avg = comp_accum['continuity_loss'] / n
                wake_avg = comp_accum['wake_loss'] / n
                print(
                    f"Epoch {epoch:4d}/{EPOCHS} | Loss: {avg_loss:.4e} "
                    f"| Cont: {cont_avg:.4e} | Wake: {wake_avg:.4e} | ({dur:.1f}s)",
                    flush=True
                )

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_count = 0
                    sd = model.module.state_dict() if is_distributed else model.state_dict()
                    torch.save(sd, MODEL_OUT)
                    print(f"  ★ Best model saved (loss={best_loss:.4e})", flush=True)
                else:
                    patience_count += 1
                    if patience_count >= PATIENCE:
                        print(f"Early stopping at epoch {epoch}", flush=True)
                        break

                if epoch % CKPT_INT == 0:
                    sd = model.module.state_dict() if is_distributed else model.state_dict()
                    ckpt = os.path.join(EPOCHS_DIR, f'pinn_epoch_{epoch}.pth')
                    torch.save(sd, ckpt)
                    print(f"  > Checkpoint: {ckpt}", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR rank {rank}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        if is_distributed: cleanup_distributed()
        sys.exit(1)

    cleanup_distributed()


if __name__ == '__main__':
    main()
