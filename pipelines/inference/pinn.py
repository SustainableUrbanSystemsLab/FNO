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
        # FIX: detect whether Y is absolute Mag_U (needs Delta_U conversion) or already correct.
        # Absolute Mag_U has no negatives and mean ~1.0; Delta_U is centred near 0.
        _y_sample = self.Y[:min(64, self.Y.shape[0])].ravel()
        self.convert_to_delta = (float(_y_sample.min()) >= -0.05)
        if self.convert_to_delta:
            print("[PINNNumpyDataset] Y appears to be absolute Mag_U (min>=0). "
                  "Will convert to Delta_U = (Mag_U - U_ref) / U_ref on-the-fly.", flush=True)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.from_numpy(self.Y[idx].copy()).float()

        # FIX: convert absolute Mag_U → Delta_U to match data spec
        # Spec: Delta_U = (Mag_U - U_ref) / U_ref  (range ~-1.0 to +0.5)
        # Channel 3 = U_over_Uref * 2.0  →  U_ref = ch3 / 2.0
        if self.convert_to_delta:
            u_ref = torch.clamp(x[3:4, :, :] / 2.0, min=0.01)
            y = (y - u_ref) / u_ref
            y = torch.clamp(y, -1.5, 2.0)

        # FIX: reduced mask ceiling from 20x to 5x to prevent boundary loss spikes
        sdf_meters = x[0:1, :, :] * self.sdf_scaling
        mask = 1.0 + 4.0 * torch.exp(-torch.clamp(sdf_meters, min=0.0) / 5.0)
        return x, y, mask

class PINNNPZDataset(torch.utils.data.Dataset):
    """Dataset for a list of .npz files."""
    def __init__(self, files, sdf_scaling=200.0, normalize_target=True):
        self.files = files
        self.sdf_scaling = sdf_scaling
        # FIX: detect target format by sampling the first file
        with np.load(files[0]) as d:
            y_sample = d['Y'].ravel()
            self.convert_to_delta = normalize_target and (float(y_sample.min()) >= -0.05)
        if self.convert_to_delta:
            print("[PINNNPZDataset] Y appears to be absolute Mag_U (min>=0). "
                  "Will convert to Delta_U = (Mag_U - U_ref) / U_ref on-the-fly.", flush=True)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            x = torch.from_numpy(data['X']).float()
            y = torch.from_numpy(data['Y']).float()
            if x.ndim == 4: x = x.squeeze(0)
            if y.ndim == 4: y = y.squeeze(0)

            # FIX: convert absolute Mag_U → Delta_U to match data spec
            if self.convert_to_delta:
                u_ref = torch.clamp(x[3:4, :, :] / 2.0, min=0.01)
                y = (y - u_ref) / u_ref
                y = torch.clamp(y, -1.5, 2.0)

            sdf_meters = x[0:1, :, :] * self.sdf_scaling
            # FIX: reduced mask ceiling from 20x to 5x to prevent boundary loss spikes
            mask = 1.0 + 4.0 * torch.exp(-torch.clamp(sdf_meters, min=0.0) / 5.0)
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

        # FIX: physics loss weights now use a warmup curriculum.
        # Full values are the ceiling reached after WARMUP_EPOCHS.
        # Starting at 0 and ramping prevents the model collapsing to a flat
        # constant field (which has zero gradient loss — a degenerate local min).
        GRAD_W_MAX  = config.get('loss', {}).get('gradient_weight', 0.5)   # was 2.0
        WAKE_W_MAX  = config.get('loss', {}).get('wake_weight', 0.5)        # was 1.0
        PEAK_W_MAX  = config.get('loss', {}).get('peak_weight', 0.3)        # was 0.5
        CONT_W_MAX  = config.get('loss', {}).get('continuity_weight', 0.05) # was 0.1
        MOM_W_MAX   = config.get('loss', {}).get('momentum_weight', 0.02)   # was 0.05
        WARMUP_EPOCHS = config.get('loss', {}).get('warmup_epochs', 50)

        def get_loss_weights(epoch):
            """Linearly ramp physics weights from 0 to their max over WARMUP_EPOCHS."""
            t = min(epoch / max(WARMUP_EPOCHS, 1), 1.0)
            return dict(
                grad_weight=GRAD_W_MAX * t,
                wake_weight=WAKE_W_MAX * t,
                peak_weight=PEAK_W_MAX * t,
                continuity_weight=CONT_W_MAX * t,
                momentum_weight=MOM_W_MAX * t,
            )

        if is_main(rank):
            print("=" * 60, flush=True)
            print("PINN-FNO Training", flush=True)
            print(f"  Modes: {MODES1}x{MODES2}, Width: {WIDTH}, Layers: {N_LAYERS}", flush=True)
            print(f"  Physics warmup: {WARMUP_EPOCHS} epochs", flush=True)
            print(f"  Max weights — grad={GRAD_W_MAX}, wake={WAKE_W_MAX}, "
                  f"cont={CONT_W_MAX}, mom={MOM_W_MAX}, peak={PEAK_W_MAX}", flush=True)
            print("=" * 60, flush=True)

        # --- Dataset ---
        paths_to_check = [
            "/home/hice1/athach7/scratch/Training_Dataset",
            DATA_FOLDER
        ]
        
        found_data = False
        for p in paths_to_check:
            x_npy, y_npy = os.path.join(p, 'X.npy'), os.path.join(p, 'Y.npy')
            if os.path.exists(x_npy) and os.path.exists(y_npy):
                if is_main(rank): print(f"LOADING PRE-PROCESSED: {p}", flush=True)
                temp_x   = np.load(x_npy, mmap_mode='r')
                sdf_max  = float(temp_x[0, 0].max())
                sdf_scale = 200.0 if sdf_max <= 5.0 else 1.0
                dataset  = PINNNumpyDataset(x_npy, y_npy, sdf_scale)
                found_data = True
                break
        
        if not found_data:
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
            # Diagnostic: confirm target format so you can catch the mismatch early
            _sx, _sy, _ = dataset[0]
            print(f"  Y sample  — min: {_sy.min():.3f}  max: {_sy.max():.3f}  mean: {_sy.mean():.3f}",
                  flush=True)
            if _sy.min() >= -0.05 and _sy.mean() > 0.3:
                print("  WARNING: Y still looks like absolute Mag_U (mean > 0.3, no negatives).",
                      flush=True)
                print("  Check that needs_delta_transform fired correctly in the dataset class.",
                      flush=True)
            else:
                print("  Y looks like Delta_U (has negatives / mean near 0). Good.", flush=True)

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

                # FIX: use epoch-dependent warmup weights
                w = get_loss_weights(epoch)
                loss, comps = pinn_loss(
                    pred, yb, x_input=xb, sensor_mask=mb,
                    grad_weight=w['grad_weight'],
                    continuity_weight=w['continuity_weight'],
                    momentum_weight=w['momentum_weight'],
                    wake_weight=w['wake_weight'],
                    peak_weight=w['peak_weight'],
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
                warmup_pct = min(epoch / max(WARMUP_EPOCHS, 1), 1.0) * 100
                print(
                    f"Epoch {epoch:4d}/{EPOCHS} | Loss: {avg_loss:.4e} "
                    f"| Cont: {cont_avg:.4e} | Wake: {wake_avg:.4e} "
                    f"| Warmup: {warmup_pct:.0f}% | ({dur:.1f}s)",
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