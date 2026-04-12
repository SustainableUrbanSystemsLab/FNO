"""
PINN-FNO Training Pipeline
===========================
Trains the Physics-Informed FNO on the wind field dataset.

Usage:
    bash slurm/deploy_ice.sh --script pipelines/train/pinn.py --gpu h100 --ngpus 2 --fresh
"""

import os, sys, torch, numpy as np, pandas as pd, tomllib, argparse, glob, hashlib, pickle, traceback, time
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.models.pinn_fno import PINNFNO, pinn_loss
from core.utils.training_logger import TrainingLogger
from pipelines.train.distributed import NpyDataset


# ============ Config ============
from core.utils.config_loader import load_config


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


# ============ Main ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.toml')
    parser.add_argument('--val_dir', type=str, default=None, help='Directory containing validation X.npy/Y.npy')
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
        x_path = os.path.join(DATA_FOLDER, 'X.npy')
        y_path = os.path.join(DATA_FOLDER, 'Y.npy')

        if is_main(rank):
            print(f"Loading dataset from {DATA_FOLDER}...")
            print(f"  X path: {x_path}")
            print(f"  Y path: {y_path}")

        if args.val_dir and os.path.exists(os.path.join(args.val_dir, 'X.npy')):
            train_dataset = NpyDataset(x_path, y_path, augment=True)
            val_dataset = NpyDataset(os.path.join(args.val_dir, 'X.npy'), os.path.join(args.val_dir, 'Y.npy'), augment=False)
            if is_main(rank): print(f"Using explicitly specified val_dir: {args.val_dir}", flush=True)
        else:
            full_dataset = NpyDataset(x_path, y_path, augment=True)
            val_dataset_full = NpyDataset(x_path, y_path, augment=False)
            VAL_SPLIT = config.get('training', {}).get('val_split', 0.1)
            train_size = int((1.0 - VAL_SPLIT) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            indices = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(42)).tolist()
            from torch.utils.data import Subset
            train_dataset = Subset(full_dataset, indices[:train_size])
            val_dataset = Subset(val_dataset_full, indices[train_size:])
            if is_main(rank): print(f"Using {((1.0 - VAL_SPLIT)*100):.0f}/{VAL_SPLIT*100:.0f} random train/val split natively. Train: {train_size}, Val: {val_size}", flush=True)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
        loader = DataLoader(train_dataset, batch_size=BATCH, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH, sampler=val_sampler, shuffle=False, num_workers=2)

        # --- Model ---
        sample_x, _ = train_dataset[0]
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
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger = TrainingLogger(output_dir='training_logs', experiment_name=f"PINN_{_ts}")
            logger.start_training({
                'batch_size': BATCH,
                'epochs': EPOCHS,
                'learning_rate': LR,
                'modes': (MODES1, MODES2),
                'width': WIDTH,
                'physics_weights': {'continuity': CONT_W, 'momentum': MOM_W}
            }, model=model.module if is_distributed else model)

        # --- Training Loop ---
        best_loss      = float('inf')
        patience_count = 0
        train_losses   = []
        val_losses     = []

        for epoch in range(1, EPOCHS + 1):
            if is_distributed: train_sampler.set_epoch(epoch)
            model.train()
            running = 0.0
            comp_accum = {k: 0.0 for k in ['mse_loss','gradient_loss','continuity_loss',
                                            'momentum_loss','wake_loss','peak_loss']}
            epoch_start = time.time()

            for batch in loader:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)

                # Build mask from SDF channel
                sdf = xb[:, 0:1, :, :]
                mb = torch.where(sdf > 0, torch.ones_like(sdf), torch.full_like(sdf, 0.2))

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

<<<<<<< HEAD
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
                    v_loss, _ = pinn_loss(
                        pred, yb, x_input=xb, sensor_mask=mb,
                        grad_weight=GRAD_W,
                        continuity_weight=CONT_W,
                        momentum_weight=MOM_W,
                        wake_weight=WAKE_W,
                        peak_weight=PEAK_W
                    )
                    val_running += v_loss.item() * xb.shape[0]

            # Sync across GPUs
=======
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e
            if is_distributed:
                torch.cuda.synchronize()
                dist.barrier()

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
                    v_loss, _ = pinn_loss(
                        pred, yb, x_input=xb, sensor_mask=mb,
                        grad_weight=GRAD_W, continuity_weight=CONT_W,
                        momentum_weight=MOM_W, wake_weight=WAKE_W, peak_weight=PEAK_W
                    )
                    val_running += float(v_loss.item()) * xb.shape[0]

            if is_distributed:
                keys = ['mse_loss','gradient_loss','continuity_loss','momentum_loss','wake_loss','peak_loss']
                vals = [running] + [comp_accum[k] for k in keys] + [val_running]
                t = torch.tensor(vals, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                running = t[0].item()
                for i, k in enumerate(keys):
                    comp_accum[k] = t[i+1].item()
                val_running = t[-1].item()

            scheduler.step()
<<<<<<< HEAD
            n_train = len(train_dataset)
            n_val = len(val_dataset)
            avg_loss = running / n_train
            avg_val_loss = val_running / n_val
=======
            n_samples_train = len(train_dataset)
            n_samples_val   = len(val_dataset)
            avg_loss = running / n_samples_train
            avg_val_loss = val_running / n_samples_val
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e

            if is_main(rank):
                dur = time.time() - epoch_start
                # Calculate averages
<<<<<<< HEAD
                avgs = {k: comp_accum[k] / n_train for k in comp_accum}
=======
                avgs = {k: comp_accum[k] / n_samples_train for k in comp_accum}
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e
                cont_avg = avgs['continuity_loss']
                wake_avg = avgs['wake_loss']

                # Log to metrics CSV
                logger.log_epoch(epoch, {
                    'total_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'mse_loss': avgs['mse_loss'],
                    'gradient_loss': avgs['gradient_loss'],
                    'continuity_loss': cont_avg,
                    'momentum_loss': avgs['momentum_loss'],
                    'wake_loss': wake_avg,
                    'peak_loss': avgs['peak_loss'],
<<<<<<< HEAD
                    'spectral_loss': 0.0,
=======
                    'spectral_loss': 0.0,
>>>>>>> fd0fe5c05207208c33450bd11d7b559ad210fa8e
                    'learning_rate': scheduler.get_last_lr()[0],
                    'epoch_time_sec': dur,
                    'best_loss': best_loss
                })
                train_losses.append(avg_loss)
                val_losses.append(avg_val_loss)

                print(
                    f"Epoch {epoch:4d}/{EPOCHS} | Loss: {avg_loss:.4e} | Val Loss: {avg_val_loss:.4e} "
                    f"| Cont: {cont_avg:.4e} | Wake: {wake_avg:.4e} | ({dur:.1f}s)",
                    flush=True
                )

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_count = 0

                    # Payload including training history for tools/plot_comparison_curves.py
                    sd = model.module.state_dict() if is_distributed else model.state_dict()
                    payload = {
                        'model_state_dict': sd,
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
                    print(f"  ★ Best model saved (loss={best_loss:.4e})", flush=True)
                else:
                    patience_count += 1
                    if patience_count >= PATIENCE:
                        print(f"Early stopping at epoch {epoch}", flush=True)
                        break

                if epoch % CKPT_INT == 0:
                    sd = model.module.state_dict() if is_distributed else model.state_dict()
                    ckpt = os.path.join(EPOCHS_DIR, 'pinn_rolling_checkpoint.pth')
                    temp_ckpt = ckpt + ".tmp"
                    torch.save(sd, temp_ckpt)
                    if os.path.exists(ckpt): os.remove(ckpt)
                    os.rename(temp_ckpt, ckpt)
                    print(f"  > Checkpoint: {ckpt}", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR rank {rank}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        if is_distributed: cleanup_distributed()
        sys.exit(1)

    if is_main(rank):
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


if __name__ == '__main__':
    main()
