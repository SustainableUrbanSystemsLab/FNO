"""
Distributed Training for Geometry-Aware FNO (GeoFNO) Model
=========================================
Features:
- PyTorch Distributed Data Parallel (DDP)
- Standardized Train/Val Split (90/10)
- Manual Validation Directory Override
- Physics-Informed Metrics Logging
- Automated Model Checkpointing
"""
import os, sys, torch, numpy as np, pandas as pd, argparse, traceback, time
from multiprocessing import cpu_count
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Local imports
from core.models.fno2d import sensor_weighted_mse
from core.models.geo_fno import GeoFNO
from core.utils.training_logger import TrainingLogger
from core.utils.config_loader import load_config
from pipelines.train.distributed import NpyDataset, build_channel_mask

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

def is_main(rank):
    return rank == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.toml')
    parser.add_argument('--val_dir', type=str, default=None, help='Directory containing validation X.npy/Y.npy')
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    local_rank, rank, world_size, is_distributed = setup_distributed()
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        if is_main(rank):
            print(f"Device: {torch.cuda.get_device_name(device)} (ID: {local_rank})")
            print(f"Free Memory: {torch.cuda.mem_get_info(device)[0]/1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        if is_main(rank): print("Warning: No CUDA detected, using CPU.")

    try:
        # 1. Path detection
        if sys.platform == 'win32':
            DATA_FOLDER = config.get('paths', {}).get('data_folder_windows', 'train_csv')
        else:
            linux_path = config.get('paths', {}).get('data_folder_linux', '')
            ice_path = config.get('paths', {}).get('data_folder_ice', '')
            
            if ice_path: ice_path = os.path.expanduser(ice_path)
            if linux_path: linux_path = os.path.expanduser(linux_path)

            if ice_path and os.path.exists(os.path.join(ice_path, "X.npy")):
                DATA_FOLDER = ice_path
            elif linux_path and os.path.exists(os.path.join(linux_path, "X.npy")):
                DATA_FOLDER = linux_path
            else:
                DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'train_csv')

        # 2. Hyperparameters
        MODES1  = config.get('model', {}).get('modes1', 64)
        MODES2  = config.get('model', {}).get('modes2', 64)
        WIDTH   = config.get('model', {}).get('width', 32)
        BATCH   = config.get('training', {}).get('batch_size', 16)
        EPOCHS  = config.get('training', {}).get('epochs', 100)
        LR      = config.get('training', {}).get('learning_rate', 1e-4)
        MODEL_OUT = config.get('paths', {}).get('model_checkpoint', 'geo_fno_weights.pth')
        _nw = config.get('performance', {}).get('num_workers', 0)
        if _nw > 0:
            NUM_WORKERS = _nw
        elif sys.platform == 'win32':
            NUM_WORKERS = min(8, max(1, cpu_count() // 2))
        else:
            NUM_WORKERS = max(1, cpu_count() // 2)
        CHECKPOINT_INTERVAL = config.get('training', {}).get('checkpoint_interval', 10)

        # Physics Weights (balanced with MSE to prevent zero-velocity field collapse)
        PEAK_W   = config.get('loss', {}).get('peak_weight', 0.5)
        WAKE_W   = config.get('loss', {}).get('wake_weight', 1.0)
        GRAD_W   = config.get('loss', {}).get('gradient_weight', 1.5)
        SPEC_W   = config.get('loss', {}).get('spectral_weight', 0.001)

        def get_loss_weights(epoch):
            t = min(1.0, epoch / 30.0) 
            return {
                'grad_weight': GRAD_W * t,
                'spectral_weight': SPEC_W * t,
                'peak_weight': PEAK_W * t,
                'wake_weight': WAKE_W * t,
            }

        if is_main(rank):
            os.makedirs("epochs", exist_ok=True)
            os.makedirs("training_logs", exist_ok=True)

        # 3. Data Loading
        x_path, y_path = os.path.join(DATA_FOLDER, 'X.npy'), os.path.join(DATA_FOLDER, 'Y.npy')
        
        if args.val_dir and os.path.exists(os.path.join(args.val_dir, 'X.npy')):
            train_dataset = NpyDataset(x_path, y_path, augment=True)
            val_dataset = NpyDataset(os.path.join(args.val_dir, 'X.npy'), os.path.join(args.val_dir, 'Y.npy'), augment=False)
            if is_main(rank): print(f"Using manual val_dir: {args.val_dir}")
        else:
            full_dataset = NpyDataset(x_path, y_path, augment=True)
            val_dataset_full = NpyDataset(x_path, y_path, augment=False)
            VAL_SPLIT = config.get('training', {}).get('val_split', 0.1)
            train_size = int((1.0 - VAL_SPLIT) * len(full_dataset))
            indices = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(42)).tolist()
            train_dataset = Subset(full_dataset, indices[:train_size])
            val_dataset = Subset(val_dataset_full, indices[train_size:])
            if is_main(rank): print(f"Using {((1.0-VAL_SPLIT)*100):.0f}/{VAL_SPLIT*100:.0f} native split.")

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_distributed else None
        val_sampler   = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
        loader = DataLoader(train_dataset, batch_size=BATCH, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH, sampler=val_sampler, shuffle=False, num_workers=NUM_WORKERS)

        # 4. Model
        sample_x, sample_y = train_dataset[0]
        out_ch = sample_y.shape[0]
        model = GeoFNO(in_channels=sample_x.shape[0], out_channels=out_ch, n_modes=(MODES1, MODES2), hidden_channels=WIDTH).to(device)

        if os.path.exists(MODEL_OUT) and not args.fresh:
            if is_main(rank): print(f"Loading weights from {MODEL_OUT}")
            saved = torch.load(MODEL_OUT, map_location=device, weights_only=False)
            sd = saved['model_state_dict'] if isinstance(saved, dict) and 'model_state_dict' in saved else saved
            model.load_state_dict(sd, strict=True)

        if is_distributed: model = DDP(model, device_ids=[local_rank])
        
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

        if is_main(rank):
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger = TrainingLogger(output_dir="training_logs", experiment_name=f"GEO_{_ts}")
            logger.start_training({'batch': BATCH, 'lr': LR, 'modes': (MODES1, MODES2)}, model=model.module if is_distributed else model)

        best_loss = float('inf')
        train_hist, val_hist = [], []

        for epoch in range(1, EPOCHS + 1):
            if is_distributed: train_sampler.set_epoch(epoch)
            model.train()
            r_loss, r_mse, r_grad, r_spec, r_peak, r_wake = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                mb = build_channel_mask(xb, out_ch)
                pred = model(xb)
                w = get_loss_weights(epoch)
                loss, comps = sensor_weighted_mse(
                    pred, yb, sensor_mask=mb,
                    grad_weight=w['grad_weight'], spectral_weight=w['spectral_weight'],
                    peak_weight=w['peak_weight'], wake_weight=w['wake_weight'],
                    wake_threshold=-0.5, return_components=True
                )
                opt.zero_grad(); loss.backward(); opt.step()
                bs = xb.shape[0]
                r_loss += loss.item() * bs
                r_mse += comps['mse_loss'] * bs
                r_grad += comps['gradient_loss'] * bs
                r_spec += comps['spectral_loss'] * bs
                r_peak += comps.get('peak_loss', 0.0) * bs
                r_wake += comps.get('wake_loss', 0.0) * bs

            # Evaluation
            model.eval()
            v_loss_acc = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    mb = build_channel_mask(xb, out_ch)
                    v_l = sensor_weighted_mse(model(xb), yb, sensor_mask=mb, **get_loss_weights(epoch), wake_threshold=-0.5)
                    v_loss_acc += v_l.item() * xb.shape[0]

            if is_distributed:
                rt = torch.tensor([r_loss, r_mse, r_grad, r_spec, r_peak, r_wake, v_loss_acc], device=device)
                dist.all_reduce(rt, op=dist.ReduceOp.SUM)
                r_loss, r_mse, r_grad, r_spec, r_peak, r_wake, v_loss_acc = rt.tolist()

            scheduler.step()
            avg_train, avg_val = r_loss/len(train_dataset), v_loss_acc/len(val_dataset)
            
            if is_main(rank):
                print(f"Epoch {epoch}/{EPOCHS} | Train: {avg_train:.4e} | Val: {avg_val:.4e}")
                logger.log_epoch(epoch, {'total_loss': avg_train, 'val_loss': avg_val, 'best_loss': best_loss})
                train_hist.append(avg_train); val_hist.append(avg_val)

                if avg_val < best_loss:
                    best_loss = avg_val
                    sd = model.module.state_dict() if is_distributed else model.state_dict()
                    torch.save({'model_state_dict': sd, 'history': {'train': train_hist, 'val': val_hist}, 'epoch': epoch}, MODEL_OUT)
                    print(f"   * Best model saved (Val: {best_loss:.4e})")

                if epoch % CHECKPOINT_INTERVAL == 0:
                    torch.save(sd, f"epochs/geo_ep{epoch}.pth")

    except Exception as e:
        print(f"ERROR [Rank {rank}]: {e}"); traceback.print_exc()
        if is_distributed: cleanup_distributed()
        sys.exit(1)

    if is_main(rank):
        print(f"Finished. Best Val: {best_loss:.6e}")
        if 'logger' in locals(): logger.finish_training({'best_loss': best_loss})
    cleanup_distributed()

if __name__ == "__main__":
    main()
