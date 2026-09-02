"""
Distributed Training for Physics-Informed FNO (PINN-FNO)
=======================================================
Features:
- Standardized 90/10 Train/Val Split
- Physics-Informed Loss (Continuity + Momentum)
- DDP Scaling
"""
import os, sys, torch, numpy as np, argparse, traceback, time
from multiprocessing import cpu_count
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Local imports
from core.models.pinn_fno import PINNFNO, pinn_loss
from core.models.fno2d import sensor_weighted_mse
from core.utils.training_logger import TrainingLogger
from core.utils.config_loader import load_config
from pipelines.train.distributed import NpyDataset, build_channel_mask, loader_kwargs

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank, world_size, local_rank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size, True
    return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized(): dist.destroy_process_group()

def is_main(rank): return rank == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.toml')
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    local_rank, rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    try:
        # 1. Hyperparams
        TC = config.get('training', {})
        LC = config.get('loss', {})
        MC = config.get('model', {})
        
        MODES1, MODES2 = MC.get('modes1', 64), MC.get('modes2', 64)
        WIDTH, N_LAYERS = MC.get('width', 32), MC.get('n_layers', 4)
        BATCH, EPOCHS, LR = TC.get('batch_size', 16), TC.get('epochs', 100), TC.get('learning_rate', 1e-4)
        MODEL_OUT = config.get('paths', {}).get('model_checkpoint', 'pinn_fno_weights.pth')
        _nw = config.get('performance', {}).get('num_workers', 0)
        if _nw > 0:
            NUM_WORKERS = _nw
        elif sys.platform == 'win32':
            NUM_WORKERS = min(8, max(1, cpu_count() // 2))
        else:
            NUM_WORKERS = min(8, max(1, cpu_count() // 2))
        
        GRAD_W = LC.get('gradient_weight', 1.5)
        CONT_W = LC.get('continuity_weight', 0.05)
        MOM_W  = LC.get('momentum_weight', 0.01)
        PEAK_W = LC.get('peak_weight', 0.5)
        WAKE_W = LC.get('wake_weight', 1.0)
        CHANNEL_WEIGHTS = LC.get('channel_weights', None)
        PATIENCE = TC.get('patience', 50)
        CHECKPOINT_INTERVAL = TC.get('checkpoint_interval', 10)

        # 2. Data
        DATA_FOLDER = config.get('paths', {}).get('data_folder_ice', 'train_csv')
        if not os.path.exists(DATA_FOLDER): DATA_FOLDER = config.get('paths', {}).get('data_folder_linux', DATA_FOLDER)
        
        x_p, y_p = os.path.join(DATA_FOLDER, 'X.npy'), os.path.join(DATA_FOLDER, 'Y.npy')

        if args.val_dir and os.path.exists(os.path.join(args.val_dir, 'X.npy')):
            train_dataset = NpyDataset(x_p, y_p, augment=True)
            val_dataset   = NpyDataset(os.path.join(args.val_dir, 'X.npy'), os.path.join(args.val_dir, 'Y.npy'), augment=False)
        else:
            full_dataset = NpyDataset(x_p, y_p, augment=True)
            val_dataset_f = NpyDataset(x_p, y_p, augment=False)
            VAL_S = TC.get('val_split', 0.1)
            train_sz = int((1.0 - VAL_S) * len(full_dataset))
            idx = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(42)).tolist()
            train_dataset = Subset(full_dataset, idx[:train_sz])
            val_dataset   = Subset(val_dataset_f, idx[train_sz:])

        t_samp = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_distributed else None
        v_samp = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
        _lk = loader_kwargs(NUM_WORKERS)
        loader = DataLoader(train_dataset, batch_size=BATCH, sampler=t_samp, shuffle=(t_samp is None), **_lk)
        v_loader = DataLoader(val_dataset, batch_size=BATCH, sampler=v_samp, shuffle=False, **_lk)

        # 3. Model
        sample_x, sample_y = train_dataset[0]
        out_ch = sample_y.shape[0]
        model = PINNFNO(in_channels=sample_x.shape[0], out_channels=out_ch, n_modes=(MODES1, MODES2), hidden_channels=WIDTH, n_layers=N_LAYERS).to(device)
        
        if os.path.exists(MODEL_OUT) and not args.fresh:
            sd = torch.load(MODEL_OUT, map_location=device, weights_only=False)
            model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd, strict=False)

        if is_distributed: model = DDP(model, device_ids=[local_rank])
        
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        # PINN has no epoch-ramped loss weights, but we still key the LR
        # schedule / early-stop / checkpointing to the same stable pure
        # per-channel MSE used by the other pipelines so the four are
        # directly comparable.
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=8, min_lr=1e-6)

        if is_main(rank):
            os.makedirs("epochs", exist_ok=True)
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger = TrainingLogger(output_dir='training_logs', experiment_name=f"PINN_{_ts}")
            logger.start_training({'batch': BATCH, 'lr': LR, 'epochs': EPOCHS, 'modes1': MODES1,
                                   'modes2': MODES2, 'width': WIDTH, 'n_layers': N_LAYERS,
                                   'patience': PATIENCE},
                                  model=model.module if is_distributed else model)

        best_v = float('inf')
        patience_counter = 0
        t_hist, v_hist = [], []

        for ep in range(1, EPOCHS + 1):
            epoch_start = time.time()
            if is_distributed: t_samp.set_epoch(ep)
            model.train()
            r_l = r_mse = r_grad = r_wake = r_peak = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                mb = build_channel_mask(xb, out_ch)
                pred = model(xb)
                loss, comps = pinn_loss(pred, yb, x_input=xb, sensor_mask=mb, grad_weight=GRAD_W, continuity_weight=CONT_W, momentum_weight=MOM_W, wake_weight=WAKE_W, peak_weight=PEAK_W, channel_weights=CHANNEL_WEIGHTS)
                opt.zero_grad(); loss.backward(); opt.step()
                bs = xb.shape[0]
                r_l += loss.item() * bs
                r_mse += comps['mse_loss'] * bs
                r_grad += comps['gradient_loss'] * bs
                r_wake += comps.get('wake_loss', 0.0) * bs
                r_peak += comps.get('peak_loss', 0.0) * bs

            model.eval()
            v_l_acc = 0.0     # composite PINN val loss, for the curve
            v_sel_acc = 0.0   # stable selection metric: pure per-channel MSE
            with torch.no_grad():
                for xb, yb in v_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    mb = build_channel_mask(xb, out_ch)
                    pred = model(xb)
                    v_l, _ = pinn_loss(pred, yb, x_input=xb, sensor_mask=mb, grad_weight=GRAD_W, continuity_weight=CONT_W, momentum_weight=MOM_W, wake_weight=WAKE_W, peak_weight=PEAK_W, channel_weights=CHANNEL_WEIGHTS)
                    v_sel = sensor_weighted_mse(pred, yb, sensor_mask=mb, channel_weights=CHANNEL_WEIGHTS)
                    v_l_acc += v_l.item() * xb.shape[0]
                    v_sel_acc += v_sel.item() * xb.shape[0]

            if is_distributed:
                rt = torch.tensor([r_l, r_mse, r_grad, r_wake, r_peak, v_l_acc, v_sel_acc], device=device)
                dist.all_reduce(rt, op=dist.ReduceOp.SUM)
                r_l, r_mse, r_grad, r_wake, r_peak, v_l_acc, v_sel_acc = rt.tolist()

            n_tr, n_va = len(train_dataset), len(val_dataset)
            avg_t, avg_v, avg_sel = r_l/n_tr, v_l_acc/n_va, v_sel_acc/n_va
            avg_mse, avg_grad = r_mse/n_tr, r_grad/n_tr
            avg_wake, avg_peak = r_wake/n_tr, r_peak/n_tr

            sched.step(avg_sel)
            improved = avg_sel < best_v
            if improved:
                best_v = avg_sel
                patience_counter = 0
            else:
                patience_counter += 1
            epoch_dt = time.time() - epoch_start

            if is_main(rank):
                print(f"Ep {ep}/{EPOCHS} | T: {avg_t:.4e} | V: {avg_v:.4e} | ValSel: {avg_sel:.4e} "
                      f"| LR: {opt.param_groups[0]['lr']:.2e} | patience {patience_counter}/{PATIENCE} ({epoch_dt:.1f}s)")
                t_hist.append(avg_t); v_hist.append(avg_v)
                sd = model.module.state_dict() if is_distributed else model.state_dict()
                if improved:
                    torch.save({'model_state_dict': sd, 'history': {'train': t_hist, 'val': v_hist}, 'epoch': ep}, MODEL_OUT)
                    print(f"   * Saved best (ValSel: {best_v:.4e}, epoch {ep})")
                if ep % CHECKPOINT_INTERVAL == 0:
                    torch.save(sd, f"epochs/pinn_ep{ep}.pth")
                logger.log_epoch(ep, {
                    'total_loss': avg_t, 'val_loss': avg_v, 'val_sel_loss': avg_sel,
                    'mse_loss': avg_mse, 'gradient_loss': avg_grad,
                    'peak_loss': avg_peak, 'wake_loss': avg_wake,
                    'learning_rate': opt.param_groups[0]['lr'], 'epoch_time_sec': epoch_dt,
                    'best_loss': best_v, 'patience_counter': patience_counter,
                })

            if patience_counter >= PATIENCE:
                if is_main(rank): print(f"Early stopping at epoch {ep}")
                break

    except Exception as e:
        print(f"Error {rank}: {e}"); traceback.print_exc()
        if is_distributed: cleanup_distributed()
        sys.exit(1)

    if is_main(rank):
        print(f"Done. Best: {best_v:.6e}")
        if 'logger' in locals(): logger.finish_training({'best_loss': best_v})
    cleanup_distributed()

if __name__ == "__main__": main()
