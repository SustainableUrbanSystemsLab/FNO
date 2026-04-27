"""
Distributed Training for Hybrid FNO Model
=========================================
Features:
- Standardized 90/10 Train/Val Split
- DDP Scaling
- Integrated Validation Loop
"""
import os, sys, torch, numpy as np, argparse, traceback, time
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Local imports
from core.models.hybrid import HybridFNO
from core.models.fno2d import sensor_weighted_mse
from core.utils.training_logger import TrainingLogger
from core.utils.config_loader import load_config
from pipelines.train.distributed import NpyDataset

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
        MODES1, MODES2 = config.get('model', {}).get('modes1', 64), config.get('model', {}).get('modes2', 64)
        WIDTH = config.get('model', {}).get('width', 32)
        BATCH, EPOCHS, LR = TC.get('batch_size', 16), TC.get('epochs', 100), TC.get('learning_rate', 1e-4)
        MODEL_OUT = config.get('paths', {}).get('model_checkpoint', 'hybrid_weights.pth')
        
        # 2. Data
        DATA_FOLDER = config.get('paths',{}).get('data_folder_ice', 'train_csv')
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
        loader = DataLoader(train_dataset, batch_size=BATCH, sampler=t_samp, shuffle=(t_samp is None), num_workers=2)
        v_loader = DataLoader(val_dataset, batch_size=BATCH, sampler=v_samp, shuffle=False, num_workers=2)

        # 3. Model
        sample_x, sample_y = train_dataset[0]
        out_ch = sample_y.shape[0]
        model = HybridFNO(in_channels=sample_x.shape[0], out_channels=out_ch, n_modes=(MODES1, MODES2), hidden_channels=WIDTH).to(device)
        
        if os.path.exists(MODEL_OUT) and not args.fresh:
            saved = torch.load(MODEL_OUT, map_location=device, weights_only=False)
            model.load_state_dict(saved['model_state_dict'] if 'model_state_dict' in saved else saved, strict=False)

        if is_distributed: model = DDP(model, device_ids=[local_rank])
        
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

        if is_main(rank):
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger = TrainingLogger(output_dir='training_logs', experiment_name=f"HYBRID_{_ts}")

        best_v = float('inf')
        t_hist, v_hist = [], []

        for ep in range(1, EPOCHS + 1):
            if is_distributed: t_samp.set_epoch(ep)
            model.train()
            r_l = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                sdf = xb[:, 0:1, :, :]
                mb = torch.where(sdf > 0, torch.ones_like(sdf), torch.full_like(sdf, 0.2))
                pred = model(xb)
                loss = sensor_weighted_mse(pred, yb, sensor_mask=mb) # Standard hybrid loss
                opt.zero_grad(); loss.backward(); opt.step()
                r_l += loss.item() * xb.shape[0]

            model.eval()
            v_l_acc = 0.0
            with torch.no_grad():
                for xb, yb in v_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    v_l = sensor_weighted_mse(model(xb), yb, sensor_mask=torch.where(xb[:,0:1]>0, 1.0, 0.2))
                    v_l_acc += v_l.item() * xb.shape[0]

            if is_distributed:
                rt = torch.tensor([r_l, v_l_acc], device=device)
                dist.all_reduce(rt, op=dist.ReduceOp.SUM)
                r_l, v_l_acc = rt.tolist()

            sched.step()
            avg_t, avg_v = r_l/len(train_dataset), v_l_acc/len(val_dataset)

            if is_main(rank):
                print(f"Ep {ep}/{EPOCHS} | T: {avg_t:.4e} | V: {avg_v:.4e}")
                logger.log_epoch(ep, {'total_loss': avg_t, 'val_loss': avg_v, 'best_loss': best_v})
                t_hist.append(avg_t); v_hist.append(avg_v)
                if avg_v < best_v:
                    best_v = avg_v
                    sd = model.module.state_dict() if is_distributed else model.state_dict()
                    torch.save({'model_state_dict': sd, 'history': {'train': t_hist, 'val': v_hist}, 'epoch': ep}, MODEL_OUT)
                    print(f"   * Saved best hybrid model (Val: {best_v:.4e})")

    except Exception as e:
        print(f"Error {rank}: {e}"); traceback.print_exc()
        if is_distributed: cleanup_distributed()
        sys.exit(1)

    if is_main(rank):
        print(f"Done. Best: {best_v:.6e}")
        if 'logger' in locals(): logger.finish_training({'best_loss': best_v})
    cleanup_distributed()

if __name__ == "__main__": main()
