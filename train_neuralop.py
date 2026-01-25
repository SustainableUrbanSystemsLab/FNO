#!/usr/bin/env python3
"""
FNO Training Script using NeuralOperator Library
=================================================
Uses the official NeuralOperator library for improved architecture:
- Positional embeddings
- Domain padding
- H1Loss (Sobolev norm)
- Better skip connections
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import hashlib

# NeuralOperator imports
from neuralop.models import FNO
from neuralop.losses import LpLoss, H1Loss

# Local imports
from gh_to_fno import build_input_tensor_from_gh, infer_grid_from_coords_simple
from training_logger import TrainingLogger

# ============ Load Configuration ============
CONFIG_FILE = "config.toml"

def load_config():
    """Load configuration from config.toml file."""
    import tomllib
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            return tomllib.load(f)
    return {}

config = load_config()

# Paths
DATA_FOLDER_WIN = config.get('paths', {}).get('data_folder_windows', 'C:/LabShare/Dataset/FormFluxCases/Compressed/Training_Dataset')
DATA_FOLDER_LINUX = config.get('paths', {}).get('data_folder_linux', '/storage/coda1/p-pkastner3/0/ikaradag3/FNO/Training_Dataset')
MODEL_OUT = config.get('paths', {}).get('model_output', 'fno_mag_weights.pth')

# Training params
BATCH = config.get('training', {}).get('batch_size', 4)
EPOCHS = config.get('training', {}).get('epochs', 500)
LR = config.get('training', {}).get('learning_rate', 0.0005)
PATIENCE = config.get('training', {}).get('patience', 75)

# Model params
MODES = config.get('model', {}).get('modes1', 48)
WIDTH = config.get('model', {}).get('width', 64)
N_LAYERS = config.get('model', {}).get('n_layers', 5)

# Loss weights
GRAD_WEIGHT = config.get('loss', {}).get('gradient_weight', 0.3)

# ============ Distributed Setup ============
def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False

def is_main_process(rank):
    return rank == 0

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

# ============ Data Loading ============
def get_data_folder():
    if os.path.exists(DATA_FOLDER_WIN):
        return DATA_FOLDER_WIN
    elif os.path.exists(DATA_FOLDER_LINUX):
        return DATA_FOLDER_LINUX
    else:
        raise RuntimeError(f"Data folder not found: {DATA_FOLDER_WIN} or {DATA_FOLDER_LINUX}")

def load_single_csv(fp):
    """Load and process a single CSV file."""
    try:
        df = pd.read_csv(fp)
        
        # Column renaming for compatibility
        rename_map = {
            'X': 'X_coords', 'Y': 'Y_coords',
            'x': 'X_coords', 'y': 'Y_coords',
            'U_at_z': 'U_over_Uref',
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Required columns
        cols = ['SDF', 'Bldg_height', 'Z_relative', 'U_over_Uref', 'X_coords', 'Y_coords', 'dir_sin', 'dir_cos']
        for c in cols:
            if c not in df.columns:
                return None, f"Missing column {c} in {fp}"
        
        # Grid inference
        nx, ny, _, _, idx_map = infer_grid_from_coords_simple(df['X_coords'], df['Y_coords'])
        gh_out = {c: df[c].tolist() for c in cols}
        X_tensor, chs = build_input_tensor_from_gh(gh_out, H=ny, W=nx, device='cpu')
        
        # Find magnitude target column
        mag_cols = ['mag_U_dimensionless', 'mag_U', 'mag_dimensionless']
        mag_vals = None
        for c in mag_cols:
            if c in df.columns:
                mag_vals = df[c].to_numpy().astype(float)
                break
        
        if mag_vals is None:
            # Try to compute from velocity components
            if all(cc in df.columns for cc in ['Ux_dimensionless', 'Uy_dimensionless', 'Uz_dimensionless']):
                uxs = df['Ux_dimensionless'].to_numpy().astype(float)
                uys = df['Uy_dimensionless'].to_numpy().astype(float)
                uzs = df['Uz_dimensionless'].to_numpy().astype(float)
                mag_vals = np.sqrt(uxs**2 + uys**2 + uzs**2)
        
        if mag_vals is None:
            return None, f"No mag target found in {fp}"
        
        # Create target grid: delta_u = (mag - U_over_Uref) / U_over_Uref
        Y_grid = np.zeros((1, ny, nx), dtype=np.float32)
        mask_grid = np.zeros((1, ny, nx), dtype=np.float32)
        
        for i, (iy, ix) in enumerate(idx_map):
            val = mag_vals[i]
            u_over_uref = float(df['U_over_Uref'].iloc[i])
            
            if not np.isfinite(val):
                val = 0.0
                valid = 0.2
            else:
                valid = 1.0
            
            delta_u = (val - u_over_uref) / (u_over_uref + 1e-6)
            delta_u = np.clip(delta_u, -1.0, 0.5)
            Y_grid[0, iy, ix] = float(delta_u)
            
            # Weighting: higher near buildings
            sdf_val = max(float(df['SDF'].iloc[i]), 0.0)
            sdf_w = 1.0 + 19.0 * np.exp(-sdf_val / 5.0)
            mask_grid[0, iy, ix] = valid * sdf_w
        
        Y_grid = np.nan_to_num(Y_grid, nan=0.0)
        return (X_tensor.squeeze(0), torch.from_numpy(Y_grid), torch.from_numpy(mask_grid), chs), None
    
    except Exception as e:
        return None, f"Error processing {fp}: {e}"


class CSVDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X, Y, mask, _ = self.data[idx]
        return X, Y, mask


# ============ Main Training ============
def main():
    rank, world_size, local_rank, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process(rank):
        print("=" * 50)
        print("FNO Training with NeuralOperator Library")
        print("=" * 50)
        print(f"Device: {device}, Distributed: {is_distributed}, World size: {world_size}")
    
    # Load data
    data_folder = get_data_folder()
    csv_files = glob.glob(os.path.join(data_folder, '**/*.csv'), recursive=True)
    
    if is_main_process(rank):
        print(f"Found {len(csv_files)} CSV files in {data_folder}")
    
    # Process CSV files
    data_list = []
    errors = []
    
    for fp in tqdm(csv_files, desc="Loading CSVs", disable=not is_main_process(rank)):
        result, error = load_single_csv(fp)
        if result is not None:
            data_list.append(result)
        elif error:
            errors.append(error)
    
    if is_main_process(rank):
        print(f"Loaded {len(data_list)} samples, {len(errors)} errors")
        if errors[:3]:
            for e in errors[:3]:
                print(f"  Error: {e}")
    
    if len(data_list) == 0:
        print("No data loaded! Exiting.")
        return
    
    # Create dataset and dataloader
    dataset = CSVDataset(data_list)
    
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler, num_workers=0, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    
    # Get input shape from first sample
    sample_x, sample_y, _ = data_list[0][:3]
    in_channels = sample_x.shape[0]
    
    if is_main_process(rank):
        print(f"Input shape: {sample_x.shape}, Target shape: {sample_y.shape}")
        print(f"In channels: {in_channels}")
    
    # Create model using NeuralOperator library
    model = FNO(
        n_modes=(MODES, MODES),
        in_channels=in_channels,
        out_channels=1,
        hidden_channels=WIDTH,
        n_layers=N_LAYERS,
        positional_embedding='grid',  # Adds positional awareness
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        fno_skip='linear',  # Linear skip connections
        norm='instance_norm',  # Instance normalization
    ).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process(rank):
        print(f"Model parameters: {total_params:,}")
    
    # Loss functions
    l2_loss = LpLoss(d=2, p=2, reduction='mean')
    h1_loss = H1Loss(d=2, reduction='mean')
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Training logger
    logger = None
    if is_main_process(rank):
        logger = TrainingLogger(output_dir="training_logs")
        logger.start_training({
            'batch_size': BATCH,
            'epochs': EPOCHS,
            'learning_rate': LR,
            'modes': MODES,
            'width': WIDTH,
            'n_layers': N_LAYERS,
            'model': 'NeuralOperator FNO',
        })
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    if is_main_process(rank):
        print("\nStarting training...")
        print(f"Epochs: {EPOCHS}, Batch: {BATCH}, LR: {LR}, Patience: {PATIENCE}")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        if is_distributed:
            sampler.set_epoch(epoch)
        
        running_loss = 0.0
        running_l2 = 0.0
        running_h1 = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False) if is_main_process(rank) else loader
        
        for xb, yb, mb in pbar:
            xb = xb.float().to(device)
            yb = yb.float().to(device)
            mb = mb.float().to(device)
            
            pred = model(xb)
            
            # Combined loss: L2 + weighted H1
            loss_l2 = l2_loss(pred, yb)
            loss_h1 = h1_loss(pred, yb)
            loss = loss_l2 + GRAD_WEIGHT * loss_h1
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_size = xb.shape[0]
            running_loss += loss.item() * batch_size
            running_l2 += loss_l2.item() * batch_size
            running_h1 += loss_h1.item() * batch_size
            
            if is_main_process(rank) and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({"loss": f"{loss.item():.4e}"})
        
        scheduler.step()
        
        # Aggregate losses
        n_samples = len(dataset)
        avg_loss = running_loss / n_samples
        avg_l2 = running_l2 / n_samples
        avg_h1 = running_h1 / n_samples
        
        if is_main_process(rank):
            print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.6e} (L2: {avg_l2:.4e}, H1: {avg_h1:.4e})")
            
            # Save checkpoints
            if epoch % 100 == 0:
                os.makedirs("epochs", exist_ok=True)
                state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save(state_dict, f"epochs/{MODEL_OUT}.epoch{epoch}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save(state_dict, MODEL_OUT)
                print(f"  > New best loss! Saved {MODEL_OUT}")
            else:
                patience_counter += 1
                print(f"  > No improvement. Patience {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Log metrics
            if logger:
                logger.log_epoch(epoch, {
                    'total_loss': avg_loss,
                    'l2_loss': avg_l2,
                    'h1_loss': avg_h1,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'best_loss': best_loss,
                    'patience': patience_counter,
                })
    
    if is_main_process(rank):
        print(f"\nTraining finished. Best loss: {best_loss:.6e}")
        if logger:
            logger.finish_training({'best_loss': best_loss})
    
    cleanup()


if __name__ == "__main__":
    main()
