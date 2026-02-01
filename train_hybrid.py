"""
Training Script for Hybrid FNO Model
=====================================
Integrates:
- HybridFNO Model (Spectral + Attention + SDF Conditioning)
- Physics-Informed Loss (Divergence / Continuity)
- H1Loss (Sobolev) for sharpness
"""

import os, sys, glob, torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import tomllib

# Local imports
from fno_hybrid_model import HybridFNO, physics_informed_loss
from train_neuralop import load_or_prepare_dataset, CSVDataset, pad_collate_fn
from neuralop.losses import LpLoss, H1Loss
from training_logger import TrainingLogger

def train():
    # 1. Load Config
    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)
    
    # Paths & Params
    data_folder = config['paths']['data_folder_windows']
    batch_size = config['training'].get('batch_size', 4)
    epochs = 20 # Small run for demo
    lr = 5e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load Data
    csv_files = sorted(glob.glob(os.path.join(data_folder, '**/*.csv'), recursive=True))[:50] # Subset for speed
    Xs, Ys, Masks, chs = load_or_prepare_dataset(csv_files, rank=0, is_main=True)
    
    dataset = CSVDataset(list(zip(Xs, Ys, Masks, [chs]*len(Xs))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    # 3. Create Hybrid Model
    in_channels = Xs[0].shape[0]
    model = HybridFNO(
        n_modes=(32, 32),
        in_channels=in_channels,
        out_channels=1,
        hidden_channels=64,
        n_layers=4
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Losses
    l2_loss = LpLoss(d=2, p=2)
    h1_loss = H1Loss(d=2)
    
    print("\nStarting Hybrid Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for xb, yb, mb in tqdm(loader, desc=f"Epoch {epoch}"):
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            
            pred = model(xb)
            
            # Hybrid Loss Components
            loss_data = l2_loss(pred, yb)
            loss_grad = h1_loss(pred, yb)
            
            # Physics-Informed component (Divergence regularizer)
            loss_phys = physics_informed_loss(pred, yb, xb, device)
            
            # Combine
            loss = loss_data + 0.3 * loss_grad + 0.1 * loss_phys
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch} Avg Loss: {total_loss/len(loader):.6e}")

if __name__ == "__main__":
    train()
