import torch
import torch.nn as nn
from neuralop.models import FNO

class SpatialAttention(nn.Module):
    """
    Simple spatial attention module to help the model focus on critical 
    flow regions (like wakes or acceleration zones).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        attn = self.conv(x)
        return x * attn

class HybridFNO(nn.Module):
    """
    Hybrid FNO Model:
    1. Spectral Learning (via neuralop FNO)
    2. Geometry Conditioning (SDF weighting)
    3. Spatial Attention (to focus on wakes)
    """
    def __init__(self, n_modes=(32, 32), in_channels=8, out_channels=1, hidden_channels=64, n_layers=4):
        super().__init__()
        
        # 1. Base FNO from neuraloperator
        self.fno = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            use_channel_mlp=True,
            positional_embedding='grid'
        )
        
        # 2. Attention Module
        self.attention = SpatialAttention(hidden_channels)
        
        # 3. Final refinement layers (Geometry-Aware)
        self.refinement = nn.Sequential(
            nn.Conv2d(hidden_channels + 1, hidden_channels, kernel_size=3, padding=1), # +1 for SDF skip
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Assume x has 8 channels, where channel 0 is SDF
        sdf = x[:, 0:1, :, :] 
        
        # Spectral pass
        feat = self.fno(x)
        
        # Attention pass
        feat = self.attention(feat)
        
        # Concatenate SDF for geometry conditioning in the final stage
        combined = torch.cat([feat, sdf], dim=1)
        
        out = self.refinement(combined)
        return out
