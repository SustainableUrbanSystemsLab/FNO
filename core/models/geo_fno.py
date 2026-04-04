import torch
import torch.nn as nn
from neuralop.models import FNO

class LocalGeometryEncoder(nn.Module):
    """
    Learns to 'deform' and encode the strict physical geometry (SDF, X, Y)
    into a feature-rich latent space before passing it to the global spectral FNO.
    This effectively acts like the mapping phase of Geo-FNO for uniform grids.
    """
    def __init__(self, in_channels, latent_channels):
        super().__init__()
        # Use 1x1 convolutions like a point-wise MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(latent_channels // 2, latent_channels, 1),
            nn.GELU()
        )

    def forward(self, x):
        return self.mlp(x)

class BoundaryReconstructor(nn.Module):
    """
    A high-frequency spatial decoder that re-injects the physical geometry mask (SDF) 
    at the very end of the network. This prevents the FNO's spectral blurring 
    from bleeding into the building wakes.
    """
    def __init__(self, fno_channels, out_channels):
        super().__init__()
        # Takes the FNO flow output + original SDF
        self.decode = nn.Sequential(
            nn.Conv2d(fno_channels + 1, fno_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fno_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, fno_out, sdf):
        # Concatenate SDF as a hard physical boundary constraint
        x = torch.cat([fno_out, sdf], dim=1)
        return self.decode(x)

class GeoFNO(nn.Module):
    """
    Grid-based Geometry-Aware FNO.
    1. Encodes (X, Y, SDF, Wind Conditions) into a mapped latent space.
    2. Runs Fourier Spectral convolutions to solve global pressure/momentum fields.
    3. Uses a Boundary Reconstructor that forces sharp edges along the True SDF.
    """
    def __init__(self, n_modes=(32, 32), in_channels=8, out_channels=1, hidden_channels=64, n_layers=4):
        super().__init__()
        
        # 1. Coordinate & Geometry Mapping
        self.geo_encode = LocalGeometryEncoder(in_channels=in_channels, latent_channels=hidden_channels)
        
        # 2. Spectral Solver (FNO)
        self.fno = FNO(
            n_modes=n_modes,
            in_channels=hidden_channels, 
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            use_channel_mlp=True,
            positional_embedding=None # We handle geometry implicitly in the encoder
        )
        
        # 3. Sharp Boundary Reconstructor
        self.reconstruct = BoundaryReconstructor(fno_channels=hidden_channels, out_channels=out_channels)

    def forward(self, x):
        # Extract the SDF channel (assuming it is channel 0 based on our data pipeline)
        sdf = x[:, 0:1, :, :]
        
        # Map physical space to latent continuous geometry
        latent_geo = self.geo_encode(x)
        
        # Calculate global fluid dynamics (this will be blurry)
        flow_field = self.fno(latent_geo)
        
        # Re-inject the physical geometry to force exact, sharp boundaries and wakes
        return self.reconstruct(flow_field, sdf)
