import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.complex64))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.complex64))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier transform
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.complex64, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to spatial domain
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=16, modes2=16, width=64, n_layers=4):
        super().__init__()
        self.width = width
        self.in_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        
        self.spectral_layers = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)])
        self.spatial_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=3, padding=1) for _ in range(n_layers)])
        
        self.activation = nn.GELU()
        self.out_proj = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.in_proj(x)
        for spec, spat in zip(self.spectral_layers, self.spatial_layers):
            x1 = spec(x)
            x2 = spat(x)
            x = x + x1 + x2
            x = self.activation(x)
        out = self.out_proj(x)
        return out

def sensor_weighted_mse(pred, target, sensor_mask=None):
    """pred,target: (B,C,H,W). sensor_mask: (B,1,H,W) or None."""
    if sensor_mask is None:
        return F.mse_loss(pred, target)
    mask = sensor_mask.repeat(1, pred.shape[1], 1, 1)
    diff2 = (pred - target)**2 * mask
    denom = mask.sum()
    return diff2.sum() / (denom + 1e-8)
