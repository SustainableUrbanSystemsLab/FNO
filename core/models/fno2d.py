import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of modes to keep in first dimension (H)
        self.modes2 = modes2 # Number of modes to keep in second dimension (W)
        self.scale = (1.0 / (in_channels * out_channels))
        # For rfft2 training on real input, we need weights for two corners
        # to correctly capture the symmetry of the spectrum.
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Real-to-complex FFT
        x_ft = torch.fft.rfft2(x)

        # Output shape matches rfft2 output: (B, C, H, W//2 + 1)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.complex64, device=x.device)
        
        # Multiply relevant corners
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Inverse complex-to-real FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=16, modes2=16, width=64, n_layers=4):
        super().__init__()
        self.width = width
        self.in_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.fourier_layers = nn.ModuleList([
            nn.Sequential(
                SpectralConv2d(width, width, modes1, modes2),
                nn.Conv2d(width, width, kernel_size=1)
            ) for _ in range(n_layers)
        ])
        self.activation = nn.GELU()
        self.out_proj = nn.Sequential(
            nn.Conv2d(width, width//2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width//2, out_channels, kernel_size=1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.in_proj(x)
        for block in self.fourier_layers:
            spec = block[0](x)
            point = block[1](x)
            x = x + spec + point # Residual connection
            x = self.activation(x)
        out = self.out_proj(x)
        return out

def compute_wake_loss(y_pred, y_target, sensor_mask, wake_threshold=-0.3):
    """Compute additional loss for wake regions (low wind speed areas)."""
    wake_mask = (y_target < wake_threshold).float()
    wake_error = ((y_pred - y_target) ** 2) * wake_mask * sensor_mask
    return wake_error.sum() / (wake_mask.sum() + 1e-8)

def compute_peak_loss(y_pred, y_target, sensor_mask, high_threshold=0.5, low_threshold=-0.5):
    """
    Optimized peak loss: Focus on extremes (High and Low wind speeds).
    Uses fixed thresholds instead of torch.quantile for 10x speedup.
    """
    high_mask = (y_target >= high_threshold).float()
    low_mask = (y_target <= low_threshold).float()
    extreme_mask = torch.maximum(high_mask, low_mask)
    extreme_error = ((y_pred - y_target) ** 2) * extreme_mask * sensor_mask
    return extreme_error.sum() / (extreme_mask.sum() + 1e-8)

def sensor_weighted_mse(y_pred, y_target, sensor_mask=None,
                        grad_weight=0.0, spectral_weight=0.0, 
                        peak_weight=0.0, wake_weight=0.0,
                        wake_threshold=-0.3,
                        return_components=False):
    """Multi-component loss for FNO training."""
    if sensor_mask is None:
        sensor_mask = torch.ones_like(y_pred)
    
    mse_loss = (sensor_mask * (y_pred - y_target) ** 2).sum() / (sensor_mask.sum() + 1e-8)
    
    gradient_loss = torch.tensor(0.0, device=y_pred.device)
    if grad_weight > 0:
        p_dx = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
        p_dy = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        t_dx = y_target[:, :, :, 1:] - y_target[:, :, :, :-1]
        t_dy = y_target[:, :, 1:, :] - y_target[:, :, :-1, :]
        m_dx = sensor_mask[:, :, :, 1:]
        m_dy = sensor_mask[:, :, 1:, :]
        gradient_loss = ((p_dx - t_dx)**2 * m_dx).mean() + ((p_dy - t_dy)**2 * m_dy).mean()
    
    spectral_loss = torch.tensor(0.0, device=y_pred.device)
    if spectral_weight > 0:
        pred_fft = torch.fft.rfft2(y_pred, norm='ortho')
        tgt_fft = torch.fft.rfft2(y_target, norm='ortho')
        spectral_loss = torch.mean(torch.abs(pred_fft - tgt_fft) ** 2)
    
    peak_loss = torch.tensor(0.0, device=y_pred.device)
    if peak_weight > 0:
        peak_loss = compute_peak_loss(y_pred, y_target, sensor_mask)
    
    wake_loss = torch.tensor(0.0, device=y_pred.device)
    if wake_weight > 0:
        wake_loss = compute_wake_loss(y_pred, y_target, sensor_mask, wake_threshold=wake_threshold)
    
    total_loss = (mse_loss + 
                  grad_weight * gradient_loss + 
                  spectral_weight * spectral_loss + 
                  peak_weight * peak_loss +
                  wake_weight * wake_loss)
    
    if return_components:
        return total_loss, {
            'mse_loss': mse_loss.item(),
            'gradient_loss': gradient_loss.item(),
            'spectral_loss': spectral_loss.item(),
            'peak_loss': peak_loss.item(),
            'wake_loss': wake_loss.item(),
        }
    return total_loss
