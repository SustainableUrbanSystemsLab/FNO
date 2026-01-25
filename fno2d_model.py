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

def spectral_loss(pred, target, mask=None):
    """FFT-based loss to preserve high-frequency spatial features.
    
    Compares the magnitude of Fourier coefficients between prediction and target,
    which helps preserve sharp edges and fine-scale flow features.
    """
    # Apply mask before FFT to focus on valid regions
    if mask is not None:
        pred_masked = pred * mask
        target_masked = target * mask
    else:
        pred_masked = pred
        target_masked = target
    
    # Compute 2D FFT
    pred_fft = torch.fft.rfft2(pred_masked)
    target_fft = torch.fft.rfft2(target_masked)
    
    # Compare magnitudes (ignoring phase for stability)
    pred_mag = pred_fft.abs()
    target_mag = target_fft.abs()
    
    # Log-scale to balance low and high frequencies
    # Add small epsilon for numerical stability
    pred_log = torch.log1p(pred_mag)
    target_log = torch.log1p(target_mag)
    
    return F.mse_loss(pred_log, target_log)


def physics_loss(pred, target, mask=None, grad_weight=0.15, spectral_weight=0.05, return_components=False):
    """Combined MSE, Gradient Loss, Spectral Loss, and Acceleration Curb.
    
    Args:
        pred: Predicted field (B, C, H, W)
        target: Target field (B, C, H, W)
        mask: Optional weight mask (B, C, H, W)
        grad_weight: Weight for gradient loss (default 0.15)
        spectral_weight: Weight for spectral/FFT loss (default 0.05)
        return_components: If True, return dict with individual loss components
    
    Returns:
        If return_components=False: total_loss (tensor)
        If return_components=True: (total_loss, components_dict)
    """
    # 1. Main Weighted MSE
    m = mask if mask is not None else torch.ones_like(pred)
    
    # Precision Fix: Acceleration Curb
    # If the target is acceleration (>0) and we over-predict it, punish harder.
    # This specifically stops high-wind areas from getting "too large".
    diff = pred - target
    penalty = torch.ones_like(diff)
    # Target > 0 AND Pred > Target (Overshooting high wind)
    high_wind_overshoot = (target > 0) & (diff > 0)
    penalty[high_wind_overshoot] *= 2.0 
    
    mse = (diff**2 * penalty * m).sum() / (m.sum() + 1e-8)
    
    # 2. Gradient Sharpness (Sobel-like)
    def get_grads(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy
    
    p_dx, p_dy = get_grads(pred)
    t_dx, t_dy = get_grads(target)
    
    # Mask gradients
    m_dx = mask[:, :, :, 1:] if mask is not None else 1.0
    m_dy = mask[:, :, 1:, :] if mask is not None else 1.0
    
    grad_loss = (((p_dx - t_dx)**2 * m_dx).mean() + ((p_dy - t_dy)**2 * m_dy).mean())
    
    # 3. Spectral Loss (preserve high-frequency features)
    spec_loss = spectral_loss(pred, target, mask) if spectral_weight > 0 else torch.tensor(0.0, device=pred.device)
    
    total_loss = mse + grad_weight * grad_loss + spectral_weight * spec_loss
    
    if return_components:
        components = {
            'mse_loss': float(mse.item()),
            'gradient_loss': float(grad_loss.item()),
            'spectral_loss': float(spec_loss.item()) if isinstance(spec_loss, torch.Tensor) else 0.0,
        }
        return total_loss, components
    
    return total_loss


def sensor_weighted_mse(pred, target, sensor_mask=None, grad_weight=0.15, spectral_weight=0.05, return_components=False):
    """Backward compatible wrapper with configurable loss weights."""
    return physics_loss(pred, target, sensor_mask, grad_weight, spectral_weight, return_components)

