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
        self.weight_real = nn.Parameter( (1.0/(in_channels*out_channels)) * torch.randn(in_channels, out_channels, modes1, modes2) )
        self.weight_imag = nn.Parameter( (1.0/(in_channels*out_channels)) * torch.randn(in_channels, out_channels, modes1, modes2) )

    def compl_mul2d(self, input_ft, weight_r, weight_i):
        x = input_ft[:, :, :self.modes1, :self.modes2]
        xr = x.real.unsqueeze(2)
        xi = x.imag.unsqueeze(2)
        wr = weight_r.unsqueeze(0)
        wi = weight_i.unsqueeze(0)
        out_r = (xr * wr - xi * wi).sum(dim=1)
        out_i = (xr * wi + xi * wr).sum(dim=1)
        return torch.complex(out_r, out_i)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.fft2(x, dim=(-2, -1))
        out_ft = torch.zeros((B, self.out_channels, H, W), dtype=torch.complex64, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft, self.weight_real, self.weight_imag)
        x_out = torch.fft.ifft2(out_ft, dim=(-2, -1)).real
        return x_out

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
        x = self.in_proj(x)
        for block in self.fourier_layers:
            spec = block[0](x)
            point = block[1](x)
            x = x + spec + point
            x = self.activation(x)
        out = self.out_proj(x)
        return out

def physics_loss(pred, target, mask=None, grad_weight=0.1):
    """Combined MSE and Gradient Loss for sharpness.
    pred, target: (B,C,H,W)
    mask: (B,1,H,W)
    grad_weight: importance of sharpness vs magnitude
    """
    # 1. Main Weighted MSE
    m = mask if mask is not None else torch.ones_like(pred)
    mse = ((pred - target)**2 * m).sum() / (m.sum() + 1e-8)
    
    # 2. Gradient Sharpness (Sobel-like)
    # Penalize the 'blurry' edges
    def get_grads(x):
        # x is (B,C,H,W)
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy
    
    p_dx, p_dy = get_grads(pred)
    t_dx, t_dy = get_grads(target)
    
    # Mask gradients (simple approximation for simplicity)
    m_dx = mask[:, :, :, 1:] if mask is not None else 1.0
    m_dy = mask[:, :, 1:, :] if mask is not None else 1.0
    
    grad_loss = (((p_dx - t_dx)**2 * m_dx).mean() + ((p_dy - t_dy)**2 * m_dy).mean())
    
    return mse + grad_weight * grad_loss

def sensor_weighted_mse(pred, target, sensor_mask=None):
    # Backward compatibility
    return physics_loss(pred, target, sensor_mask)
