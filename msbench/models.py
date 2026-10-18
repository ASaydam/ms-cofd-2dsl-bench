"""
The reference networks of the benchmark (docs/tasks-and-metrics.md), plain PyTorch:

    UNet    4 levels, two 3x3 convolutions (GroupNorm, GELU) per level, max pooling and transposed convolutions
    FNO2d   Fourier neural operator (Li et al. 2021): a lifting layer, Fourier layers that keep the lowest modes,
            and a projection

Both map (batch, in_channels, nx, ny) to (batch, out_channels, nx, ny) for any grid size (the UNet pads to a multiple
of 2^levels and crops back).
"""

import torch
import torch.nn.functional as F
from torch import nn


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(8, out_channels), nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(8, out_channels), nn.GELU())


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, width=32, levels=4):
        super().__init__()
        widths = [width * 2 ** i for i in range(levels + 1)]
        self.levels = levels
        self.down = nn.ModuleList([block(in_channels, widths[0])] +
                                  [block(widths[i], widths[i + 1]) for i in range(levels)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
                                 for i in reversed(range(levels))])
        self.merge = nn.ModuleList([block(2 * widths[i], widths[i]) for i in reversed(range(levels))])
        self.out = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x):
        nx, ny = x.shape[-2:]
        step = 2 ** self.levels
        x = F.pad(x, (0, (-ny) % step, 0, (-nx) % step))
        skips = []
        for i, down in enumerate(self.down):
            x = down(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, merge in zip(self.up, self.merge):
            x = merge(torch.cat([up(x), skips.pop()], dim=1))
        return self.out(x)[..., :nx, :ny]


class SpectralConv2d(nn.Module):
    """Fourier layer: FFT, a learned complex weight on the lowest modes in both directions, inverse FFT"""

    def __init__(self, channels, modes_x, modes_y):
        super().__init__()
        self.modes_x, self.modes_y = modes_x, modes_y
        scale = 1 / (channels * channels)
        self.weight_low = nn.Parameter(scale * torch.randn(channels, channels, modes_x, modes_y, dtype=torch.cfloat))
        self.weight_high = nn.Parameter(scale * torch.randn(channels, channels, modes_x, modes_y, dtype=torch.cfloat))

    def forward(self, x):
        nx, ny = x.shape[-2:]
        spectrum = torch.fft.rfft2(x.float())
        mx, my = min(self.modes_x, nx // 2), min(self.modes_y, ny // 2 + 1)
        out = torch.zeros_like(spectrum)
        out[:, :, :mx, :my] = torch.einsum('bixy,ioxy->boxy', spectrum[:, :, :mx, :my], self.weight_low[:, :, :mx, :my])
        out[:, :, -mx:, :my] = torch.einsum('bixy,ioxy->boxy', spectrum[:, :, -mx:, :my],
                                            self.weight_high[:, :, :mx, :my])
        return torch.fft.irfft2(out, s=(nx, ny))


class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width=64, modes_x=48, modes_y=16, layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels + 2, width, 1)              # + the two coordinates of the grid
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes_x, modes_y) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_channels, 1))

    def forward(self, x):
        b, _, nx, ny = x.shape
        gx = torch.linspace(0, 1, nx, device=x.device, dtype=x.dtype)[None, None, :, None].expand(b, 1, nx, ny)
        gy = torch.linspace(0, 1, ny, device=x.device, dtype=x.dtype)[None, None, None, :].expand(b, 1, nx, ny)
        x = self.lift(torch.cat([x, gx, gy], dim=1))
        for i, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            x = spectral(x).to(x.dtype) + local(x)
            if i < len(self.spectral) - 1:
                x = F.gelu(x)
        return self.project(x)


MODELS = {'unet': UNet, 'fno': FNO2d}
