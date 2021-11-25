import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair  # noqa

from . import functional as _f


MaxPool2d = nn.MaxPool2d


class SpectralPool2d(nn.Module):

    def __init__(self, scale_factor):
        super(SpectralPool2d, self).__init__()
        self.scale_factor = _pair(scale_factor)

    def forward(self, x):
        return _f.spectral_pool2d(x, self.scale_factor)


class HydPool2d(nn.Module):

    def __init__(self, channels: int, kernel_size: int):
        super(HydPool2d, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size)
        self.spectral_pool = SpectralPool2d(1 / kernel_size)
        self.conv1x1 = nn.Conv2d(
            in_channels=int(channels * 2), out_channels=int(channels), kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x):
        max_pool2d = self.max_pool(x)
        spectral_pool2d = self.spectral_pool(x)
        concat_pool2d = torch.cat((max_pool2d, spectral_pool2d), dim=1)
        return self.conv1x1(concat_pool2d)


__all__ = ['MaxPool2d', 'SpectralPool2d', 'HydPool2d']
