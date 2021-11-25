import torch
import torch.nn as nn

from .normalization import SwitchNorm2d
from .pooling import HydPool2d

from collections import OrderedDict

from typing import *


class SkipConnection(nn.Module):

    @staticmethod
    def forward(x, skip):
        diff_y = skip.size()[-2] - x.size()[-2]
        diff_x = skip.size()[-1] - x.size()[-1]
        if diff_x > 0 or diff_y > 0:
            pad = [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
            x = torch.constant_pad_nd(x, pad=pad, value=0)
        return torch.cat([skip, x], dim=1)


class ConvBnRelu2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False):  # todo: bias?
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4)  # todo: eps setting?
        self.relu = nn.ReLU(inplace=True)


class DoubleConv2d(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
    ):
        mid_channels = mid_channels or out_channels
        super().__init__(
            *ConvBnRelu2d(in_channels, mid_channels, kernel_size=(3, 3), padding=(1, 1), bias=True),  # todo: bias?
            *ConvBnRelu2d(mid_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=True),  # todo: bias?
        )


class Inception(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
    ):
        if mid_channels is not None:
            import warnings
            warnings.warn("In Inception block, mid channels argument will be ignored.")
        super().__init__()
        self.conv1x1 = ConvBnRelu2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv3x3 = ConvBnRelu2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv5x5 = ConvBnRelu2d(in_channels, out_channels, kernel_size=(5, 5), padding=(2, 2))
        self.filter = nn.Conv2d(out_channels * 3 + in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))

    def _inception_forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        return torch.cat((x1, x3, x5, x), dim=1)

    def forward(self, x):
        x_cat = self._inception_forward(x)
        return self.filter(x_cat)


class InceptionCenter(Inception):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_channels: int,
            n_blocks: int,
            hybrid_pool:bool,
    ):
        super().__init__(in_channels, out_channels)
        sn_scale_factor = 2 ** (n_blocks - 1)
        if hybrid_pool:
            self.sn_pool = HydPool2d(n_channels, sn_scale_factor)
        else:
            self.sn_pool = nn.MaxPool2d(sn_scale_factor)
        self.filter = nn.Conv2d(
            out_channels * 3 + in_channels + n_channels, out_channels, kernel_size=(1, 1), padding=(0, 0)
        )

    def _forward_impl(self, x, original):
        x_cat = self._inception_forward(x)
        sn_pool = self.sn_pool(original)
        x_cat = torch.cat((x_cat, sn_pool), dim=1)
        return self.filter(x_cat)

    forward = _forward_impl


class DownConv(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int],
            block: Optional[Type[Union[DoubleConv2d, Inception]]],
            hybrid_pool: bool = False,
    ):
        super().__init__()
        if hybrid_pool:
            self.down = HydPool2d(in_channels, 2)
        else:
            self.down = nn.MaxPool2d(2)
        if block is not None:
            self.conv = block(in_channels, out_channels)
        else:
            self.add_module('conv', None)  # just add namespace for wrap_conv class method.

    @classmethod
    def wrap_conv(cls, conv: nn.Module, in_channels: int, hybrid_pool: bool = False):
        self = cls(in_channels=in_channels, out_channels=None, block=None, hybrid_pool=hybrid_pool)
        self.conv = conv
        return self

    def forward(self, x, cat=None):
        down = self.down(x)
        if cat is None:
            return self.conv(down)
        else:
            return self.conv(down, cat)


class UpConv(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            block: Type[Union[DoubleConv2d, Inception]],
            bilinear=False
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.skip_conn = SkipConnection()
            self.conv = block(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.skip_conn = SkipConnection()
            self.conv = block(in_channels, out_channels)

    def forward(self, x, skip):
        return self.conv(self.skip_conn(self.up(x), skip))


class EncoderPath(nn.Sequential):

    def __init__(
            self,
            in_channels: int,
            start_filters: int,
            n_blocks: int,
            bilinear: bool = False,
            hybrid_pool: bool = False,
            block: Type[Union[DoubleConv2d, Inception]] = DoubleConv2d
    ):
        self.in_channels = in_channels
        self.start_filters = start_filters
        self.n_blocks = n_blocks
        self.bilinear = bilinear
        layers = []
        assert n_blocks >= 2
        for i in range(1, n_blocks + 1):
            if i == 1:
                layer = block(in_channels, start_filters)
                in_channels = start_filters
            elif i == n_blocks:
                layer = DownConv(in_channels, in_channels * (1 if bilinear else 2), block, hybrid_pool)
            else:
                layer = DownConv(in_channels, in_channels * 2, block, hybrid_pool)
                in_channels *= 2
            layers.append(layer)
        super().__init__(*layers)

    def _forward_impl(self, x):
        result = OrderedDict()
        out = x
        for i, layer in enumerate(self, 1):
            out = layer(out)
            result[i] = out
        return result

    forward = _forward_impl


class InceptionEncoderPath(EncoderPath):

    def __init__(
            self,
            in_channels,
            start_filters,
            n_blocks,
            bilinear=False,
            hybrid_pool=True,
            block=Inception,
            center=InceptionCenter,
    ):
        super().__init__(
            in_channels, start_filters, n_blocks,
            bilinear=bilinear, hybrid_pool=hybrid_pool, block=block
        )
        last_in = start_filters * 2 ** (n_blocks - 2)
        last_out = last_in * (1 if bilinear else 2)
        self[-1] = DownConv.wrap_conv(
            center(last_in, last_out, in_channels, n_blocks, hybrid_pool=hybrid_pool),
            in_channels=last_in, hybrid_pool=hybrid_pool
        )

    def _forward_impl(self, x):
        result = OrderedDict()
        out = x
        length = len(self)
        for i, layer in enumerate(self, 1):
            if i == length:
                out = layer(out, x)
            else:
                out = layer(out)
            result[i] = out
        return result

    forward = _forward_impl


class DecoderPath(nn.Sequential):

    def __init__(
            self,
            out_channels: int,
            start_filters: int,
            n_blocks: int,
            bilinear: bool = False,
            block: Type[Union[DoubleConv2d, Inception]] = DoubleConv2d
    ):
        self.bilinear = bilinear
        in_channels = start_filters * 2 ** (n_blocks - 1)
        layers = []
        for i in range(1, n_blocks + 1):
            if i == n_blocks:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)))
            elif i == n_blocks - 1:
                layers.append(UpConv(in_channels, in_channels // 2, bilinear=bilinear, block=block))
            else:
                layers.append(UpConv(in_channels, in_channels // (4 if bilinear else 2), bilinear=bilinear, block=block))
            in_channels //= 2
        super().__init__(*layers)

    def _forward_impl(self, x):
        assert len(x) == len(self), "Input length must be same with decoder path."
        if isinstance(x, OrderedDict):
            x = list(x.values())
        else:
            x = list(x)
        for layer in self:
            if len(x) == 1:
                return layer(x.pop())
            x.append(layer(x.pop(), x.pop()))

    forward = _forward_impl


class UNet(nn.Sequential):  # add simply last activation layer with `net.add_module('activation', nn.Sigmoid())`

    def __init__(self, n_channels, n_classes, start_filters=64, depth=5, bilinear=False, hybrid_pool=False):

        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoder = EncoderPath(n_channels, start_filters, depth, bilinear=bilinear, hybrid_pool=hybrid_pool, block=DoubleConv2d)
        self.decoder = DecoderPath(n_classes, start_filters, depth, bilinear=bilinear, block=DoubleConv2d)


class InceptionUNet(nn.Sequential):

    def __init__(self, n_channels, n_classes, start_filters=16, depth=5, bilinear=False,hybrid_pool=True):

        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.swithnorm = SwitchNorm2d(n_channels)
        self.encoder = InceptionEncoderPath(
            n_channels, start_filters, depth, bilinear=bilinear,
            hybrid_pool=hybrid_pool, block=Inception, center=InceptionCenter
        )
        self.decoder = DecoderPath(n_classes, start_filters, depth, bilinear=bilinear, block=Inception)


# class StackEncoder(nn.Module):  # inception backbone
#
#     def __init__(self, x_channels, y_channels):
#         super().__init__()
#         self.inception = Inception(x_channels, y_channels)
#         self.pool = HydPool2d(y_channels, kernel_size=2)
#
#     def forward(self, x):
#         x_conv = self.inception(x)
#         xpool = self.pool(x_conv)
#         return x_conv, xpool
#
#
# class StackDecoder(nn.Module):
#     def __init__(self, x_channels, y_channels):
#         super(StackDecoder, self).__init__()
#         self.inception = Inception(x_channels, y_channels)
#         self.skip_conn = SkipConnection()
#         self.upsample = nn.ConvTranspose2d(x_channels, y_channels, kernel_size=(2, 2), stride=(2, 2))
#
#     def forward(self, x, down_tensor):
#         x = self.upsample(x)
#         x = self.skip_conn(x, down_tensor)
#         return self.inception(x)
#
#
# class Center(nn.Module):
#     def __init__(self, x_channels, y_channels):
#         super().__init__()
#         self.conv1x1 = ConvBnRelu2d(x_channels, y_channels, kernel_size=(1, 1), padding='same')
#         self.conv3x3 = ConvBnRelu2d(x_channels, y_channels, kernel_size=(3, 3), padding='same')
#         self.conv5x5 = ConvBnRelu2d(x_channels, y_channels, kernel_size=(5, 5), padding='same')
#         self.selfconv = nn.Conv2d(y_channels * 3 + 3 + x_channels, y_channels, kernel_size=1, padding='same')
#
#     def forward(self, x, SNpool):
#         x1 = self.conv1x1(x)
#         x3 = self.conv3x3(x)
#         x5 = self.conv5x5(x)
#         x_cat = torch.cat((x1, x3, x5, x, SNpool), 1)
#         x_conv = self.selfconv(x_cat)
#         return x_conv
#
#
# class InceptionUNet(nn.Module):
#     def __init__(self, in_channels, n_classes, filters=16):
#         super(InceptionUNet, self).__init__()
#         self.SN = SwitchNorm2d(in_channels)
#         self.SNpool = HydPool2d(in_channels, kernel_size=16)
#
#         self.down1 = StackEncoder(in_channels, filters)  ## 16
#         self.down2 = StackEncoder(filters, filters * 2)  ## 32
#         self.down3 = StackEncoder(filters * 2, filters * 2 * 2)  ## 64
#         self.down4 = StackEncoder(filters * 2 * 2, filters * 2 * 2 * 2)  ## 128
#
#         self.center = Center(filters * 2 * 2 * 2, filters * 2 * 2 * 2 * 2)  ## 256
#
#         self.up4 = StackDecoder(filters * 2 * 2 * 2 * 2, filters * 2 * 2 * 2)  ## 128
#         self.up3 = StackDecoder(filters * 2 * 2 * 2, filters * 2 * 2)  ## 64
#         self.up2 = StackDecoder(filters * 2 * 2, filters * 2)  ## 32
#         self.up1 = StackDecoder(filters * 2, filters)  ## 16
#         self.classify = nn.Conv2d(filters, n_classes, kernel_size=1, bias=True)
#
#     def forward(self, x):
#         x = self.SN(x)  # !!! switch norm !!!
#         SNpool = self.SNpool(x)
#
#         down1, out = self.down1(x)  # 'down' for concat, 'out' for pool
#         down2, out = self.down2(out)
#         down3, out = self.down3(out)
#         down4, out = self.down4(out)
#
#         out = self.center(out, SNpool)
#
#         out = self.up4(out, down4)
#         out = self.up3(out, down3)
#         out = self.up2(out, down2)
#         out = self.up1(out, down1)
#         out = self.classify(out)
#         return out
