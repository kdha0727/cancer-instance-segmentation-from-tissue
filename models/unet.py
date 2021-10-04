import torch
import torch.nn as nn

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


# https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py

class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        mean_weight = torch.softmax(self.mean_weight, dim=0)
        var_weight = torch.softmax(self.var_weight, dim=0)

        if self.using_bn:

            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn

        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


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
    ):
        super().__init__(in_channels, out_channels)
        sn_scale_factor = 2 ** (n_blocks - 1)
        self.sn_pool = HydPool2d(n_channels, sn_scale_factor)
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
            center(last_in, last_out, in_channels, n_blocks),
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

    def __init__(self, n_channels, n_classes, start_filters=64, depth=5, bilinear=False):

        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoder = EncoderPath(n_channels, start_filters, depth, bilinear=bilinear, block=DoubleConv2d)
        self.decoder = DecoderPath(n_classes, start_filters, depth, bilinear=bilinear, block=DoubleConv2d)


class InceptionUNet(nn.Sequential):

    def __init__(self, n_channels, n_classes, start_filters=32, depth=5, bilinear=False):

        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoder = InceptionEncoderPath(
            n_channels, start_filters, depth, bilinear=bilinear,
            hybrid_pool=True, block=Inception, center=InceptionCenter
        )
        self.decoder = DecoderPath(n_classes, start_filters, depth, bilinear=bilinear, block=Inception)
