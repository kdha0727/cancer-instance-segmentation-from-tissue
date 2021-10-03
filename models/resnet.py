# Resnet Backbone for RefineNet, but fully works.
import torch.nn as nn
from typing import List, Optional, Type, Union


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=stride, bias=bias)


class BasicBlock(nn.Module):

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: [nn.Module, None] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetBlock(nn.Sequential):
    pass


class ResNet(nn.Sequential):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            n_classes: int = 1000,
            n_channels: int = 3,
            groups: int = 1,
            width_per_group: int = 64,
            include_top: bool = True,
            in_planes: int = 64,
            dropout: float = 0.5,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            init_weight: bool = True,
    ) -> None:

        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.in_planes = planes = in_planes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.block = block
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False] * (len(layers) - 1)
        elif len(replace_stride_with_dilation) != len(layers) - 1:
            raise ValueError("replace_stride_with_dilation should be None or a {}-element tuple, got {}"
                             .format(len(layers) - 1, replace_stride_with_dilation))

        self.pre_resnet = nn.Sequential(
            nn.Conv2d(n_channels, planes, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        resnet_block = []
        for i, num_blocks in enumerate(layers):
            if not i:
                resnet_block.append(self._make_layer(block, planes, num_blocks, stride=1))
            else:
                resnet_block.append(self._make_layer(
                    block, planes, num_blocks, stride=2, dilate=replace_stride_with_dilation[i - 1]
                ))
            planes <<= 1
        self.resnet_block = nn.Sequential(*resnet_block)
        if include_top:
            self.post_resnet = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Dropout(dropout),
                nn.Linear(2 ** (5 + len(layers)) * block.expansion, n_classes),
            )
        if init_weight:
            self._init_weight()

    def extract_features(self, x):
        features = []
        x = self.pre_resnet(x)
        for layer in self.resnet_block:
            x = layer(x)
            features.append(x)
        return features

    def _init_weight(self):  # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False):

        in_planes = self.in_planes

        dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        downsample = None
        if stride != 1 or in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        kwargs = dict(groups=self.groups, base_width=self.base_width)

        layers = [block(in_planes, planes, stride, downsample, dilation=dilation, **kwargs)]

        # configure the number of next in-channel
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=self.dilation, **kwargs))

        return ResNetBlock(*layers)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


del List, Optional, Type, Union
