import torch.nn as nn
import torch.nn.functional as F

from typing import *

from .resnet import BasicBlock, Bottleneck, ResNet, conv3x3, conv1x1


class CRPBlock(nn.Sequential):

    def __init__(self, in_planes, out_planes, n_stages):
        layers = []
        for i in range(n_stages):
            layers.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                conv3x3(out_planes if i else in_planes, out_planes, stride=1, bias=False)
            ))
        super().__init__(*layers)

    def forward(self, x):
        y = x
        for layer in self:
            x = layer(x)
            y += x
        return y


class RCUBlock(nn.Sequential):

    def __init__(self, in_planes, out_planes, n_blocks, n_stages):
        layers = []
        for i in range(n_blocks):
            layer = []
            for j in range(n_stages):
                layer.append(nn.ReLU())
                layer.append(conv3x3((in_planes if i or j else out_planes), out_planes, stride=1, bias=(not j)))
            layers.append(nn.Sequential(*layer))
        super().__init__(*layers)

    def forward(self, x):
        for layer in self:
            residual = x
            x = layer(x)
            x += residual
        return x


class RefineBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            mid_channels: int,
            out_channels: int = None,
            top: bool = False,
            bottom: bool = False,
            light_weight: bool = False,
            
    ) -> None:
        
        assert not (top and bottom)
        assert bottom or out_channels

        super().__init__()
        self.top = top
        self.bottom = bottom
        self.light_weight = light_weight
        conv = conv1x1 if light_weight else conv3x3

        self.p_ims1d2_out_dimred = conv(in_channels, mid_channels, bias=False)
        if not light_weight:
            self.adapt_stage_b = RCUBlock(mid_channels, mid_channels, 2, 2)  # light weight: omit
        if not top:
            self.adapt_stage_b2_joint_varout_dimred = conv(256, 256, bias=False)
        self.mflow_conv_g_pool = CRPBlock(mid_channels, mid_channels, 4)
        if not light_weight:
            self.mflow_conv_g_b = RCUBlock(mid_channels, mid_channels, 3, 2)  # light weight: omit
        if not bottom:
            self.mflow_conv_g_b3_joint_varout_dimred = conv(mid_channels, out_channels, bias=False)

    def forward(self, x, prior=None):

        top = self.top
        bottom = self.bottom
        light_weight = self.light_weight
        assert (prior is not None) ^ top

        x = self.p_ims1d2_out_dimred(x)
        if not light_weight:
            x = self.adapt_stage_b(x)
        if not top:
            x = self.adapt_stage_b2_joint_varout_dimred(x)
            x = x + F.interpolate(prior, size=x.size()[-2:], mode="bilinear", align_corners=True)
        x = F.relu(x)
        x = self.mflow_conv_g_pool(x)
        if not light_weight:
            x = self.mflow_conv_g_b(x)
        if not bottom:
            x = self.mflow_conv_g_b3_joint_varout_dimred(x)
        return x


class RefineNet(ResNet):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            n_classes: int = 1000,
            n_channels: int = 3,
            groups: int = 1,
            width_per_group: int = 64,
            light_weight: bool = False,
            in_planes: int = 64,
            dropout: float = 0.5,
            min_refine_planes: int = 256,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            init_weight: bool = True,
            pretrained : bool = True,
    ) -> None:

        super().__init__(
            block, layers,
            include_top=False, n_channels=n_channels, groups=groups, width_per_group=width_per_group,
            in_planes=in_planes, replace_stride_with_dilation=replace_stride_with_dilation, init_weight=False, pretrained = True
        )

        self.do = nn.Dropout(dropout)

        out_channels = in_planes
        refine_block = []

        for i in reversed(range(len(layers))):

            mid_channels = in_planes << i
            in_channels = max(mid_channels * block.expansion, min_refine_planes)
            out_channels = max(mid_channels >> 1, min_refine_planes)
            mid_channels = max(mid_channels, min_refine_planes)

            if i + 1 == len(layers):  # first
                layer = RefineBlock(in_channels, mid_channels, out_channels, top=True, light_weight=light_weight)
            elif i == 0:  # last
                layer = RefineBlock(in_channels, mid_channels, out_channels, bottom=True, light_weight=light_weight)
            else:
                layer = RefineBlock(in_channels, mid_channels, out_channels, light_weight=light_weight)

            refine_block.append(layer)

        self.refine_block = nn.Sequential(*refine_block)
        self.classifier = nn.Conv2d(
            out_channels, n_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True
        )

        if init_weight:
            self._init_weight()

    def forward(self, x):
        features = list(self.extract_features(x))
        for i, feature in enumerate(features):
            if i >= 2:
                features[i] = self.do(feature)
        for i, (feature, layer) in enumerate(zip(reversed(features), self.refine_block)):
            if not i:  # first
                x = layer(feature)
            else:
                x = layer(feature, x)
        x = self.do(x)
        out = self.classifier(x)
        return out


def refinenet50(n_classes, **kwargs):
    return RefineNet(Bottleneck, [3, 4, 6, 3], n_classes=n_classes, **kwargs)


def refinenet101(n_classes, **kwargs):
    return RefineNet(Bottleneck, [3, 4, 23, 3], n_classes=n_classes, **kwargs)


def refinenet152(n_classes, **kwargs):
    return RefineNet(Bottleneck, [3, 8, 36, 3], n_classes=n_classes, **kwargs)


def rf_lw50(n_classes, **kwargs):
    return RefineNet(Bottleneck, [3, 4, 6, 3], n_classes=n_classes, light_weight=True, **kwargs)


def rf_lw101(n_classes, **kwargs):
    return RefineNet(Bottleneck, [3, 4, 23, 3], n_classes=n_classes, light_weight=True, **kwargs)


def rf_lw152(n_classes, **kwargs):
    return RefineNet(Bottleneck, [3, 8, 36, 3], n_classes=n_classes, light_weight=True, **kwargs)
