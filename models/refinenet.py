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
                layer.append(conv3x3((out_planes if i or j else in_planes), out_planes, stride=1, bias=(not j)))
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
    ) -> None:

        super().__init__(
            block, layers,
            include_top=False, n_channels=n_channels, groups=groups, width_per_group=width_per_group,
            in_planes=in_planes, replace_stride_with_dilation=replace_stride_with_dilation, init_weight=False
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



def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=int(kernel_size / 2.0),
                groups=groups,
                bias=False,
            ),
            batchnorm(out_planes),
            nn.ReLU6(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=int(kernel_size / 2.0),
                groups=groups,
                bias=False,
            ),
            batchnorm(out_planes),
        )


class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block from https://arxiv.org/abs/1801.04381"""

    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super(InvertedResidualBlock, self).__init__()
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1)
        self.output = nn.Sequential(
            convbnrelu(in_planes, intermed_planes, 1),
            convbnrelu(
                intermed_planes,
                intermed_planes,
                3,
                stride=stride,
                groups=intermed_planes,
            ),
            convbnrelu(intermed_planes, out_planes, 1, act=False),
        )

    def forward(self, x):
        residual = x
        out = self.output(x)
        if self.residual:
            return out + residual
        else:
            return out


class MBv2(nn.Module):  # light weight
    """Net Definition"""

    mobilenet_config = [
        [1, 16, 1, 1],  # expansion rate, output channels, number of repeats, stride
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    in_planes = 32  # number of input channels
    num_layers = len(mobilenet_config)

    def __init__(self, num_classes):
        super(MBv2, self).__init__()

        self.layer1 = convbnrelu(3, self.in_planes, kernel_size=3, stride=2)
        c_layer = 2
        for t, c, n, s in self.mobilenet_config:
            layers = []
            for idx in range(n):
                layers.append(
                    InvertedResidualBlock(
                        self.in_planes,
                        c,
                        expansion_factor=t,
                        stride=s if idx == 0 else 1,
                    )
                )
                self.in_planes = c
            setattr(self, "layer{}".format(c_layer), nn.Sequential(*layers))
            c_layer += 1

        ## Light-Weight RefineNet ##
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)
        self.crp4 = self._make_crp(256, 256, 4)
        self.crp3 = self._make_crp(256, 256, 4)
        self.crp2 = self._make_crp(256, 256, 4)
        self.crp1 = self._make_crp(256, 256, 4)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.segm = conv3x3(256, num_classes, bias=True)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)  # x / 2
        l3 = self.layer3(x)  # 24, x / 4
        l4 = self.layer4(l3)  # 32, x / 8
        l5 = self.layer5(l4)  # 64, x / 16
        l6 = self.layer6(l5)  # 96, x / 16
        l7 = self.layer7(l6)  # 160, x / 32
        l8 = self.layer8(l7)  # 320, x / 32
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.Upsample(size=l6.size()[2:], mode="bilinear", align_corners=True)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.Upsample(size=l4.size()[2:], mode="bilinear", align_corners=True)(l5)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.Upsample(size=l3.size()[2:], mode="bilinear", align_corners=True)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        out_segm = self.segm(l3)

        return out_segm

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)


def mbv2(num_classes):
    """Constructs the network.
    Args:
        num_classes (int): the number of classes for the segmentation head to output.
    """
    return MBv2(num_classes)
