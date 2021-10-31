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
    ) -> None:

        super().__init__(
            block, layers,
            include_top=False, n_channels=n_channels, groups=groups, width_per_group=width_per_group,
            in_planes=in_planes, replace_stride_with_dilation=replace_stride_with_dilation, init_weight=False
        )

        self.do = nn.Dropout(dropout)

        out_channels = in_planes
        refine_block = []

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.p_ims1d2_outl1_dimred = conv3x3(2048, 512, bias=False)
        self.adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv3x3(1024, 256, bias=False)
        self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv3x3(512, 256, bias=False)
        self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv3x3(256, 256, bias=False)
        self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)

        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1 = self.do(x1)

        out = self.clf_conv(x1)
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
