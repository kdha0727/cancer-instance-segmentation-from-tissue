import collections
import math
import torch
import torch.nn.functional as _F  # noqa

from ._functions import SpectralPooling2dFunction

_default_reduction = 'mean'
_epsilon = 1e-7


# Layers


def spectral_pool2d(x, scale_factor):
    if isinstance(scale_factor, collections.Iterable):
        scale_h, scale_w = scale_factor
    else:
        scale_h = scale_w = scale_factor
    h, w = math.ceil(x.size(-2) * scale_h), math.ceil(x.size(-1) * scale_w)
    return SpectralPooling2dFunction.apply(x, h, w)


# Losses


def _apply_reduction(tensor, reduction):
    if reduction is None:
        return tensor
    elif reduction == 'mean':
        return tensor.mean()
    elif reduction == 'sum':
        return tensor.sum()
    raise ValueError("Reduction expected to be None, 'mean', or 'sum', got '%s'" % reduction)


def _dice_loss(mul, add, nd, reduction=_default_reduction, epsilon=_epsilon):
    intersection = mul.sum(dim=tuple(range(-nd, 0, -1))) + epsilon
    union = add.sum(dim=tuple(range(-nd, 0, -1))) + epsilon * 2
    loss = 1. - (2. * intersection / union)
    return _apply_reduction(loss, reduction)


def _iou_loss(mul, add, nd, reduction=_default_reduction, epsilon=_epsilon):
    intersection = mul.sum(dim=tuple(range(-nd, 0, -1))) + epsilon
    union = (add - mul).sum(dim=tuple(range(-nd, 0, -1))) + epsilon
    loss = 1. - (intersection / union)
    return _apply_reduction(loss, reduction)


def dice_loss_nd(output, target, nd, reduction=_default_reduction):
    return _dice_loss(output * target, output + target, nd=nd, reduction=reduction)


def iou_loss_nd(output, target, nd, reduction=_default_reduction):
    return _iou_loss(output * target, output + target, nd=nd, reduction=reduction)


# Utils


def convert_by_one_hot_nd(tensor, nd):
    index = tensor.argmax(dim=-nd - 1).long()
    return torch.zeros_like(tensor).scatter(
        dim=-nd - 1, index=index.unsqueeze(dim=-nd - 1), src=torch.ones_like(tensor)).to(tensor.dtype)


def one_hot_nd(tensor, n_classes, nd):  # N H W
    new_shape = list(range(tensor.ndim))
    new_shape.insert(-nd, tensor.ndim)
    return _F.one_hot(tensor.long(), n_classes).permute(new_shape)  # N C H W


def __getattr__(name):
    from torch.nn import functional
    return getattr(functional, name)
