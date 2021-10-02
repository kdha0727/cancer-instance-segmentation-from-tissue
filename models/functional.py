import collections
import math
import torch
import torch.nn.functional as _F  # noqa

_default_reduction = 'mean'
_epsilon = 1e-7


# Layers


def _spectral_crop(x, h, w):

    cutoff_freq_h = math.ceil(h / 2)
    cutoff_freq_w = math.ceil(w / 2)

    if h % 2 == 1:
        if w % 2 == 1:
            top_left = x[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = x[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            bottom_left = x[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            bottom_right = x[:, -(cutoff_freq_h - 1):, -(cutoff_freq_w - 1):]
        else:
            top_left = x[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = x[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = x[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            bottom_right = x[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:]
    else:
        if w % 2 == 1:
            top_left = x[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = x[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            bottom_left = x[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = x[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):]
        else:
            top_left = x[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = x[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = x[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = x[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

    top_combined = torch.cat((top_left, top_right), dim=-2)
    bottom_combined = torch.cat((bottom_left, bottom_right), dim=-2)
    all_together = torch.cat((top_combined, bottom_combined), dim=-1)
    return all_together


def _spectral_pad(x, output, h, w):

    cutoff_freq_h = math.ceil(h / 2)
    cutoff_freq_w = math.ceil(w / 2)

    pad = torch.zeros_like(x)

    if h % 2 == 1:
        if w % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            pad[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h - 1):, -(cutoff_freq_w - 1):] = output[:, :, -(cutoff_freq_h - 1):,
                                                                             -(cutoff_freq_w - 1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:] = output[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:]
    else:
        if w % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):] = output[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -cutoff_freq_w:] = output[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

    return pad


def discrete_hartley_transform(x, *args, **kwargs):  # original signature: (x)
    return torch.fft.rfftn(x, *args, **kwargs)


def i_discrete_hartley_transform(x, size, *args, **kwargs):  # original signature: (x, size)
    return torch.fft.irfftn(x, size, *args, **kwargs)


class SpectralPoolingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, height, width):  # noqa
        ctx.oh = height
        ctx.ow = width
        ctx.save_for_backward(x)
        # Hartley transform by discrete Fourier transform
        dht = discrete_hartley_transform(x)
        # frequency cropping
        all_together = _spectral_crop(dht, height, width)
        # inverse Hartley transform
        dht = i_discrete_hartley_transform(all_together, all_together.size())
        return dht

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        x, = ctx.saved_tensors
        # Hartley transform by discrete Fourier transform
        dht = discrete_hartley_transform(grad_output)
        # frequency padding
        grad_input = _spectral_pad(x, dht, ctx.oh, ctx.ow)
        # inverse Hartley transform
        grad_input = i_discrete_hartley_transform(grad_input, grad_input.size())
        return grad_input, None, None


def spectral_pool2d(x, scale_factor):
    if isinstance(scale_factor, collections.Iterable):
        scale_h, scale_w = scale_factor
    else:
        scale_h = scale_w = scale_factor
    h, w = math.ceil(x.size(-2) * scale_h), math.ceil(x.size(-1) * scale_w)
    return SpectralPoolingFunction.apply(x, h, w)


# Losses


def _assert_proper_fraction(*tensor, raise_exc=True):
    for t in tensor:
        if not (t.ge(0.).all() and t.le(1.).all()):
            if raise_exc:
                raise TypeError("Assertion failed: 0 <= tensor <= 1")
            return False
    return True


def _apply_reduction(tensor, reduction):
    if reduction is None:
        return tensor
    elif reduction == 'mean':
        return tensor.mean()
    elif reduction == 'sum':
        return tensor.sum()
    raise ValueError("Reduction expected to be None, 'mean', or 'sum', got '%s'" % reduction)


def _dice_loss(mul, add, nd, epsilon=_epsilon):
    intersection = mul.sum(dim=tuple(range(-nd, 0, -1))) + epsilon
    union = add.sum(dim=tuple(range(-nd, 0, -1))) + epsilon * 2
    loss = 1. - (2. * intersection / union)
    return loss


def _iou_loss(mul, add, nd, epsilon=_epsilon):
    intersection = mul.sum(dim=tuple(range(-nd, 0, -1))) + epsilon
    union = (add - mul).sum(dim=tuple(range(-nd, 0, -1))) + epsilon
    loss = 1. - (intersection / union)
    return loss


def dice_loss_nd(output, target, nd, reduction=_default_reduction):
    _assert_proper_fraction(output, target)
    return _apply_reduction(_dice_loss(output * target, output + target, nd=nd), reduction)


def iou_loss_nd(output, target, nd, reduction=_default_reduction):
    _assert_proper_fraction(output, target)
    return _apply_reduction(_iou_loss(output * target, output + target, nd=nd), reduction)


def dice_loss_2d(output, target, reduction=_default_reduction):
    return dice_loss_nd(output, target, nd=2, reduction=reduction)


def iou_loss_2d(output, target, reduction=_default_reduction):
    return iou_loss_nd(output, target, nd=2, reduction=reduction)


# Utils


def convert_by_one_hot_nd(tensor, nd):
    index = tensor.argmax(dim=-nd - 1).long()
    return torch.zeros_like(tensor).scatter(
        dim=-nd - 1, index=index.unsqueeze(dim=-nd - 1), src=torch.ones_like(tensor)).to(tensor.dtype)


def one_hot_nd(tensor, n_classes, nd):  # N H W
    new_shape = list(range(tensor.ndim))
    new_shape.insert(-nd, tensor.ndim)
    return _F.one_hot(tensor.long(), n_classes).permute(new_shape)  # N C H W


def convert_by_one_hot_2d(tensor):
    return convert_by_one_hot_nd(tensor, nd=2)


def one_hot_2d(tensor, n_classes):
    return one_hot_nd(tensor, n_classes, nd=2)


def __getattr__(name):
    return getattr(_F, name)
