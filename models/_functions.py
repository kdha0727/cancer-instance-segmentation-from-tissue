import math
import torch

# Layers


def _spectral_crop_2d(x, h, w):

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


def _spectral_pad_2d(x, output, h, w):

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


class SpectralPooling2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, height, width):  # noqa
        ctx.oh = height
        ctx.ow = width
        ctx.save_for_backward(x)
        # Hartley transform by discrete Fourier transform
        dht = torch.fft.rfftn(x)
        # frequency cropping
        all_together = _spectral_crop_2d(dht, height, width)
        # inverse Hartley transform
        dht = torch.fft.irfftn(all_together, all_together.size())
        return dht

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        x, = ctx.saved_tensors
        # Hartley transform by discrete Fourier transform
        dht = torch.fft.rfftn(grad_output)
        # frequency padding
        grad_input = _spectral_pad_2d(x, dht, ctx.oh, ctx.ow)
        # inverse Hartley transform
        grad_input = torch.fft.irfftn(grad_input, grad_input.size())
        return grad_input, None, None
