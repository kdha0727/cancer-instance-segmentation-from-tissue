import torch
from torch.nn.modules.conv import _ConvNd  # noqa
from torch.nn.modules.linear import Linear
from torch.nn.modules.batchnorm import _NormBase  # noqa


def weights_init(init_type='xavier'):
    def init(m):
        import math
        from torch.nn import init
        if isinstance(m, (_ConvNd, Linear)):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                raise TypeError("Unsupported initialization: {}".format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                with torch.no_grad():
                    m.bias.data.zero_()
        elif isinstance(m, _NormBase):
            m.reset_parameters()
    return init
