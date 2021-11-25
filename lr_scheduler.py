from utils.lr_scheduler import *

import warnings
warnings.warn("This module is deprecated. Use utils.lr_scheduler instead.")
del warnings

__all__ = [k for k in globals().keys() if not k.startswith('_')]
