from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import math
from pyjazz.core import Zhangliang, no_grad


# The following codes borrow much from the pytorch source code.
# See https://pytorch.org/docs/stable/nn.init.html
def _calculate_fan_in_and_fan_out(zl):
    if zl.ndim == 2:
        fan_in, fan_out = zl.shape
    else:
        num_feat_out, num_feat_in = zl.shape[:2]
        receptive_field_size = zl[0][0].size
        fan_in = num_feat_in * receptive_field_size
        fan_out = num_feat_out * receptive_field_size
    return fan_in, fan_out


def xavier_initializer(zl, gain=1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(zl)
    std = gain * math.sqrt(2. / (fan_in + fan_out))
    with no_grad():
        return zl.normal_(0., std)


def he_initializer(zl, gain=1, a=0, mode='fan_in'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(zl)
    if mode == 'fan_in':
        mag = fan_in
    elif mode == 'fan_out':
        mag = fan_out
    else:
        raise ValueError
    std = gain * math.sqrt(2. / ((1. + a*a) * mag))
    with no_grad():
        return zl.normal_(0., std)
