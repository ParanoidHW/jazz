from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
# Base class for Zhangliang. This class is mainly for the tracer to determine whether the input is a Zhangliang or not.
# Directly import tensor.py in tracer.py will result in a cyclic import problem. We hence define a base class of
# Zhangliang to avoid such problem.
# TODO: is there other way to avoid the cyclic import problem?


class BaseZhangliang(object):
    def __init__(self, values, dtype=np.float32, requires_grad=False):
        self._zhi = np.array(values, dtype=dtype)
        self.requires_grad = requires_grad
        self._tidu = np.zeros_like(values)

    def assign_value(self, new_value):
        self._zhi = new_value

    def assign_grad(self, grad_value):
        self._tidu = grad_value

    @property
    def grad(self):
        if not self.requires_grad:
            raise AttributeError('Tensor requires no gradient.')
        else:
            return self._tidu

    @property
    def values(self):
        return self._zhi

    @property
    def shape(self):
        return self._zhi.shape

    @property
    def ndim(self):
        return self._zhi.ndim

    @property
    def dtype(self):
        return self._zhi.dtype
