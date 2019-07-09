from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

# Base class for Zhangliang. This class is mainly for the tracer to determine whether the input is a Zhangliang or not.
# Directly import tensor.py in tracer.py will result in a cyclic import problem. We hence define a base class of
# Zhangliang to avoid such problem.
# TODO: is there other way to avoid the cyclic import problem?


class BaseZhangliang(object):
    def __init__(self, values, dtype=np.float32, name='', requires_grad=False):
        self.zhi = np.array(values, dtype=dtype)
        self.name = name
        self.requires_grad = requires_grad
        self.tidu = np.zeros_like(values)

    def assign(self, new_value):
        self.zhi = new_value
        return self

    @property
    def grad(self):
        if not self.requires_grad:
            raise AttributeError('Tensor requires no gradient.')
        else:
            return self.tidu

    @property
    def shape(self):
        return self.zhi.shape

    @property
    def dtype(self):
        return self.zhi.dtype

    @classmethod
    def from_array(cls, values, dtype=np.float32):
        return cls(values, dtype=dtype)

    @classmethod
    def zeros(cls, shape, dtype=np.float32):
        zeros_ = np.zeros(shape, dtype=dtype)
        return cls(zeros_)

    @classmethod
    def ones(cls, shape, dtype=np.float32):
        ones_ = np.ones(shape, dtype=dtype)
        return cls(ones_)
