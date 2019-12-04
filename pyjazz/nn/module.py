from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
from .. import core
from ..utils.initializer import xavier_initializer, he_initializer


class Module(object):
    def __init__(self, *args, **kwargs):
        self._parameters = OrderedDict()
        self._sub_modules = OrderedDict()

    @property
    def parameters(self):
        return self._parameters

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, key, value):
        # First, add the item to self.__dict__
        object.__setattr__(self, key, value)

        # Check whether it's a sub-module or a parameter
        if isinstance(value, core.Parameters):
            self._parameters[key] = value
        elif isinstance(value, Module):
            self._sub_modules[key] = value
            submodule_params = value.parameters
            for k, v in submodule_params.items():
                self._parameters['{}.{}'.format(key, k)] = v

    # def backward(self, grad, *args, **kwargs):
    #     raise NotImplementedError


class Linear(Module):
    def __init__(self, in_feat, out_feat, bias=True, initializer=he_initializer):
        super(Linear, self).__init__()
        self.weight = core.Parameters.randn((in_feat, out_feat))
        initializer(self.weight)

        self.apply_bias = bias
        if bias:
            self.bias = core.Parameters.zeros((1, out_feat))

    def forward(self, x):
        y = core.linear(x, self.weight)
        if self.apply_bias:
            y = y + self.bias
        return y


class Conv2d(Module):
    def __init__(self, in_feat, out_feat, k, s=1, p=1, d=1, bias=True, initializer=he_initializer):
        super(Conv2d, self).__init__()
        if isinstance(k, (list, tuple)):
            kh, kw = k
        else:
            kh, kw = k, k
        self.weight = core.Parameters.randn((out_feat, in_feat, kh, kw))
        initializer(self.weight)

        self.bias = None
        if bias:
            self.bias = core.Parameters.zeros((out_feat, ))

        self._stride = s
        self._padding = p
        self._dilation = d

    def forward(self, x):
        y = core.conv2d(x, self.weight, self.bias, self._stride, self._padding, self._dilation)
        return y


class ReLU(Module):
    def __init__(self, inplace=False):
        # TODO: implement inplace calculation
        super(ReLU, self).__init__()
        self._inplace = inplace

    def forward(self, x):
        return core.relu(x)


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return core.sigmoid(x)


class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x, dim=-1):
        return core.softmax(x, dim=dim)
