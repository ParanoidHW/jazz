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
        self._buffer = OrderedDict()
        self._training = True

    @property
    def is_training(self):
        return self._training

    def train(self, flag=True):
        self._training = flag
        for _, sub_m in self._sub_modules.items():
            sub_m.train(flag)

    def register_buffer(self, name, value):
        if name is None or value is None:
            raise ValueError
        elif isinstance(value, core.Parameters) or not isinstance(value, core.Zhangliang):
            raise TypeError

        self._buffer[name] = value

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
        # elif isinstance(value, core.Zhangliang):
        #     self._buffer[key] = value
        elif isinstance(value, Module):
            self._sub_modules[key] = value
            submodule_params = value.parameters
            for k, v in submodule_params.items():
                self._parameters['{}.{}'.format(key, k)] = v


class Stacked(Module):
    def __init__(self, *args):
        super(Stacked, self).__init__()
        layer_count = {}
        for layer in args:
            if not isinstance(layer, Module):
                raise TypeError
            layer_type = type(layer)
            layer_num = layer_count.setdefault(layer_type, 0)
            self._sub_modules[f'{layer_type}{layer_num}'] = layer
            layer_count[layer_type] += 1

    def forward(self, *args, **kwargs):
        for layer_id, (_, layer) in enumerate(self._sub_modules.items()):
            if layer_id == 0:
                result = layer(*args, **kwargs)
            else:
                result = layer(result)
        return result


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


class BatchNorm2d(Module):
    def __init__(self, in_feat, momentum=0.1, affine=True, tracking_running_stat=True, eps=1e-5):
        super(BatchNorm2d, self).__init__()
        self.running_average = core.Zhangliang.zeros((1, in_feat, 1, 1), requires_grad=False)
        self.running_var = core.Zhangliang.ones((1, in_feat, 1, 1), requires_grad=False)

        self.register_buffer('mean', self.running_average)
        self.register_buffer('var', self.running_var)

        if affine:
            self.gamma = core.Parameters.ones((1, in_feat, 1, 1))
            self.beta = core.Parameters.zeros((1, in_feat, 1, 1))
        else:
            self.gamma = 1.
            self.beta = 0.
        self.momentum = momentum
        self.tracking_running_stat = tracking_running_stat
        self.eps = eps

    def forward(self, x):
        track_stat = self.is_training and self.tracking_running_stat
        y = core.batch_norm(x, self.gamma, self.beta, self.running_average, self.running_var,
                            track_stat, self.momentum, self.eps)
        return y

