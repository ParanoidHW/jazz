from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict


class Module(object):
    def __init__(self, namespace, *args, **kwargs):
        self._parameters = OrderedDict()
        self._sub_modules = OrderedDict()
        self._name_scope = namespace

    @property
    def parameters(self):
        return self._parameters

    def forward_propagate(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward_propagate(*args, **kwargs)

    def backward_propagate(self, grad, *args, **kwargs):
        raise NotImplementedError
