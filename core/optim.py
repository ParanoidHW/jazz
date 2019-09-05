from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import collections
import numbers
import numpy as np

from core.tensor import Zhangliang
from utils.tracer import graph


class AbstractOptimizer(object):
    def __init__(self, params):
        self.params = params

    def apply(self):
        raise NotImplementedError

    def state_dict(self):
        pass

    def clear_grad(self):
        for p in self.params:
            if p.requires_grad:
                p.release()


class SGDOptimizer(AbstractOptimizer):
    pass
