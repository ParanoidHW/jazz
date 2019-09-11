from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import collections
import numbers
import numpy as np

from core.tensor import Zhangliang
from core.lr_schedule import FixedLRSchedule, AbstractLRSchedule


class AbstractOptimizer(object):
    def __init__(self, lr, params, regularize=None, weight_decay=0):
        if np.isscalar(lr):
            self.lr_schedule = FixedLRSchedule(lr)
        elif isinstance(lr, AbstractLRSchedule):
            self.lr_schedule = lr
        else:
            raise TypeError('`lr` should be a scalar or a `LRSchedule` class.')
        self.params = list(params)
        self.iter = 0
        self.regularize = regularize
        self.weight_decay = weight_decay

    def update(self):
        raise NotImplementedError

    def step(self):
        self.lr_schedule.step(self.iter)

    def state_dict(self):
        pass

    def clear_grad(self):
        for p in self.params:
            if p.requires_grad:
                p.release()


class SGD(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, momentum=.9, regularize=None, weight_decay=1e-3):
        super(SGD, self).__init__(lr, params, regularize, weight_decay)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.values) for p in params]

    def update(self):
        cur_lr = self.step()

        for index, p in enumerate(self.params):
            self.velocities[index] = self.momentum * self.velocities[index] + cur_lr * p.grad
            p.values += self.velocities[index]


class NesterovSGD(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, momentum=.9, regularize=None, weight_decay=1e-3):
        super(NesterovSGD, self).__init__(lr, params, regularize, weight_decay)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.values) for p in params]
        self.buf_weights = [np.array(p.values) for p in params]

    def update(self):
        cur_lr = self.step()

        for index, p in enumerate(self.params):
            self.velocities[index] = self.momentum * self.velocities[index] + cur_lr * p.grad
            self.buf_weights[index] += self.velocities[index]
            p.values = self.buf_weights[index] + self.momentum * self.velocities[index]


class Rprop(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, scales=(1.2, 0.5), lr_range=(1e-6, 50),
                 regularize=None, weight_decay=1e-3, eta0=0.1):
        super(Rprop, self).__init__(lr, params, regularize, weight_decay)
        self.alpha, self.beta = scales
        self.eta_min, self.eta_max = lr_range

        self.velocities = [np.zeros_like(p.values) for p in params]
        self.buf_grad = [np.zeros_like(p.values) for p in params]
        self.eta = [np.ones_like(p.values) * eta0 for p in params]

    def update(self):
        #TODO: there is something wrong.
        cur_lr = self.lr_schedule.step(self.iter)

        for index, p in enumerate(self.params):
            pos = (p.grad * self.buf_grad[index]) > 0
            neg = (p.grad * self.buf_grad[index]) < 0
            self.eta[index][pos] = np.min(self.eta[index][pos] * self.alpha, self.eta_max)
            self.eta[index][neg] = np.max(self.eta[index][neg] * self.beta, self.eta_min)

            self.velocities[index] = np.sign(p.grad) * self.eta[index]
            p.values += self.velocities[index]

