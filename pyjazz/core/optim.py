from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
from collections import OrderedDict

from pyjazz.core.lr_schedule import FixedLRSchedule, AbstractLRSchedule
from pyjazz.core.tensor import Parameters

class AbstractOptimizer(object):
    def __init__(self, lr, params, regularize=None, weight_decay=0):
        if np.isscalar(lr):
            self.lr_schedule = FixedLRSchedule(lr)
        elif isinstance(lr, AbstractLRSchedule):
            self.lr_schedule = lr
        else:
            raise TypeError('`lr` should be a scalar or a `LRSchedule` class.')

        if isinstance(params, (dict, OrderedDict)):
            self.params = [v for k, v in params.items()]
        elif isinstance(params, (list, tuple)):
            self.params = list(params)
        elif isinstance(params, Parameters):
            self.params = [params]
        else:
            raise TypeError
        self.iter = 0
        self.regularize = regularize
        self.weight_decay = weight_decay

    def update(self):
        raise NotImplementedError

    def step(self):
        return self.lr_schedule.step(self.iter)

    def state_dict(self):
        pass

    def clear_grad(self):
        for p in self.params:
            if p.requires_grad:
                p.release()


class SGD(AbstractOptimizer):
    """
        \Delta w_{t+1}=\rho\Delta w_{t}-\sum_{i=1}^B\nabla_w L(x_i, y_i)-\lambda w_t
        w_{t+1} =w_t + \eta\Delta w_{t+1}
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, regularize=None, weight_decay=0):
        super(SGD, self).__init__(lr, params, regularize, weight_decay)
        self.momentum = momentum
        if isinstance(params, (dict, OrderedDict)):
            self.velocities = [np.zeros_like(v.values) for k, v in params.items()]
        elif isinstance(params, (list, tuple)):
            self.velocities = [np.zeros_like(v.values) for v in params]
        elif isinstance(params, Parameters):
            self.velocities = [np.zeros_like(params.values)]
        else:
            raise TypeError

    def update(self):
        cur_lr = self.step()

        for index, p in enumerate(self.params):
            dw = p.grad
            if dw is not None:
                # Some node may not in the computation graph, e.g., defined modules but not ever used.
                if self.iter == 0:
                    self.velocities[index] = - dw
                else:
                    self.velocities[index] = self.momentum * self.velocities[index] - dw - self.weight_decay * p.values
                p.add_(cur_lr * self.velocities[index])


class NesterovSGD(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, momentum=.9, regularize=None, weight_decay=0):
        super(NesterovSGD, self).__init__(lr, params, regularize, weight_decay)
        self.momentum = momentum
        if isinstance(params, (dict, OrderedDict)):
            self.velocities = [np.zeros_like(v.values) for k, v in params.items()]
            self.buf_weights = [np.array(v.values) for k, v in params.items()]
        elif isinstance(params, (list, tuple)):
            self.velocities = [np.zeros_like(v.values) for v in params]
            self.buf_weights = [np.array(p.values) for p in params]
        elif isinstance(params, Parameters):
            self.velocities = [np.zeros_like(params.values)]
            self.buf_weights = [np.array(params.values)]
        else:
            raise TypeError

    def update(self):
        cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
            if dw is not None:
                # Some node may not in the computation graph, e.g., defined modules but not ever used.
                if self.iter == 0:
                    self.velocities[index] = - dw
                else:
                    self.velocities[index] = self.momentum * self.velocities[index] - dw - self.weight_decay * p.values
                p.add_(cur_lr * (self.velocities[index] + self.momentum * self.velocities[index]))


class Rprop(AbstractOptimizer):
    def __init__(self, params, lr=1e-2, scales=(1.2, 0.5), lr_range=(1e-6, 50.),
                 regularize=None, weight_decay=0):
        super(Rprop, self).__init__(lr, params, regularize, weight_decay)
        self.alpha, self.beta = scales
        self.eta_min, self.eta_max = lr_range

        if isinstance(params, (dict, OrderedDict)):
            self.buf_grad = [np.zeros_like(v.values) for k, v in params.items()]
            self.eta = [np.ones_like(v.values) for k, v in params.items()]
        elif isinstance(params, (list, tuple)):
            self.buf_grad = [np.zeros_like(v.values) for v in params]
            self.eta = [np.ones_like(p.values) for p in params]
        elif isinstance(params, Parameters):
            self.buf_grad = [np.zeros_like(params.values)]
            self.eta = [np.ones_like(params.values)]
        else:
            raise TypeError

    def update(self):
        # LR schedule won't take effect in Rprop, except for initialization
        # cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
            if dw is not None:
                # Some node may not in the computation graph, e.g., defined modules but not ever used.
                pos = (dw * self.buf_grad[index]) > 0
                neg = (dw * self.buf_grad[index]) < 0
                self.eta[index][pos] = self.eta[index][pos] * self.alpha
                self.eta[index][neg] = self.eta[index][neg] * self.beta

                self.eta[index] = np.clip(self.eta[index], a_min=self.eta_min, a_max=self.eta_max)

                dw[neg] = 0
                p.add_(- self.eta[index] * (np.sign(dw) + self.weight_decay * p.values))
                self.buf_grad[index] = dw


class RMSprop(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, momentum=0, regularize=None, weight_decay=0):
        super(RMSprop, self).__init__(lr, params, regularize, weight_decay)
        self.momentum = momentum
        self.alpha = alpha
        self.eps = eps
        if isinstance(params, (dict, OrderedDict)):
            self.velocities = [np.zeros_like(v.values) for k, v in params.items()]
            self.square_sum = [np.zeros_like(v.values) for k, v in params.items()]
        elif isinstance(params, (list, tuple)):
            self.velocities = [np.zeros_like(v.values) for v in params]
            self.square_sum = [np.zeros_like(p.values) for p in params]
        elif isinstance(params, Parameters):
            self.velocities = [np.zeros_like(params.values)]
            self.square_sum = [np.zeros_like(params.values)]
        else:
            raise TypeError

    def update(self):
        # TODO: implement centered RMSprop
        cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
            if dw is not None:
                # Some node may not in the computation graph, e.g., defined modules but not ever used.
                if self.iter == 0:
                    self.square_sum[index] = np.square(dw)
                else:
                    self.square_sum[index] = self.alpha * self.square_sum[index] + (1. - self.alpha) * np.square(dw)

                rsquare_sum = np.sqrt(self.square_sum[index]) + self.eps
                vel = - dw / rsquare_sum - self.weight_decay * p.values

                p.add_(cur_lr * vel)


class AdaGrad(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, regularize=None, weight_decay=0):
        super(AdaGrad, self).__init__(lr, params, regularize, weight_decay)
        self.eps = eps
        if isinstance(params, (dict, OrderedDict)):
            self.velocities = [np.zeros_like(v.values) for k, v in params.items()]
            self.square_sum = [np.zeros_like(v.values) for k, v in params.items()]
        elif isinstance(params, (list, tuple)):
            self.velocities = [np.zeros_like(v.values) for v in params]
            self.square_sum = [np.zeros_like(p.values) for p in params]
        elif isinstance(params, Parameters):
            self.velocities = [np.zeros_like(params.values)]
            self.square_sum = [np.zeros_like(params.values)]
        else:
            raise TypeError

    def update(self):
        cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
            if dw is not None:
                # Some node may not in the computation graph, e.g., defined modules but not ever used.
                self.square_sum[index] += np.square(dw)
                rsquare_sum = np.sqrt(self.square_sum[index]) + self.eps
                grad = - dw / rsquare_sum - self.weight_decay * p.values
                p.add_(cur_lr * grad)


class AdaDelta(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, alpha=0.9, eps=1e-8, regularize=None, weight_decay=0):
        super(AdaDelta, self).__init__(lr, params, regularize, weight_decay)

        self.alpha = alpha
        self.eps = eps

        if isinstance(params, (dict, OrderedDict)):
            self.square_grad_sum = [np.zeros_like(v.values) for k, v in params.items()]
            self.square_vel_sum = [np.zeros_like(v.values) for k, v in params.items()]
            self.velocities = [np.zeros_like(v.values) for k, v in params.items()]
        elif isinstance(params, (list, tuple)):
            self.square_grad_sum = [np.zeros_like(v.values) for v in params]
            self.square_vel_sum = [np.zeros_like(p.values) for p in params]
            self.velocities = [np.zeros_like(p.values) for p in params]
        elif isinstance(params, Parameters):
            self.square_grad_sum = [np.zeros_like(params.values)]
            self.square_vel_sum = [np.zeros_like(params.values)]
            self.velocities = [np.zeros_like(params.values)]
        else:
            raise TypeError

    def update(self):
        cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
            if dw is not None:
                # Some node may not in the computation graph, e.g., defined modules but not ever used.
                v = self.velocities[index]
                self.square_grad_sum[index] = self.alpha * self.square_grad_sum[index] + (1 - self.alpha) * np.square(dw)
                rms_g = np.sqrt(self.square_grad_sum[index]) + self.eps
                # RMS[v] is one lag behind RMS[g]
                rms_v = np.sqrt(self.square_vel_sum[index])
                self.velocities[index] = - rms_v / rms_g * dw - self.weight_decay * p.values
                p.add_(cur_lr * self.velocities[index])
                self.square_vel_sum[index] = self.alpha * self.square_vel_sum[index] + (1 - self.alpha) * np.square(v)


class Adam(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, regularize=None, weight_decay=0):
        super(Adam, self).__init__(lr, params, regularize, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.moment1 = [np.zeros_like(p.values) for p in params]
        self.moment2 = [np.zeros_like(p.values) for p in params]
        self.beta2_root = np.sqrt(1 - self.beta2)

    def update(self):
        cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
            if dw is not None:
                # Some node may not in the computation graph, e.g., defined modules but not ever used.
                m = self.moment1[index]
                v = self.moment2[index]

                if self.iter == 0:
                    m = dw
                    v = np.square(dw)
                else:
                    m = self.beta1 * m + (1 - self.beta1) * dw
                    v = self.beta2 * v + (1 - self.beta2) * np.square(dw)
                v_root = np.sqrt(v) + self.eps
                grad = - self.beta2_root / (1 - self.beta1) * m / v_root - self.weight_decay * p.values

                # TODO: add bias correction

                p.add_(cur_lr * grad)

                self.moment1[index] = m
                self.moment2[index] = v


class AdaMax(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, regularize=None, weight_decay=0):
        super(AdaMax, self).__init__(lr, params, regularize, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.moment1 = [np.zeros_like(p.values) for p in params]
        self.moment2 = [np.zeros_like(p.values) for p in params]
        if isinstance(params, (dict, OrderedDict)):
            self.moment1 = [np.zeros_like(v.values) for k, v in params.items()]
            self.moment2 = [np.zeros_like(v.values) for k, v in params.items()]
        elif isinstance(params, (list, tuple)):
            self.moment1 = [np.zeros_like(v.values) for v in params]
            self.moment2 = [np.zeros_like(p.values) for p in params]
        elif isinstance(params, Parameters):
            self.moment1 = [np.zeros_like(params.values)]
            self.moment2 = [np.zeros_like(params.values)]
        else:
            raise TypeError

    def update(self):
        cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
            if dw is not None:
                # Some node may not in the computation graph, e.g., defined modules but not ever used.
                m = self.moment1[index]
                v = self.moment2[index]

                if self.iter == 0:
                    m = dw
                    v = np.abs(dw)
                else:
                    m = self.beta1 * m + (1 - self.beta1) * dw
                    v = np.maximum(self.beta2 * v, np.abs(dw))
                grad = - 1 / (1 - self.beta1) * m / v - self.weight_decay * p.values

                # TODO: add bias correction

                p.add_(cur_lr * grad)

                self.moment1[index] = m
                self.moment2[index] = v
