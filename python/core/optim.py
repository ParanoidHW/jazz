from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np

from python.core.lr_schedule import FixedLRSchedule, AbstractLRSchedule


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
        self.velocities = [np.zeros_like(p.values) for p in params]

    def update(self):
        cur_lr = self.step()

        for index, p in enumerate(self.params):
            dw = p.grad
            if self.iter == 0:
                self.velocities[index] = - dw
            else:
                self.velocities[index] = self.momentum * self.velocities[index] - dw - self.weight_decay * p.values
            p.values += cur_lr * self.velocities[index]


class NesterovSGD(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, momentum=.9, regularize=None, weight_decay=0):
        super(NesterovSGD, self).__init__(lr, params, regularize, weight_decay)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.values) for p in params]
        self.buf_weights = [np.array(p.values) for p in params]

    def update(self):
        cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
            if self.iter == 0:
                self.velocities[index] = - dw
            else:
                self.velocities[index] = self.momentum * self.velocities[index] - dw - self.weight_decay * p.values
            p.values += cur_lr * (self.velocities[index] + self.momentum * self.velocities[index])


class Rprop(AbstractOptimizer):
    def __init__(self, params, lr=1e-2, scales=(1.2, 0.5), lr_range=(1e-6, 50.),
                 regularize=None, weight_decay=0):
        super(Rprop, self).__init__(lr, params, regularize, weight_decay)
        self.alpha, self.beta = scales
        self.eta_min, self.eta_max = lr_range

        self.buf_grad = [np.zeros_like(p.values) for p in params]
        self.eta = [np.ones_like(p.values) * lr for p in params]

    def update(self):
        # LR schedule won't take effect in Rprop, except for initialization
        # cur_lr = self.step()
        for index, p in enumerate(self.params):
            pos = (p.grad * self.buf_grad[index]) > 0
            neg = (p.grad * self.buf_grad[index]) < 0
            self.eta[index][pos] = self.eta[index][pos] * self.alpha
            self.eta[index][neg] = self.eta[index][neg] * self.beta

            self.eta[index] = np.clip(self.eta[index], a_min=self.eta_min, a_max=self.eta_max)

            dw = np.array(p.grad)
            dw[neg] = 0
            p.values += - self.eta[index] * (np.sign(dw) + self.weight_decay * p.values)
            self.buf_grad[index] = dw


class RMSprop(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, momentum=0, regularize=None, weight_decay=0):
        super(RMSprop, self).__init__(lr, params, regularize, weight_decay)
        self.momentum = momentum
        self.alpha = alpha
        self.eps = eps
        self.velocities = [np.zeros_like(p.values) for p in params]
        self.square_sum = [np.zeros_like(p.values) for p in params]

    def update(self):
        # TODO: implement centered RMSprop
        cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
            if self.iter == 0:
                self.square_sum[index] = np.square(dw)
            else:
                self.square_sum[index] = self.alpha * self.square_sum[index] + (1. - self.alpha) * np.square(dw)

            rsquare_sum = np.sqrt(self.square_sum[index]) + self.eps
            vel = - dw / rsquare_sum - self.weight_decay * p.values

            p.values += cur_lr * vel


class AdaGrad(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, regularize=None, weight_decay=0):
        super(AdaGrad, self).__init__(lr, params, regularize, weight_decay)
        self.square_sum = [np.zeros_like(p.values) for p in params]
        self.eps = eps

    def update(self):
        cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
            self.square_sum[index] += np.square(dw)
            rsquare_sum = np.sqrt(self.square_sum[index]) + self.eps
            grad = - dw / rsquare_sum - self.weight_decay * p.values
            p.values += cur_lr * grad


class AdaDelta(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, alpha=0.9, eps=1e-8, regularize=None, weight_decay=0):
        super(AdaDelta, self).__init__(lr, params, regularize, weight_decay)
        self.square_grad_sum = [np.zeros_like(p.values) for p in self.params]
        self.square_vel_sum = [np.zeros_like(p.values) for p in self.params]
        self.velocities = [np.zeros_like(p.values) for p in self.params]

        self.alpha = alpha
        self.eps = eps

    def update(self):
        cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
            v = self.velocities[index]
            self.square_grad_sum[index] = self.alpha * self.square_grad_sum[index] + (1 - self.alpha) * np.square(dw)
            rms_g = np.sqrt(self.square_grad_sum[index]) + self.eps
            # RMS[v] is one lag behind RMS[g]
            rms_v = np.sqrt(self.square_vel_sum[index])
            self.velocities[index] = - rms_v / rms_g * dw - self.weight_decay * p.values
            p.values += cur_lr * self.velocities[index]
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

            p.values += cur_lr * grad

            self.moment1[index] = m
            self.moment2[index] = v


class AdaMax(AbstractOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, regularize=None, weight_decay=0):
        super(AdaMax, self).__init__(lr, params, regularize, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.moment1 = [np.zeros_like(p.values) for p in params]
        self.moment2 = [np.zeros_like(p.values) for p in params]

    def update(self):
        cur_lr = self.step()
        for index, p in enumerate(self.params):
            dw = p.grad
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

            p.values += cur_lr * grad

            self.moment1[index] = m
            self.moment2[index] = v
