from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from itertools import product
import numpy as np
from core import Zhangliang
from utils.register import func_lib, grad_lib
from utils.misc import multiplicative_broadcast_analysis, additive_broadcast_analysis


TOL = 1e-6
RTOL = 1e-6
EPS = 1e-6


def arg_space():
    scalar = 2.0
    vector = np.random.randn(4)
    mat = np.random.randn(3, 4)
    mat2 = np.random.randn(1, 4)
    allargs = [scalar, vector, mat, mat2]
    for arg1, arg2 in product(allargs, allargs):
        yield arg1, arg2


def array_close(a, b):
    array_flag = np.logical_or(np.abs(a - b) < TOL, np.abs(a - b) / np.abs(a + b) < RTOL)
    return np.all(array_flag)


class VarianceSpace(object):
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    def __len__(self):
        return np.size(self.data)

    def __getitem__(self, index):
        variance = np.random.randn(self.data.shape)
        return variance


def approx_numeric_grad(z_p, z_m):
    return (z_p - z_m) / EPS


def check_element_binary_numeric_grad(op_name, a, b, *args, **kwargs):
    fn = func_lib[op_name]
    gn = grad_lib[op_name]
    x = Zhangliang(a, requires_grad=True)
    y = Zhangliang(b, requires_grad=True)
    z = fn(x, y, *args, **kwargs)
    z.assign_grad(np.ones_like(z))
    gn(z, x, y, *args, **kwargs)
    axes_to_reduce = additive_broadcast_analysis([x.shape, y.shape], z.shape)

    for i in range(10):
        x_var = np.random.rand(*(x.shape)) * EPS/2
        x_var = np.where(x_var, x_var, EPS/2)

        x_plus = Zhangliang(x + x_var)
        x_minus = Zhangliang(x - x_var)

        z_plus = fn(x_plus, y, *args, **kwargs)
        z_minus = fn(x_minus, y, *args, **kwargs)

        x_numer_grad = (z_plus.values - z_minus.values) / (2 * x_var)
        x_numer_grad = np.sum(x_numer_grad, axis=axes_to_reduce[0])

        if not array_close(x_numer_grad, x.grad):
            print('Test derivative check of `{}` w.r.t. `x` ({}/10) failed.\n' \
                  ' Analytic: {}\n Numeric: {}'.format(op_name, i, x, y, x.grad, x_numer_grad))
        else:
            print('Test derivative check of `{}` w.r.t. `x` ({}/10) passed.'.format(op_name, i))

    gn(z, x, y, *args, **kwargs)
    for i in range(10):
        y_var = np.random.rand(*(y.shape)) * EPS / 2
        y_var = np.where(y_var, y_var, EPS / 2)

        y_plus = Zhangliang(y + y_var, dtype=np.float64)
        y_minus = Zhangliang(y - y_var, dtype=np.float64)

        z_plus = fn(x, y_plus, *args, **kwargs)
        z_minus = fn(x, y_minus, *args, **kwargs)

        y_numer_grad = (z_plus.values - z_minus.values) / (2 * y_var)
        y_numer_grad = np.sum(y_numer_grad, axis=axes_to_reduce[1])

        if not array_close(y_numer_grad, y.grad):
            print('Test derivative check of `{}` w.r.t. `y` ({}/10) failed.\n' \
                  ' Analytic: {}\n Numeric: {}'.format(op_name, i, x, y, y.grad, y_numer_grad))
        else:
            print('Test derivative check of `{}` w.r.t. `y` ({}/10) passed.'.format(op_name, i))


def test_check_forward_func_and_backkward_func():
    from utils import func_lib, grad_lib
    forward_keys = func_lib.keys()
    backward_keys = grad_lib.keys()
    diff = set(forward_keys) - set(backward_keys)
    assert len(diff) == 0


def test_check_binary_func():
    get_arg = arg_space()
    op_list = {'add', 'sub', 'mul', 'div', 'minimum', 'maximum'}
    for arg1, arg2 in get_arg:
        for op_name in sorted(op_list):
            print('---------------------------------------------------------')
            print('Arg1: {}\nArg2: {}\nop: {}'.format(arg1, arg2, op_name))
            check_element_binary_numeric_grad(op_name, arg1, arg2)

