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


def array_close(a, b):
    array_flag = np.logical_or(np.abs(a - b) < TOL, np.abs(a - b) / np.abs(a + b) < RTOL)
    return np.all(array_flag) and \
            (a.shape == b.shape) and \
           not np.any(np.isnan(a)) and \
            not np.any(np.isnan(b))


def approx_numeric_grad(z_p, z_m):
    return (z_p - z_m) / EPS


def test_check_forward_func_and_backkward_func():
    from utils import func_lib, grad_lib
    forward_keys = func_lib.keys()
    backward_keys = grad_lib.keys()
    diff = set(forward_keys) - set(backward_keys)
    assert len(diff) == 0


def test_element_binary_func():
    def binary_arg_space():
        scalar = 2.0
        vector = np.random.randn(4)
        mat = np.random.randn(3, 4)
        mat2 = np.random.randn(1, 4)
        allargs = [scalar, vector, mat, mat2]
        for arg1, arg2 in product(allargs, allargs):
            yield arg1, arg2

    def check_grad(op_name, a, b, *args, **kwargs):
        max_trial = 5
        fn = func_lib[op_name]
        gn = grad_lib[op_name]
        x = Zhangliang(a, requires_grad=True)
        y = Zhangliang(b, requires_grad=True)
        z = fn(x, y, *args, **kwargs)
        z.assign_grad(np.ones_like(z))
        gn(z, x, y, *args, **kwargs)
        axes_to_reduce = additive_broadcast_analysis([x.shape, y.shape], z.shape)

        for i in range(max_trial):
            x_var = np.random.rand(*(x.shape)) * EPS / 2
            x_var = np.where(x_var, x_var, EPS / 2)

            x_plus = Zhangliang(x + x_var)
            x_minus = Zhangliang(x - x_var)

            z_plus = fn(x_plus, y, *args, **kwargs)
            z_minus = fn(x_minus, y, *args, **kwargs)

            x_numer_grad = (z_plus.values - z_minus.values) / (2 * x_var)
            x_numer_grad = np.sum(x_numer_grad, axis=axes_to_reduce[0])
            if np.isscalar(x_numer_grad):
                x_numer_grad = np.array([x_numer_grad], dtype=np.float64)
            x_numer_grad = np.reshape(x_numer_grad, x.shape)

            if not array_close(x_numer_grad, x.grad):
                print('Test derivative of `{}` w.r.t. `x` ({}/{}) failed.\n' \
                      ' Analytic: {}\n Numeric: {}'.format(op_name, i, max_trial, x.grad, x_numer_grad))
            else:
                print('Test derivative of `{}` w.r.t. `x` ({}/{}) passed.'.format(op_name, i, max_trial))

        gn(z, x, y, *args, **kwargs)
        for i in range(max_trial):
            y_var = np.random.rand(*(y.shape)) * EPS / 2
            y_var = np.where(y_var, y_var, EPS / 2)

            y_plus = Zhangliang(y + y_var, dtype=np.float64)
            y_minus = Zhangliang(y - y_var, dtype=np.float64)

            z_plus = fn(x, y_plus, *args, **kwargs)
            z_minus = fn(x, y_minus, *args, **kwargs)

            y_numer_grad = (z_plus.values - z_minus.values) / (2 * y_var)
            y_numer_grad = np.sum(y_numer_grad, axis=axes_to_reduce[1])

            if np.isscalar(y_numer_grad):
                y_numer_grad = np.array([y_numer_grad], dtype=np.float64)
            y_numer_grad = np.reshape(y_numer_grad, y.shape)

            if not array_close(y_numer_grad, y.grad):
                print('Test derivative of `{}` w.r.t. `y` ({}/{}) failed.\n' \
                      ' Analytic: {}\n Numeric: {}'.format(op_name, i, max_trial, y.grad, y_numer_grad))
            else:
                print('Test derivative of `{}` w.r.t. `y` ({}/{}) passed.'.format(op_name, i, max_trial))

    op_list = ['add', 'sub', 'mul', 'div', 'minimum', 'maximum', 'pow']

    for op_name in sorted(op_list):
        get_arg = binary_arg_space()
        for arg1, arg2 in get_arg:
            if op_name == 'pow':
                arg1, arg2 = np.abs(arg1), np.abs(arg2)
                # 'Zhangliang' does not support `complex` data type right now.
                # So make sure the input of the `pow` is positive.

            print('---------------------------------------------------------')
            print('Arg1: {}\nArg2: {}\nop: {}'.format(arg1, arg2, op_name))
            check_grad(op_name, arg1, arg2)


def test_arg_free_unary_func():
    def unary_arg_space():
        scalar = 2.0
        vector = np.random.randn(4)
        mat = np.random.randn(3, 4)
        mat2 = np.random.randn(1, 4, 5)
        allargs = [scalar, vector, mat, mat2]
        for arg1 in allargs:
            yield arg1

    def check_grad(op_name, a, *args, **kwargs):
        max_trial = 5
        fn = func_lib[op_name]
        gn = grad_lib[op_name]
        x = Zhangliang(a, requires_grad=True)
        z = fn(x, *args, **kwargs)
        z.assign_grad(np.ones_like(z))
        gn(z, x, *args, **kwargs)
        axes_to_reduce = additive_broadcast_analysis([x.shape], z.shape)

        for i in range(max_trial):
            x_var = np.random.rand(*(x.shape)) * EPS / 2
            x_var = np.where(x_var, x_var, EPS / 2)

            x_plus = Zhangliang(x + x_var)
            x_minus = Zhangliang(x - x_var)

            z_plus = fn(x_plus, *args, **kwargs)
            z_minus = fn(x_minus, *args, **kwargs)

            x_numer_grad = (z_plus.values - z_minus.values) / (2 * x_var)
            x_numer_grad = np.sum(x_numer_grad, axis=axes_to_reduce[0])
            if np.isscalar(x_numer_grad):
                x_numer_grad = np.array([x_numer_grad], dtype=np.float64)
            x_numer_grad = np.reshape(x_numer_grad, x.shape)

            if not array_close(x_numer_grad, x.grad):
                print('Test derivative of `{}` w.r.t. `x` ({}/{}) failed.\n' \
                      ' Analytic: {}\n Numeric: {}'.format(op_name, i, max_trial, x.grad, x_numer_grad))
            else:
                print('Test derivative of `{}` w.r.t. `x` ({}/{}) passed.'.format(op_name, i, max_trial))

    op_list = ['neg', 'abs', 'clamp', 'sin', 'cos', 'tan',
               'sinh', 'cosh', 'tanh', 'arctan', 'arcsinh',
               'arcsin', 'arccos',
               'arccosh', 'arctanh',
               'log', 'log2', 'log10', 'log1p']

    for op_name in sorted(op_list):
        get_arg = unary_arg_space()
        for arg1 in get_arg:
            if op_name in ('arcsin', 'arccos', 'arctanh'):
                # TODO: Unstable near boundary
                arg1 = np.clip(arg1, a_min=-0.99999, a_max=.99999)
            elif op_name == 'arccosh':
                # TODO: Unstable near boundary
                arg1 = np.maximum(arg1, 1.00001)
            elif op_name in ['log', 'log2', 'log10', 'log1p']:
                # TODO: Unstable near boundary
                arg1 = np.abs(arg1)
            print('---------------------------------------------------------')
            print('Arg1: {}\nop: {}'.format(arg1, op_name))
            check_grad(op_name, arg1)


def test_check_shape_changed_unary_func():
    op_list = {'max', 'min',  'reduce_mean', 'reduce_sum', 'squeeze', 'unsqueeze'}
