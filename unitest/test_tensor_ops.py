from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from itertools import product
import pickle as pkl
import numpy as np

from core import Zhangliang
from utils.register import func_lib, grad_lib
from utils.misc import multiplicative_broadcast_analysis, additive_broadcast_analysis
from utils.tracer import graph


TOL = 1e-6
RTOL = 1e-6
EPS = 1e-6


def array_close(a, b, tol=TOL, rtol=RTOL):
    array_flag = np.logical_or(np.abs(a - b) < tol, (np.abs(a - b) / np.abs(a + b)) < rtol)
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
        z.update_grad(np.ones_like(z))
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
        z.update_grad(np.ones_like(z))
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


def test_maxmin_unary_func():
    def unary_arg_space():
        vector = np.random.randn(4)
        mat = np.random.randn(3, 4)
        mat2 = np.random.randn(1, 4, 5)
        allargs = [vector, mat, mat2]
        reduced_dim = [(1,), (2,), None]
        for arg1, dim in product(allargs, reduced_dim):
            yield arg1, dim

    def check_grad(op_name, a, dim=None):
        max_trial = 5
        fn = func_lib[op_name]
        gn = grad_lib[op_name]
        x = Zhangliang(a, requires_grad=True)
        z = fn(x, dim=dim)
        z.update_grad(np.ones_like(z))
        gn(z, x, dim=dim)

        for i in range(max_trial):
            x_var = np.random.rand(*(x.shape)) * EPS / 2
            x_var = np.where(x_var, x_var, EPS / 2)

            x_plus = Zhangliang(x + x_var)
            x_minus = Zhangliang(x - x_var)

            z_plus = fn(x_plus, dim=dim, keepdims=True)
            z_pmask = z_plus == x_plus
            z_minus = fn(x_minus, dim=dim, keepdims=True)
            z_mmask = z_minus == x_minus
            z_res = 1.0 * z_pmask.values * x_plus.values - 1.0 * z_mmask.values * x_minus.values

            x_numer_grad = z_res / (2 * x_var)
            if np.isscalar(x_numer_grad):
                x_numer_grad = np.array([x_numer_grad], dtype=np.float64)

            if not array_close(x_numer_grad, x.grad):
                print('Test derivative of `{}` w.r.t. `x` ({}/{}) failed.\n' \
                      ' Analytic: {}\n Numeric: {}'.format(op_name, i, max_trial, x.grad, x_numer_grad))
            else:
                print('Test derivative of `{}` w.r.t. `x` ({}/{}) passed.'.format(op_name, i, max_trial))

    op_list = ['max', 'min']

    for op_name in sorted(op_list):
        get_arg = unary_arg_space()
        for arg1, dim in get_arg:
            if dim is not None and max(dim) >= len(arg1.shape):
                continue
            print('---------------------------------------------------------')
            print('Arg1: {}\ndim: {}\nop: {}'.format(arg1, dim, op_name))
            check_grad(op_name, arg1, dim=dim)


def test_backward():
    from core import sin, log
    from core.grad_mode import no_grad, has_grad
    x1 = Zhangliang(2, requires_grad=True)
    x2 = Zhangliang(5, requires_grad=True)

    f = log(x1) + x1*x2 - sin(x2)
    f.backward()
    print("Test function f=log(x1)+x1*x2-sin(x2), with initial values x1=2, x2=5.\n"
          "\tOracle grad: g_x1 = {:.5f}, g_x2 = {:.5f}\n"
          "\tResult grad: g_x1 = {:.5f}, g_x2 = {:.5f}".
          format(5.5, 1.716, x1.grad[0], x2.grad[0]))

    x1 = Zhangliang(2, requires_grad=True)
    x2 = 5

    f = log(x1) + x1 * x2 - sin(x2)
    f.backward()
    print("Test function f=log(x1)+x1*x2-sin(x2), with initial values x1=2, x2=5.\n"
          "\tOracle grad: g_x1 = {:.5f}\n"
          "\tResult grad: g_x1 = {:.5f}".
          format(5.5, x1.grad[0]))

    x1 = 2
    x2 = Zhangliang(5, requires_grad=True)
    f = log(x1) + x1 * x2 - sin(x2)
    f.backward()
    print("Test function f=log(x1)+x1*x2-sin(x2), with initial values x1=2, x2=5.\n"
          "\tOracle grad: g_x2 = {:.5f}\n"
          "\tResult grad: g_x2 = {:.5f}".
          format(1.716, x2.grad[0]))

    # Test no_grad
    x1 = Zhangliang(2, requires_grad=True)
    x2 = Zhangliang(5, requires_grad=True)

    with no_grad():
        f = log(x1) + x1 * x2 - sin(x2)

    try:
        f.backward()
        print('This line should not be print.')
    except:
        print('Backprop is disabled in `no_grad` situation.')

    # Test has_grad
    x1 = Zhangliang(2, requires_grad=True)
    x2 = Zhangliang(5, requires_grad=True)

    with no_grad():
        with has_grad():
            f = log(x1) + x1 * x2 - sin(x2)

    try:
        f.backward()
        print("Test function f=log(x1)+x1*x2-sin(x2), with initial values x1=2, x2=5.\n"
              "\tOracle grad: g_x1 = {:.5f}, g_x2 = {:.5f}\n"
              "\tResult grad: g_x1 = {:.5f}, g_x2 = {:.5f}".
              format(5.5, 1.716, x1.grad[0], x2.grad[0]))
    except:
        print('This line should not be print.')


def test_conv_forward():
    sum_fn = func_lib['reduce_sum']
    conv_fn = func_lib['conv2d']

    def do_one_run(in_x, in_k, in_b, stride, padding, dilation, out, x_grad, k_grad, b_grad, bias, **kwargs):
        print('------------------------------------------------')
        print('Test settings: bias={}, stride={}, padding={}, dilation={}'.
              format(bias, stride, padding, dilation))
        print('Forward ....', end='')
        x = Zhangliang(in_x, requires_grad=True)
        k = Zhangliang(in_k, requires_grad=True)
        if bias:
            b = Zhangliang(in_b, requires_grad=True)
            print_end = ''
        else:
            b = None
            print_end = '\n'
        y = conv_fn(x, k, bias=b, stride=stride, padding=padding, dilation=dilation)
        forward_array_close = array_close(y, out, tol=1e-5, rtol=1e-5)
        if forward_array_close:
            print('y success :)')
        else:
            print('y failed :(')

        print('Backward ....', end='')
        ysum = sum_fn(y)
        ysum.backward()
        if array_close(x.grad, x_grad):
            print(' `x` sucess ', end='')
        else:
            print(' `x` failed,', end='')

        if array_close(k.grad, k_grad):
            print(' `k` sucess ', end=print_end)
        else:
            print(' `k` failed ', end=print_end)

        if bias:
            if array_close(b.grad, b_grad):
                print(' `b` sucess ')
            else:
                print(' `b` failed ')

    with open('test_conv.pkl', 'rb') as f:
        runs = pkl.load(f)

    for one_run in runs:
        do_one_run(**one_run)
