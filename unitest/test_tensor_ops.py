from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from itertools import product
import numpy as np
from core import Zhangliang
from utils.register import func_lib, grad_lib
from utils.misc import multiplicative_broadcast_analysis, additive_broadcast_analysis
from utils.tracer import graph


TOL = 1e-6
RTOL = 1e-6
EPS = 1e-6


def array_close(a, b, tol=TOL, rtol=RTOL):
    array_flag = np.logical_or(np.abs(a - b) < tol, np.abs(a - b) / np.abs(a + b) < rtol)
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
    from utils.register import func_lib
    kernel = np.array(
        [[[[6.1059e-01, 7.3151e-01, 3.8562e-01],
           [1.0263e-01, 4.0959e-01, 4.5504e-01],
           [7.9654e-01, 1.2602e-01, 1.7744e-01]],

          [[6.1563e-01, 8.0071e-01, 3.4159e-01],
           [7.8039e-01, 9.6654e-01, 8.7575e-01],
           [8.7650e-01, 6.8633e-01, 6.3384e-01]],

          [[2.9005e-01, 9.8753e-04, 8.4554e-01],
           [2.6094e-01, 2.6123e-01, 8.2337e-01],
           [2.1928e-01, 8.7154e-01, 1.9465e-01]],

          [[8.2702e-01, 2.1772e-01, 3.7523e-01],
           [1.9798e-02, 9.9153e-01, 4.3943e-03],
           [2.9794e-01, 7.7120e-01, 1.3000e-01]]],

         [[[1.5315e-01, 8.9263e-01, 3.5529e-01],
           [1.1891e-01, 6.5386e-01, 9.8741e-01],
           [7.5431e-01, 8.0253e-01, 6.8073e-01]],

          [[6.5027e-01, 4.2663e-01, 1.3244e-01],
           [2.2020e-01, 5.0186e-01, 7.4191e-01],
           [5.2859e-01, 7.6619e-01, 4.1443e-01]],

          [[4.4136e-01, 4.2113e-01, 4.9212e-01],
           [8.6997e-02, 4.9597e-01, 2.6384e-01],
           [2.3810e-01, 3.8803e-02, 5.9454e-02]],

          [[1.6808e-01, 4.4940e-01, 4.6931e-01],
           [4.1643e-01, 5.9496e-01, 7.1671e-01],
           [8.8583e-01, 4.7752e-01, 4.7285e-01]]],

         [[[5.4441e-01, 4.1712e-01, 3.8568e-01],
           [4.7978e-01, 2.5011e-01, 2.8174e-01],
           [2.2808e-01, 5.7628e-01, 3.6194e-01]],

          [[4.8947e-01, 3.5466e-01, 2.4454e-01],
           [9.6375e-01, 4.9558e-01, 5.3628e-01],
           [5.6494e-01, 4.4015e-01, 5.9933e-01]],

          [[2.6900e-01, 1.6869e-02, 1.5678e-01],
           [7.2846e-01, 2.5035e-01, 1.4957e-01],
           [6.1529e-01, 3.1775e-01, 1.3441e-02]],

          [[3.2598e-01, 1.6510e-01, 5.2898e-01],
           [4.7248e-01, 1.0142e-01, 7.1356e-01],
           [7.2168e-01, 2.1397e-01, 7.9877e-01]]]], dtype=np.float32
    )
    kernel = Zhangliang(kernel, requires_grad=True)
    x = np.ones((2,4,5,5), dtype=np.float32)
    x = Zhangliang(x, requires_grad=True)
    gt = np.array(
        [[[[8.3785, 11.7325, 11.7325, 11.7325, 8.4380],
           [12.0774, 17.7747, 17.7747, 17.7747, 12.5322],
           [12.0774, 17.7747, 17.7747, 17.7747, 12.5322],
           [12.0774, 17.7747, 17.7747, 17.7747, 12.5322],
           [8.4864, 11.9934, 11.9934, 11.9934, 7.8869]],

          [[8.6690, 11.9184, 11.9184, 11.9184, 7.5811],
           [12.3080, 16.9702, 16.9702, 16.9702, 11.1837],
           [12.3080, 16.9702, 16.9702, 16.9702, 11.1837],
           [12.3080, 16.9702, 16.9702, 16.9702, 11.1837],
           [8.5955, 10.8509, 10.8509, 10.8509, 6.6918]],

          [[6.1003, 10.8747, 10.8747, 10.8747, 7.4201],
           [8.3700, 14.7733, 14.7733, 14.7733, 10.0027],
           [8.3700, 14.7733, 14.7733, 14.7733, 10.0027],
           [8.3700, 14.7733, 14.7733, 14.7733, 10.0027],
           [5.0483, 9.3217, 9.3217, 9.3217, 6.3245]]],

         [[[8.3785, 11.7325, 11.7325, 11.7325, 8.4380],
           [12.0774, 17.7747, 17.7747, 17.7747, 12.5322],
           [12.0774, 17.7747, 17.7747, 17.7747, 12.5322],
           [12.0774, 17.7747, 17.7747, 17.7747, 12.5322],
           [8.4864, 11.9934, 11.9934, 11.9934, 7.8869]],

          [[8.6690, 11.9184, 11.9184, 11.9184, 7.5811],
           [12.3080, 16.9702, 16.9702, 16.9702, 11.1837],
           [12.3080, 16.9702, 16.9702, 16.9702, 11.1837],
           [12.3080, 16.9702, 16.9702, 16.9702, 11.1837],
           [8.5955, 10.8509, 10.8509, 10.8509, 6.6918]],

          [[6.1003, 10.8747, 10.8747, 10.8747, 7.4201],
           [8.3700, 14.7733, 14.7733, 14.7733, 10.0027],
           [8.3700, 14.7733, 14.7733, 14.7733, 10.0027],
           [8.3700, 14.7733, 14.7733, 14.7733, 10.0027],
           [5.0483, 9.3217, 9.3217, 9.3217, 6.3245]]]], dtype=np.float32
    )

    conv_fn = func_lib['conv2d']
    pred = conv_fn(x, kernel, stride=1, padding=1)
    print(array_close(pred.values, gt, tol=1e-5, rtol=1e-5))

    sum_fn = func_lib['reduce_sum']
    loss = sum_fn(pred)
    loss.backward()
    kgrad = np.array(
        [[[[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]],
          [[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]],
          [[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]],
          [[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]]],
         [[[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]],
          [[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]],
          [[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]],
          [[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]]],
         [[[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]],
          [[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]],
          [[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]],
          [[32., 40., 32.],
           [40., 50., 40.],
           [32., 40., 32.]]]]
    )
    xgrad = np.array(
        [[[[5.3643, 8.2151, 8.2151, 8.2151, 6.2056],
           [8.6481, 12.7189, 12.7189, 12.7189, 8.9305],
           [8.6481, 12.7189, 12.7189, 12.7189, 8.9305],
           [8.6481, 12.7189, 12.7189, 12.7189, 8.9305],
           [5.2986, 8.2429, 8.2429, 8.2429, 5.7627]],
          [[7.2657, 10.1382, 10.1382, 10.1382, 6.4185],
           [11.1284, 15.6485, 15.6485, 15.6485, 9.9588],
           [11.1284, 15.6485, 15.6485, 15.6485, 9.9588],
           [11.1284, 15.6485, 15.6485, 15.6485, 9.9588],
           [7.7910, 11.5926, 11.5926, 11.5926, 7.6582]],
          [[3.5233, 6.2546, 6.2546, 6.2546, 4.1778],
           [5.8241, 8.8229, 8.8229, 8.8229, 5.6734],
           [5.8241, 8.8229, 8.8229, 8.8229, 5.6734],
           [5.8241, 8.8229, 8.8229, 8.8229, 5.6734],
           [4.3847, 5.8890, 5.8890, 5.8890, 3.7400]],
          [[4.7499, 7.5581, 7.5581, 7.5581, 5.3283],
           [8.1181, 12.3279, 12.3279, 12.3279, 8.1926],
           [8.1181, 12.3279, 12.3279, 12.3279, 8.1926],
           [8.1181, 12.3279, 12.3279, 12.3279, 8.1926],
           [5.9648, 8.8010, 8.8010, 8.8010, 5.9869]]],
         [[[5.3643, 8.2151, 8.2151, 8.2151, 6.2056],
           [8.6481, 12.7189, 12.7189, 12.7189, 8.9305],
           [8.6481, 12.7189, 12.7189, 12.7189, 8.9305],
           [8.6481, 12.7189, 12.7189, 12.7189, 8.9305],
           [5.2986, 8.2429, 8.2429, 8.2429, 5.7627]],
          [[7.2657, 10.1382, 10.1382, 10.1382, 6.4185],
           [11.1284, 15.6485, 15.6485, 15.6485, 9.9588],
           [11.1284, 15.6485, 15.6485, 15.6485, 9.9588],
           [11.1284, 15.6485, 15.6485, 15.6485, 9.9588],
           [7.7910, 11.5926, 11.5926, 11.5926, 7.6582]],
          [[3.5233, 6.2546, 6.2546, 6.2546, 4.1778],
           [5.8241, 8.8229, 8.8229, 8.8229, 5.6734],
           [5.8241, 8.8229, 8.8229, 8.8229, 5.6734],
           [5.8241, 8.8229, 8.8229, 8.8229, 5.6734],
           [4.3847, 5.8890, 5.8890, 5.8890, 3.7400]],
          [[4.7499, 7.5581, 7.5581, 7.5581, 5.3283],
           [8.1181, 12.3279, 12.3279, 12.3279, 8.1926],
           [8.1181, 12.3279, 12.3279, 12.3279, 8.1926],
           [8.1181, 12.3279, 12.3279, 12.3279, 8.1926],
           [5.9648, 8.8010, 8.8010, 8.8010, 5.9869]]]]
    )
    print(array_close(x.grad, xgrad, tol=1e-5, rtol=1e-5))
    print(array_close(kernel.grad, kgrad, tol=1e-5, rtol=1e-5))
