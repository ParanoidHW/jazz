from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numbers
import numpy as np

from core.base import BaseZhangliang
from utils import sanity
from utils.tracer import trace
from utils.register import grad_register


class Zhangliang(BaseZhangliang):
    def __init__(self, values, dtype=np.float32, name='', requires_grad=False):
        super(Zhangliang, self).__init__(values, dtype, name, requires_grad)

    def __add__(self, other):
        return zl_add(self, other)

    def __radd__(self, other):
        return zl_add(other, self)

    def __sub__(self, other):
        return zl_sub(self, other)

    def __rsub__(self, other):
        return zl_sub(other, self)

    def __truediv__(self, other):
        return zl_truediv(self, other)

    def __abs__(self):
        return zl_abs(self)

    def __iadd__(self, other):
        return zl_add(self, other)

    def __isub__(self, other):
        return zl_sub(self, other)

    def __imul__(self, other):
        return zl_mul(self, other)

    def __imatmul__(self, other):
        return zl_matmul(self, other)

    def __itruediv__(self, other):
        return zl_truediv(self, other)

    def __pow__(self, power, modulo=None):
        return zl_pow(self, power)

    def __ge__(self, other):
        return zl_ge(self, other)

    def __gt__(self, other):
        return zl_gt(self, other)

    def __le__(self, other):
        return zl_le(self, other)

    def __lt__(self, other):
        return zl_lt(self, other)

    def __eq__(self, other):
        return zl_eq(self, other)

    def __ne__(self, other):
        return zl_ne(self, other)

    def __and__(self, other):
        return zl_elt_and(self, other)

    def __or__(self, other):
        return zl_elt_or(self, other)

    def __xor__(self, other):
        return zl_elt_xor(self, other)

    def __neg__(self):
        return zl_sub(0., self)

    def sum(self, dim=None, keepdims=False):
        return zl_reduce_sum(self, dim, keepdims)

    def mean(self, dim=None, keepdims=False):
        return zl_reduce_mean(self, dim, keepdims)


# ---------------------------------------------------------- #
# math ops for Zhangliang
# ---------------------------------------------------------- #


@trace(op_name='add')
def zl_add(a, b):
    if isinstance(a, numbers.Real):
        value = a + b.zhi
    elif isinstance(b, numbers.Real):
        value = a.zhi + b
    else:
        value = a.zhi + b.zhi
    return Zhangliang(value)


@grad_register(op_name='add')
def zl_add_grad(inputs, outputs_grad):
    assert len(inputs) == 2
    inputs_grad = [0 for _ in range(len(inputs))]
    inputs_grad[0] = np.array(outputs_grad.zhi)
    inputs_grad[1] = np.array(outputs_grad.zhi)
    return


@trace(op_name='sub')
def zl_sub(a, b):
    if isinstance(a, numbers.Real):
        value = a - b.zhi
    elif isinstance(b, numbers.Real):
        value = a.zhi - b
    else:
        value = a.zhi - b.zhi
    return Zhangliang(value)


@grad_register(op_name='sub')
def zl_sub_grad(inputs, outputs_grad):
    assert len(inputs) == 2
    inputs_grad = [0 for _ in range(len(inputs))]
    inputs_grad[0] = np.array(outputs_grad.zhi)
    inputs_grad[1] = - np.array(outputs_grad.zhi)
    return


@trace(op_name='reduce_mean')
def zl_reduce_mean(a, dim=None, keepdims=False):
    values = np.mean(a.zhi, axis=dim, keepdims=keepdims)
    return Zhangliang(values)


@trace(op_name='reduce_sum')
def zl_reduce_sum(a, dim=None, keepdims=False):
    values = np.sum(a.zhi, axis=dim, keepdims=keepdims)
    return Zhangliang(values)


@trace(op_name='mul')
def zl_mul(a, b):
    if isinstance(a, numbers.Real):
        value = a * b.zhi
    elif isinstance(b, numbers.Real):
        value = a.zhi * b
    else:
        value = a.zhi * b.zhi
    return Zhangliang(value)


@trace(op_name='rdiv')
def zl_truediv(a, b):
    if isinstance(a, numbers.Real):
        value = a / b.zhi
    elif isinstance(b, numbers.Real):
        if b == 0:
            raise ValueError('0 cannot be divisor.')
        value = a.zhi / b
    else:
        value = a.zhi / b.zhi
    return Zhangliang(value)


@trace(op_name='matmul')
def zl_matmul(a, b):
    sanity.TypeCheck(a, Zhangliang)
    sanity.TypeCheck(b, Zhangliang)
    return Zhangliang(np.matmul(a.zhi, b.zhi))


@trace(op_name='abs')
def zl_abs(a):
    if isinstance(a, numbers.Real):
        value = np.abs(a)
    else:
        value = np.abs(a.zhi)
    return Zhangliang(value)


@trace(op_name='elt_pow')
def zl_pow(a, power):
    sanity.TypeCheck(a, Zhangliang)
    return Zhangliang(np.power(a.zhi, power))


@trace(op_name='log')
def zl_log(a, base=np.e):
    sanity.TypeCheck(a, Zhangliang)
    assert base > 0
    value = np.log(a.zhi) / np.log(base)
    return Zhangliang(value)


@trace(op_name='max')
def zl_max(a, dim=None):
    value = np.max(a.zhi, axis=dim)
    return Zhangliang(value)


@trace(op_name='maximum')
def zl_maximum(a, b):
    value = np.maximum(a.zhi, b.zhi)
    return Zhangliang(value)


@trace(op_name='min')
def zl_max(a, dim=None):
    value = np.min(a.zhi, axis=dim)
    return Zhangliang(value)


@trace(op_name='minimum')
def zl_maximum(a, b):
    value = np.minimum(a.zhi, b.zhi)
    return Zhangliang(value)


@trace(op_name='ge')
def zl_ge(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.zhi >= b.zhi
    elif isinstance(a, Zhangliang):
        value = a.zhi >= b
    elif isinstance(b, Zhangliang):
        value = a >= b.zhi
    else:
        value = a >= b
    return Zhangliang(value)


@trace(op_name='gt')
def zl_gt(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.zhi > b.zhi
    elif isinstance(a, Zhangliang):
        value = a.zhi > b
    elif isinstance(b, Zhangliang):
        value = a > b.zhi
    else:
        value = a > b
    return Zhangliang(value)


@trace(op_name='le')
def zl_le(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.zhi <= b.zhi
    elif isinstance(a, Zhangliang):
        value = a.zhi <= b
    elif isinstance(b, Zhangliang):
        value = a <= b.zhi
    else:
        value = a <= b
    return Zhangliang(value)


@trace(op_name='lt')
def zl_lt(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.zhi < b.zhi
    elif isinstance(a, Zhangliang):
        value = a.zhi < b
    elif isinstance(b, Zhangliang):
        value = a < b.zhi
    else:
        value = a < b
    return Zhangliang(value)


@trace(op_name='eq')
def zl_eq(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.zhi == b.zhi
    elif isinstance(a, Zhangliang):
        value = a.zhi == b
    elif isinstance(b, Zhangliang):
        value = a == b.zhi
    else:
        value = a == b
    return Zhangliang(value)


@trace(op_name='ne')
def zl_ne(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.zhi != b.zhi
    elif isinstance(a, Zhangliang):
        value = a.zhi != b
    elif isinstance(b, Zhangliang):
        value = a != b.zhi
    else:
        value = a != b
    return Zhangliang(value)


@trace(op_name='clamp')
def zl_clamp(a, xmin=0., xmax=1.):
    value = np.clip(a.zhi, a_max=xmax, a_min=xmin)
    return Zhangliang(value)


@trace(op_name='elt_and')
def zl_elt_and(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = np.logical_and(a.zhi, b.zhi)
    elif isinstance(a, Zhangliang):
        value = np.logical_and(a.zhi, b)
    elif isinstance(b, Zhangliang):
        value = np.logical_and(a, b.zhi)
    else:
        value = np.logical_and(a, b)
    return Zhangliang(value)


@trace(op_name='elt_or')
def zl_elt_or(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = np.logical_or(a.zhi, b.zhi)
    elif isinstance(a, Zhangliang):
        value = np.logical_or(a.zhi, b)
    elif isinstance(b, Zhangliang):
        value = np.logical_or(a, b.zhi)
    else:
        value = np.logical_or(a, b)
    return Zhangliang(value)


@trace(op_name='elt_not')
def zl_elt_not(a):
    if isinstance(a, Zhangliang):
        value = np.logical_not(a.zhi)
    else:
        value = np.logical_not(a)
    return Zhangliang(value)


@trace(op_name='elt_xor')
def zl_elt_xor(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = np.logical_xor(a.zhi, b.zhi)
    elif isinstance(a, Zhangliang):
        value = np.logical_xor(a.zhi, b)
    elif isinstance(b, Zhangliang):
        value = np.logical_xor(a, b.zhi)
    else:
        value = np.logical_xor(a, b)
    return Zhangliang(value)


@trace(op_name='sin')
def zl_sin(a):
    if isinstance(a, Zhangliang):
        value = np.sin(a.zhi)
    else:
        value = np.sin(a)
    return Zhangliang(value)


@trace(op_name='cos')
def zl_cos(a):
    if isinstance(a, Zhangliang):
        value = np.cos(a.zhi)
    else:
        value = np.cos(a)
    return Zhangliang(value)


@trace(op_name='tan')
def zl_tan(a):
    if isinstance(a, Zhangliang):
        value = np.tan(a.zhi)
    else:
        value = np.tan(a)
    return Zhangliang(value)


# numpy package has no `cot` function. So we skip `cot`, as well as `arccot`.
@trace(op_name='arcsin')
def zl_arcsin(a):
    if isinstance(a, Zhangliang):
        value = np.arcsin(a.zhi)
    else:
        value = np.arcsin(a)
    return Zhangliang(value)


@trace(op_name='arccos')
def zl_arccos(a):
    if isinstance(a, Zhangliang):
        value = np.arccos(a.zhi)
    else:
        value = np.arccos(a)
    return Zhangliang(value)


@trace(op_name='arctan')
def zl_arctan(a):
    if isinstance(a, Zhangliang):
        value = np.arctan(a.zhi)
    else:
        value = np.arctan(a)
    return Zhangliang(value)


@trace(op_name='sinh')
def zl_sinh(a):
    if isinstance(a, Zhangliang):
        value = np.sinh(a.zhi)
    else:
        value = np.sinh(a)
    return Zhangliang(value)


@trace(op_name='cosh')
def zl_cosh(a):
    if isinstance(a, Zhangliang):
        value = np.cosh(a.zhi)
    else:
        value = np.cosh(a)
    return Zhangliang(value)


@trace(op_name='tanh')
def zl_tanh(a):
    if isinstance(a, Zhangliang):
        value = np.tanh(a.zhi)
    else:
        value = np.tanh(a)
    return Zhangliang(value)


@trace(op_name='arcsinh')
def zl_arcsinh(a):
    if isinstance(a, Zhangliang):
        value = np.arcsinh(a.zhi)
    else:
        value = np.arcsinh(a)
    return Zhangliang(value)


@trace(op_name='arccosh')
def zl_arccosh(a):
    if isinstance(a, Zhangliang):
        value = np.arccosh(a.zhi)
    else:
        value = np.arccosh(a)
    return Zhangliang(value)


@trace(op_name='arctanh')
def zl_arctanh(a):
    if isinstance(a, Zhangliang):
        value = np.arctanh(a.zhi)
    else:
        value = np.arctanh(a)
    return Zhangliang(value)



if __name__ == '__main__':
    from utils.tracer import graph
    from utils.register import func_lib, grad_lib
    a = Zhangliang([2,3])
    b = Zhangliang([-1,0])

    z1 = a + b + a
    print(z1)
