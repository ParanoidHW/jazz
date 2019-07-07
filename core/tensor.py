from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numbers
import numpy as np

from utils import sanity
from utils.tracer import trace


class Zhangliang(object):
    def __init__(self, values, dtype=np.float32):
        self.zhi = np.array(values, dtype=dtype)

    @property
    def shape(self):
        return self.zhi.shape

    @property
    def dtype(self):
        return self.zhi.dtype

    @classmethod
    def from_array(cls, values, dtype=np.float32):
        return cls(values, dtype=dtype)

    @classmethod
    def zeros(cls, shape, dtype=np.float32):
        zeros_ = np.zeros(shape, dtype=dtype)
        return cls(zeros_, )

    @classmethod
    def ones(cls, shape, dtype=np.float32):
        ones_ = np.ones(shape, dtype=dtype)
        return cls(ones_)

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
        return zl_elt_pow(self, power)

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


# ---------------------------------------------------------- #
# math ops for Zhangliang
# ---------------------------------------------------------- #


@trace(name='add')
def zl_add(a, b):
    if isinstance(a, numbers.Real):
        value = a + b.zhi
    elif isinstance(b, numbers.Real):
        value = a.zhi + b
    else:
        value = a.zhi + b.zhi
    return Zhangliang(value)


@trace(name='sub')
def zl_sub(a, b):
    if isinstance(a, numbers.Real):
        value = a - b.zhi
    elif isinstance(b, numbers.Real):
        value = a.zhi - b
    else:
        value = a.zhi - b.zhi
    return Zhangliang(value)


@trace(name='mul')
def zl_mul(a, b):
    if isinstance(a, numbers.Real):
        value = a * b.zhi
    elif isinstance(b, numbers.Real):
        value = a.zhi * b
    else:
        value = a.zhi * b.zhi
    return Zhangliang(value)


@trace(name='rdiv')
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


@trace(name='matmul')
def zl_matmul(a, b):
    sanity.TypeCheck(a, Zhangliang)
    sanity.TypeCheck(b, Zhangliang)
    return Zhangliang(np.matmul(a.zhi, b.zhi))


@trace(name='abs')
def zl_abs(a):
    if isinstance(a, Zhangliang):
        value = np.abs(a.zhi)
    else:
        value = np.abs(a)
    return Zhangliang(value)


@trace(name='elt_pow')
def zl_elt_pow(a, power):
    sanity.TypeCheck(a, Zhangliang)
    return Zhangliang(np.power(a.zhi, power))


@trace(name='log')
def zl_log(a, base=np.e):
    sanity.TypeCheck(a, Zhangliang)
    sanity.ClosedRangeCheck(base, xmin=0)
    value = np.log(a.zhi) / np.log(base)
    return Zhangliang(value)


@trace(name='ge')
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


@trace(name='gt')
def zl_gt(a, b):
    pass


@trace(name='le')
def zl_le(a, b):
    pass


@trace(name='lt')
def zl_lt(a, b):
    pass


@trace(name='eq')
def zl_eq(a, b):
    pass


@trace(name='ne')
def zl_ne(a, b):
    pass


@trace(name='elt_and')
def zl_elt_and(a, b):
    pass


@trace(name='elt_or')
def zl_elt_or(a, b):
    pass


@trace(name='elt_not')
def zl_elt_not(a):
    pass


@trace(name='elt_xor')
def zl_elt_xor(a):
    pass


@trace(name='sin')
def zl_sin(a):
    pass


@trace(name='cos')
def zl_cos(a):
    pass


@trace(name='tan')
def zl_tan(a):
    pass


@trace(name='cot')
def zl_cot(a):
    pass

