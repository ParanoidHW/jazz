from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import collections
import numbers
import numpy as np

from core.base import BaseZhangliang
from utils import sanity
from utils.tracer import trace
from utils.register import grad_register

from utils.misc import additive_broadcast_analysis, multiplicative_broadcast_analysis


class Zhangliang(BaseZhangliang):
    def __init__(self, values, dtype=np.float32, requires_grad=False):
        super(Zhangliang, self).__init__(values, dtype, requires_grad)

    @classmethod
    def zeros(cls, shape, dtype=np.float32, requires_grad=False):
        zeros_ = np.zeros(shape, dtype=dtype)
        return cls(zeros_, requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape, dtype=np.float32, requires_grad=False):
        ones_ = np.ones(shape, dtype=dtype)
        return cls(ones_, requires_grad=requires_grad)

    @classmethod
    def array(cls, data, requires_grad=False):
        if isinstance(data, Zhangliang):
            return cls(data.values, dtype=data.dtype, requires_grad=requires_grad)
        elif isinstance(data, numbers.Integral):
            return cls(data, dtype=np.int32, requires_grad=requires_grad)
        elif isinstance(data, numbers.Real):
            return cls(data, dtype=np.float32, requires_grad=requires_grad)
        elif isinstance(data, (list, tuple)):
            return cls(data, dtype=np.float32, requires_grad=requires_grad)
        elif isinstance(data, collections.Iterable):
            data = np.array(data)
            return cls(data, dtype=np.float32, requires_grad=requires_grad)
        else:
            raise TypeError

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

    def reshape(self, new_shape):
        return zl_reshape(self, new_shape)

    def squeeze(self, dim=None):
        return zl_squeeze(self, dim=dim)

    def unsqueeze(self, dim):
        return zl_unsqueeze(self, dim)


# ---------------------------------------------------------- #
# math ops for Zhangliang
# ---------------------------------------------------------- #


@trace(op_name='add')
def zl_add(a, b):
    a = Zhangliang.array(a)
    b = Zhangliang.array(b)
    value = a.values + b.values
    return Zhangliang(value)


@grad_register(op_name='add')
def zl_add_grad(inputs, outputs_grad):
    assert len(inputs) == 2
    inputs_shapes = tuple([input_.shape for input_ in inputs])
    output_shape = outputs_grad.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(inputs[0], Zhangliang) and inputs[0].requires_grad:
        grads = np.sum(outputs_grad.grad, axis=axes_to_reduce[0])
        inputs[0]._tidu = np.reshape(grads, inputs[0].shape)
    if isinstance(inputs[1], Zhangliang) and inputs[1].requires_grad:
        grads = np.sum(outputs_grad.grad, axis=axes_to_reduce[1])
        inputs[1]._tidu = np.reshape(grads, inputs[1].shape)


@trace(op_name='sub')
def zl_sub(a, b):
    if isinstance(a, numbers.Real):
        value = a - b.values
    elif isinstance(b, numbers.Real):
        value = a.values - b
    else:
        value = a.values - b.values
    return Zhangliang(value)


@grad_register(op_name='sub')
def zl_sub_grad(inputs, outputs_grad):
    assert len(inputs) == 2
    inputs_shapes = tuple([input_.shape for input_ in inputs])
    output_shape = outputs_grad.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(inputs[0], Zhangliang) and inputs[0].requires_grad:
        grads = np.sum(outputs_grad.grad, axis=axes_to_reduce[0])
        inputs[0]._tidu = np.reshape(grads, inputs[0].shape)
    if isinstance(inputs[1], Zhangliang) and inputs[1].requires_grad:
        grads = np.sum(outputs_grad.grad, axis=axes_to_reduce[1])
        inputs[1]._tidu = np.reshape(-grads, inputs[1].shape)


@trace(op_name='reduce_mean')
def zl_reduce_mean(a, dim=None, keepdims=False):
    values = np.mean(a.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values)


@grad_register(op_name='sub')
def zl_reduce_mean_grad(input, outputs_grad, **kwargs):
    dim = kwargs.setdefault('dim', None)
    keepdims = kwargs.setdefault('keepdims', False)
    assert len(input) == 1
    inputs_shapes = input.shape
    output_shape = outputs_grad.shape
    new_output_shape = list(output_shape)
    if not keepdims:
        pass
        # TODO: how to find out the reduced axes when `dim` is not assigned.
        # We cannot compare the two shapes, since some values in different but consecutive axis may
        # be the same. It is not possible to identify which axis is the expected one.
        new_output_shape.insert(0, 1)
    if isinstance(input, Zhangliang) and input.requires_grad:
        input.tidu = np.reshape(outputs_grad.tidu, input.shape)


@trace(op_name='reduce_sum')
def zl_reduce_sum(a, dim=None, keepdims=False):
    values = np.sum(a.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values)


@trace(op_name='reshape')
def zl_reshape(a, new_shape):
    a = Zhangliang(a)
    new_value = np.reshape(a.values, new_shape)
    return Zhangliang(new_value)


@trace(op_name='mul')
def zl_mul(a, b):
    if isinstance(a, numbers.Real):
        value = a * b.values
    elif isinstance(b, numbers.Real):
        value = a.values * b
    else:
        value = a.values * b.values
    return Zhangliang(value)


@trace(op_name='rdiv')
def zl_truediv(a, b):
    if isinstance(a, numbers.Real):
        value = a / b.values
    elif isinstance(b, numbers.Real):
        if b == 0:
            raise ValueError('0 cannot be divisor.')
        value = a.values / b
    else:
        value = a.values / b.values
    return Zhangliang(value)


@trace(op_name='matmul')
def zl_matmul(a, b):
    sanity.TypeCheck(a, Zhangliang)
    sanity.TypeCheck(b, Zhangliang)
    return Zhangliang(np.matmul(a.values, b.values))


@trace(op_name='abs')
def zl_abs(a):
    if isinstance(a, numbers.Real):
        value = np.abs(a)
    else:
        value = np.abs(a.values)
    return Zhangliang(value)


@trace(op_name='elt_pow')
def zl_pow(a, power):
    sanity.TypeCheck(a, Zhangliang)
    return Zhangliang(np.power(a.values, power))


@trace(op_name='log')
def zl_log(a, base=np.e):
    sanity.TypeCheck(a, Zhangliang)
    assert base > 0
    value = np.log(a.values) / np.log(base)
    return Zhangliang(value)


@trace(op_name='max')
def zl_max(a, dim=None):
    value = np.max(a.values, axis=dim)
    return Zhangliang(value)


@trace(op_name='maximum')
def zl_maximum(a, b):
    value = np.maximum(a.values, b.values)
    return Zhangliang(value)


@trace(op_name='min')
def zl_max(a, dim=None):
    value = np.min(a.values, axis=dim)
    return Zhangliang(value)


@trace(op_name='minimum')
def zl_maximum(a, b):
    value = np.minimum(a.values, b.values)
    return Zhangliang(value)


@trace(op_name='ge')
def zl_ge(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.values >= b.values
    elif isinstance(a, Zhangliang):
        value = a.values >= b
    elif isinstance(b, Zhangliang):
        value = a >= b.values
    else:
        value = a >= b
    return Zhangliang(value)


@trace(op_name='gt')
def zl_gt(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.values > b.values
    elif isinstance(a, Zhangliang):
        value = a.values > b
    elif isinstance(b, Zhangliang):
        value = a > b.values
    else:
        value = a > b
    return Zhangliang(value)


@trace(op_name='le')
def zl_le(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.values <= b.values
    elif isinstance(a, Zhangliang):
        value = a.values <= b
    elif isinstance(b, Zhangliang):
        value = a <= b.values
    else:
        value = a <= b
    return Zhangliang(value)


@trace(op_name='lt')
def zl_lt(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.values < b.values
    elif isinstance(a, Zhangliang):
        value = a.values < b
    elif isinstance(b, Zhangliang):
        value = a < b.values
    else:
        value = a < b
    return Zhangliang(value)


@trace(op_name='eq')
def zl_eq(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.values == b.values
    elif isinstance(a, Zhangliang):
        value = a.values == b
    elif isinstance(b, Zhangliang):
        value = a == b.values
    else:
        value = a == b
    return Zhangliang(value)


@trace(op_name='ne')
def zl_ne(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = a.values != b.values
    elif isinstance(a, Zhangliang):
        value = a.values != b
    elif isinstance(b, Zhangliang):
        value = a != b.values
    else:
        value = a != b
    return Zhangliang(value)


@trace(op_name='clamp')
def zl_clamp(a, xmin=0., xmax=1.):
    value = np.clip(a.values, a_max=xmax, a_min=xmin)
    return Zhangliang(value)


@trace(op_name='elt_and')
def zl_elt_and(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = np.logical_and(a.values, b.values)
    elif isinstance(a, Zhangliang):
        value = np.logical_and(a.values, b)
    elif isinstance(b, Zhangliang):
        value = np.logical_and(a, b.values)
    else:
        value = np.logical_and(a, b)
    return Zhangliang(value)


@trace(op_name='elt_or')
def zl_elt_or(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = np.logical_or(a.values, b.values)
    elif isinstance(a, Zhangliang):
        value = np.logical_or(a.values, b)
    elif isinstance(b, Zhangliang):
        value = np.logical_or(a, b.values)
    else:
        value = np.logical_or(a, b)
    return Zhangliang(value)


@trace(op_name='elt_not')
def zl_elt_not(a):
    if isinstance(a, Zhangliang):
        value = np.logical_not(a.values)
    else:
        value = np.logical_not(a)
    return Zhangliang(value)


@trace(op_name='elt_xor')
def zl_elt_xor(a, b):
    if isinstance(a, Zhangliang) and \
            isinstance(b, Zhangliang):
        value = np.logical_xor(a.values, b.values)
    elif isinstance(a, Zhangliang):
        value = np.logical_xor(a.values, b)
    elif isinstance(b, Zhangliang):
        value = np.logical_xor(a, b.values)
    else:
        value = np.logical_xor(a, b)
    return Zhangliang(value)


@trace(op_name='sin')
def zl_sin(a):
    if isinstance(a, Zhangliang):
        value = np.sin(a.values)
    else:
        value = np.sin(a)
    return Zhangliang(value)


@trace(op_name='cos')
def zl_cos(a):
    if isinstance(a, Zhangliang):
        value = np.cos(a.values)
    else:
        value = np.cos(a)
    return Zhangliang(value)


@trace(op_name='tan')
def zl_tan(a):
    if isinstance(a, Zhangliang):
        value = np.tan(a.values)
    else:
        value = np.tan(a)
    return Zhangliang(value)


# numpy package has no `cot` function. So we skip `cot`, as well as `arccot`.
@trace(op_name='arcsin')
def zl_arcsin(a):
    if isinstance(a, Zhangliang):
        value = np.arcsin(a.values)
    else:
        value = np.arcsin(a)
    return Zhangliang(value)


@trace(op_name='arccos')
def zl_arccos(a):
    if isinstance(a, Zhangliang):
        value = np.arccos(a.values)
    else:
        value = np.arccos(a)
    return Zhangliang(value)


@trace(op_name='arctan')
def zl_arctan(a):
    if isinstance(a, Zhangliang):
        value = np.arctan(a.values)
    else:
        value = np.arctan(a)
    return Zhangliang(value)


@trace(op_name='sinh')
def zl_sinh(a):
    if isinstance(a, Zhangliang):
        value = np.sinh(a.values)
    else:
        value = np.sinh(a)
    return Zhangliang(value)


@trace(op_name='cosh')
def zl_cosh(a):
    if isinstance(a, Zhangliang):
        value = np.cosh(a.values)
    else:
        value = np.cosh(a)
    return Zhangliang(value)


@trace(op_name='tanh')
def zl_tanh(a):
    if isinstance(a, Zhangliang):
        value = np.tanh(a.values)
    else:
        value = np.tanh(a)
    return Zhangliang(value)


@trace(op_name='arcsinh')
def zl_arcsinh(a):
    if isinstance(a, Zhangliang):
        value = np.arcsinh(a.values)
    else:
        value = np.arcsinh(a)
    return Zhangliang(value)


@trace(op_name='arccosh')
def zl_arccosh(a):
    if isinstance(a, Zhangliang):
        value = np.arccosh(a.values)
    else:
        value = np.arccosh(a)
    return Zhangliang(value)


@trace(op_name='arctanh')
def zl_arctanh(a):
    if isinstance(a, Zhangliang):
        value = np.arctanh(a.values)
    else:
        value = np.arctanh(a)
    return Zhangliang(value)


@trace(op_name='squeeze')
def zl_squeeze(a, dim=None):
    values = np.squeeze(a.values, dim=dim)
    return Zhangliang(values, dtype=a.dtype, requires_grad=a.requires_grad)


@trace(op_name='unsqueeze')
def zl_unsqueeze(a, dim):
    old_shape = a.shape
    new_shape = list(old_shape).insert(dim, 1)
    values = np.reshape(a.values, newshape=new_shape)
    return Zhangliang(values, dtype=a.dtype, requires_grad=a.requires_grad)


@trace(op_name='concat')
def zl_concat(inputs, dim):
    pass


@trace(op_name='stack')
def zl_stack(inputs, dim):
    pass


@trace(op_name='hstack')
def zl_hstack(inputs):
    pass


@trace(op_name='vstack')
def zl_vstack(inputs):
    pass


@trace(op_name='repeat')
def zl_repeat(inputs, repeat_size=None):
    pass


@trace(op_name='reparray')
def zl_reparray(inputs, repeat_size=None):
    pass


if __name__ == '__main__':
    from utils.tracer import graph
    from utils.register import func_lib, grad_lib
    a = Zhangliang([2,3])
    b = Zhangliang([-1,0])

    z1 = a + b + 2
    graph.toposort()
    print(z1)
