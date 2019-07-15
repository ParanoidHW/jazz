from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import collections
import numbers
import numpy as np

from core.base import BaseZhangliang
from utils.tracer import ctx_register, graph
from utils.register import grad_register, func_register

from utils.misc import additive_broadcast_analysis, multiplicative_broadcast_analysis, recover_dim


class Zhangliang(BaseZhangliang):
    def __init__(self, data, dtype=np.float32, requires_grad=False):
        if isinstance(data, Zhangliang):
            data = data.values
        super(Zhangliang, self).__init__(data, dtype, requires_grad)

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


def is_zhangliang_requires_grad(a):
    if isinstance(a, (np.ndarray, numbers.Real)):
        return False
    elif isinstance(a, Zhangliang):
        return a.requires_grad
    else:
        return False


def where(x):
    return np.where(x)


def replace_zeros(x, replace_v):
    x[x == 0] = replace_v
    return x


# ------------ Binary operators-------------


@ctx_register(op_name='add')
def zl_add(a, b):
    local_requires_grad = is_zhangliang_requires_grad(a) or is_zhangliang_requires_grad(b)
    # Incase of non-zhangliang, convert the inputs to Zhangliang
    a_ = Zhangliang.array(a)
    b_ = Zhangliang.array(b)
    value = a_.values + b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='add')
def zl_add_grad(inputs, output):
    assert len(inputs) == 2
    x, y = inputs
    inputs_shapes = tuple([x.shape, y.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        grads = np.sum(output.grad, axis=axes_to_reduce[0])
        x.assign_grad(np.reshape(grads, x.shape))
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = np.sum(output.grad, axis=axes_to_reduce[1])
        y.assign_grad(np.reshape(grads, y.shape))


@ctx_register(op_name='sub')
def zl_sub(a, b):
    local_requires_grad = is_zhangliang_requires_grad(a) or is_zhangliang_requires_grad(b)
    a_ = Zhangliang.array(a)
    b_ = Zhangliang.array(b)
    value = a_.values - b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='sub')
def zl_sub_grad(inputs, output):
    assert len(inputs) == 2
    x, y = inputs
    inputs_shapes = tuple([x.shape, y.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        grads = np.sum(output.grad, axis=axes_to_reduce[0])
        x.assign_grad(np.reshape(grads, x.shape))
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = np.sum(output.grad, axis=axes_to_reduce[1])
        y.assign_grad(np.reshape(-grads, y.shape))


@ctx_register(op_name='mul')
def zl_mul(a, b):
    local_requires_grad = is_zhangliang_requires_grad(a) or is_zhangliang_requires_grad(b)
    a_ = Zhangliang.array(a)
    b_ = Zhangliang.array(b)
    value = a_.values * b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='mul')
def zl_mul_grad(inputs, output):
    assert len(inputs) == 2
    x, y = inputs
    inputs_shapes = tuple([x.shape, y.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        # The output definitely can broadcast with each input.
        grads = output.grad * y.values
        grads = np.sum(grads, axis=axes_to_reduce[0])
        y.assign_grad(np.reshape(grads, x.shape))
    if isinstance(inputs[1], Zhangliang) and y.requires_grad:
        grads = output.grad * x.values
        grads = np.sum(grads, axis=axes_to_reduce[0])
        y.assign_grad(np.reshape(grads, y.shape))


@ctx_register(op_name='rdiv')
def zl_truediv(a, b):
    local_requires_grad = is_zhangliang_requires_grad(a) or is_zhangliang_requires_grad(b)
    a_ = Zhangliang.array(a)
    b_ = Zhangliang.array(b)
    value = a_.values / b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='rdiv')
def zl_truediv_grad(inputs, output):
    assert len(inputs) == 2
    x, y = inputs
    inputs_shapes = tuple([x.shape, y.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        # The output definitely can broadcast with each input.
        grads = output.grad / y.values
        grads = np.sum(grads, axis=axes_to_reduce[0])
        x.assign_grad(np.reshape(grads, x.shape))
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = - output.grad * x.values / (y.values ** 2)
        grads = np.sum(grads, axis=axes_to_reduce[0])
        y.assign_grad(np.reshape(grads, y.shape))


@ctx_register(op_name='matmul')
def zl_matmul(a, b):
    local_requires_grad = is_zhangliang_requires_grad(a) or is_zhangliang_requires_grad(b)
    a_ = Zhangliang.array(a)
    b_ = Zhangliang.array(b)
    values = np.matmul(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='matmul')
def zl_matmul_grad(inputs, output):
    assert len(inputs) == 2
    x, y = inputs
    inputs_shapes = tuple([x.shape, y.shape])
    output_shape = output.shape
    axes_to_reduce = multiplicative_broadcast_analysis(inputs_shapes, output_shape)

    a_dim, b_dim = x.shape, y.shape

    # In case of (m, n) X (n, ) = (m, ).
    # (m, ) X (1, n) is impossible in forward mode. So maybe only inputs[1] needs to be checked.
    if len(b_dim) == 1:
        b_transposed = y.values[np.newaxis, :]
        output_grad = output.grad[..., np.newaxis]
    else:
        b_transposed = np.swapaxes(y.values, -1, -2)
        output_grad = output.grad

    a_grad = np.matmul(output_grad, b_transposed)
    x.assign_grad(np.sum(a_grad, axis=axes_to_reduce[0]))

    a_transposed = np.swapaxes(x.values, -1, -2)
    b_grad = np.matmul(a_transposed, output_grad)
    y.assign_grad(np.sum(b_grad, axis=axes_to_reduce[1]))


@ctx_register(op_name='pow')
def zl_pow(a, b):
    local_requires_grad = is_zhangliang_requires_grad(a) or is_zhangliang_requires_grad(b)
    a_ = Zhangliang(a)
    b_ = Zhangliang(b)
    values = np.power(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='pow')
def zl_pow_grad(inputs, output):
    assert len(inputs) == 2
    x, y = inputs
    inputs_shapes = tuple([x.shape, y.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        # The output definitely can broadcast with each input.
        power = np.where(y.values, y.values-1, 1.)
        grads = output.grad * y.values * np.power(x.values, power)
        grads = np.sum(grads, axis=axes_to_reduce[0])
        x.assign_grad(np.reshape(grads, x.shape))
    if isinstance(y, Zhangliang) and y.requires_grad:
        coef = np.log(np.where(x.values, x.values, 1.))
        grads = output.grad * np.power(x.values, y.values) * coef
        grads = np.sum(grads, axis=axes_to_reduce[1])
        y.assign_grad(np.reshape(grads, inputs[1].shape))


# Borrowed from
# https://github.com/HIPS/autograd/blob/387c373115ddd54cff2c8ba6a9fc619f28639cfb/autograd/numpy/numpy_vjps.py#L672
def balanced_eq(xin, yin, zout):
    return (xin == zout) / (1. + (xin == yin))


@ctx_register(op_name='maximum')
def zl_maximum(a, b):
    local_requires_grad = is_zhangliang_requires_grad(a) or is_zhangliang_requires_grad(b)
    a_ = Zhangliang.array(a)
    b_ = Zhangliang.array(b)
    values = np.maximum(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='maximum')
def zl_maximum_grad(inputs, output):
    assert len(inputs) == 2
    x, y = inputs
    if isinstance(x, Zhangliang) and x.requires_grad:
        x_grad = output.grad * balanced_eq(x.values, y.values, output.values)
        x.assign_grad(x_grad)
    if isinstance(y, Zhangliang) and y.requires_grad:
        y_grad = output.grad * balanced_eq(y.values, x.values, output.values)
        y.assign_grad(y_grad)


@ctx_register(op_name='minimum')
def zl_minimum(a, b):
    local_requires_grad = is_zhangliang_requires_grad(a) or is_zhangliang_requires_grad(b)
    a_ = Zhangliang.array(a)
    b_ = Zhangliang.array(b)
    values = np.minimum(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='minimum')
def zl_minimum_grad(inputs, output):
    assert len(inputs) == 2
    x, y = inputs
    if isinstance(x, Zhangliang) and x.requires_grad:
        x_grad = output.grad * balanced_eq(x.values, y.values, output.values)
        x.assign_grad(x_grad)
    if isinstance(y, Zhangliang) and y.requires_grad:
        y_grad = output.grad * balanced_eq(y.values, x.values, output.values)
        y.assign_grad(y_grad)


# Compare functions cannot backprop gradients. No need to trace them.
@func_register(op_name='ge')
def zl_ge(a, b):
    a_ = Zhangliang(a)
    b_ = Zhangliang(b)
    values = np.greater_equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='ge')
def zl_ge_grad(inputs, output):
    # The output of `ge` function does not require grad. So pass.
    pass


@func_register(op_name='gt')
def zl_gt(a, b):
    a_ = Zhangliang(a)
    b_ = Zhangliang(b)
    values = np.greater(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='gt')
def zl_gt_grad(inputs, output):
    # The output of `gt` function does not require grad. So pass.
    pass


@func_register(op_name='le')
def zl_le(a, b):
    a_ = Zhangliang(a)
    b_ = Zhangliang(b)
    values = np.less_equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='le')
def zl_le_grad(inputs, output):
    # The output of `le` function does not require grad. So pass.
    pass


@func_register(op_name='lt')
def zl_lt(a, b):
    a_ = Zhangliang(a)
    b_ = Zhangliang(b)
    values = np.less(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='lt')
def zl_lt_grad(inputs, output):
    # The output of `lt` function does not require grad. So pass.
    pass


@func_register(op_name='eq')
def zl_eq(a, b):
    a_ = Zhangliang(a)
    b_ = Zhangliang(b)
    values = np.equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='eq')
def zl_eq_grad(inputs, output):
    pass


@func_register(op_name='ne')
def zl_ne(a, b):
    a_ = Zhangliang(a)
    b_ = Zhangliang(b)
    values = np.not_equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='ne')
def zl_ne_grad(inputs, output):
    pass


@func_register(op_name='elt_and')
def zl_elt_and(a, b):
    a_ = Zhangliang(a)
    b_ = Zhangliang(b)
    values = np.logical_and(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_and')
def zl_elt_and_grad(inputs, output):
    pass


@func_register(op_name='elt_or')
def zl_elt_or(a, b):
    a_ = Zhangliang(a)
    b_ = Zhangliang(b)
    values = np.logical_or(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_or')
def zl_elt_or_grad(inputs, output):
    pass


@func_register(op_name='elt_xor')
def zl_elt_xor(a, b):
    a_ = Zhangliang(a)
    b_ = Zhangliang(b)
    values = np.logical_xor(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_xor')
def zl_elt_xor_grad(inputs, output):
    pass


# --------------- Unary operators ---------------


@ctx_register(op_name='reduce_mean')
def zl_reduce_mean(a, dim=None, keepdims=False):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.mean(a_.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='reduce_mean')
def zl_reduce_mean_grad(input, output, **kwargs):
    dim = kwargs.get('dim', None)
    assert len(input) == 1
    inputs_shapes = input.shape
    reduced_shapes = list(inputs_shapes)

    if dim is None:
        dim = list(range(input.ndim))
    reduced_scale = 1
    for i in dim:
        reduced_scale *= inputs_shapes[i]
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        reduced_shapes[i] = 1

    if isinstance(input, Zhangliang) and input.requires_grad:
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        input.assign_grad(np.reshape(output.grad / reduced_scale, reduced_shapes))


@ctx_register(op_name='reduce_sum')
def zl_reduce_sum(a, dim=None, keepdims=False):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.sum(a_.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='reduce_sum')
def zl_reduce_sum_grad(input, output, **kwargs):
    dim = kwargs.get('dim', None)
    assert len(input) == 1
    inputs_shapes = input.shape
    reduced_shapes = list(inputs_shapes)
    if dim is None:
        dim = list(range(input.ndim))

    for i in dim:
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        reduced_shapes[i] = 1

    if isinstance(input, Zhangliang) and input.requires_grad:
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        input.assign_grad(np.reshape(output.grad, reduced_shapes))


@ctx_register(op_name='reshape')
def zl_reshape(a, new_shape):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.reshape(a_.values, new_shape)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='reshape')
def zl_reshape_grad(input, output):
    old_shape = input.shape
    if isinstance(input, Zhangliang) and input.requires_grad:
        input.assign_grad(np.reshape(output.grad, old_shape))


@ctx_register(op_name='abs')
def zl_abs(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.abs(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='abs')
def zl_abs_grad(input, output):
    values = np.where(input.values, input.values, 0.)
    if isinstance(input, Zhangliang) and input.requires_grad:
        input.assign_grad(output.grad * values)


@ctx_register(op_name='log')
def zl_log(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.log(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='log')
def zl_log_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad / input.values
        input.assign_grad(values)


@ctx_register(op_name='log2')
def zl_log2(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.log2(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='log2')
def zl_log2_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad / input.values / np.log(2)
        input.assign_grad(values)


@ctx_register(op_name='log10')
def zl_log10(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.log10(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='log10')
def zl_log10_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad / input.values / np.log(10)
        input.assign_grad(values)


@ctx_register(op_name='log1p')
def zl_log1p(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.log1p(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='log1p')
def zl_log1p_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad / (1. + input.values)
        input.assign_grad(values)


def grad_minmax(xin, zout, grad, dim=None, keepdims=False):
    new_shape = recover_dim(xin.shape, zout.shape, dim=dim, keepdims=keepdims)
    zout = np.reshape(zout, newshape=new_shape)
    max_value_map = xin == zout
    nmax = np.sum(max_value_map, axis=dim)
    values = grad * max_value_map / nmax
    return values


@ctx_register(op_name='max')
def zl_max(a, dim=None, keepdims=False):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.max(a_.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='max')
def zl_max_grad(input, output, **kwargs):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = grad_minmax(input.values, output.values, output.grad,
                             dim=kwargs.get('dim', None), keepdims=kwargs.get('keepdims', False))
        input.assign_grad(values)


@ctx_register(op_name='min')
def zl_min(a, dim=None):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.min(a_.values, axis=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='min')
def zl_min_grad(input, output, **kwargs):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = grad_minmax(input.values, output.values, output.grad,
                             dim=kwargs.get('dim', None), keepdims=kwargs.get('keepdims', False))
        input.assign_grad(values)


@ctx_register(op_name='argmax')
def zl_argmax(a, dim=None):
    a_ = Zhangliang(a)
    values = np.argmax(a_.values, axis=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='argmax')
def zl_argmax_grad(input, output, **kwargs):
    pass


@ctx_register(op_name='argmin')
def zl_argmin(a, dim=None):
    a_ = Zhangliang(a)
    values = np.min(a_.values, axis=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='argmin')
def zl_argmin_grad(input, output, **kwargs):
    pass


@ctx_register(op_name='clamp')
def zl_clamp(a, xmin=0., xmax=1.):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.clip(a_.values, a_max=xmax, a_min=xmin)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='clamp')
def zl_clamp_grad(input, output, **kwargs):
    xmin = kwargs.get('xmin', 0.)
    xmax = kwargs.get('xmax', 1.)
    if isinstance(input, Zhangliang) and input.requires_grad:
        valid_region = np.logical_and(input.values != xmin, input.values != xmax)
        values = output.grad * valid_region
        input.assign_grad(values)


@func_register(op_name='elt_not')
def zl_elt_not(a):
    a_ = Zhangliang(a)
    values = np.logical_not(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_not')
def zl_elt_not_grad(inputs, output):
    pass


@ctx_register(op_name='sin')
def zl_sin(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.sin(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='sin')
def zl_sin_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad * np.cos(input.values)
        input.assign_grad(values)


@ctx_register(op_name='cos')
def zl_cos(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.cos(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='cos')
def zl_cos_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = - output.grad * np.sin(input.values)
        input.assign_grad(values)


@ctx_register(op_name='tan')
def zl_tan(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.tan(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='tan')
def zl_tan_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad / (np.cos(input.values) ** 2)
        input.assign_grad(values)


# numpy package has no `cot` function. So we skip `cot`, as well as `arccot`.
@ctx_register(op_name='arcsin')
def zl_arcsin(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.arcsin(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='arcsin')
def zl_arcsin_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad / (np.sqrt(1 - input.values ** 2))
        input.assign_grad(values)


@ctx_register(op_name='arccos')
def zl_arccos(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.arccos(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='arccos')
def zl_arccos_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = - output.grad / (np.sqrt(1 - input.values ** 2))
        input.assign_grad(values)


@ctx_register(op_name='arctan')
def zl_arctan(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.arctan(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='arctan')
def zl_arctan_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad / (1 + input.values ** 2)
        input.assign_grad(values)


@ctx_register(op_name='sinh')
def zl_sinh(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.sinh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='sinh')
def zl_sinh_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad * np.cosh(a_.values)
        input.assign_grad(values)


@ctx_register(op_name='cosh')
def zl_cosh(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.cosh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='cosh')
def zl_cosh_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad * np.sinh(a_.values)
        input.assign_grad(values)


@ctx_register(op_name='tanh')
def zl_tanh(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.tanh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='tanh')
def zl_tanh_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad / (np.cosh(a_.values) ** 2)
        input.assign_grad(values)


@ctx_register(op_name='arcsinh')
def zl_arcsinh(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.arcsinh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='arcsinh')
def zl_arcsinh_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad / np.sqrt(a_.values ** 2 + 1)
        input.assign_grad(values)


@ctx_register(op_name='arccosh')
def zl_arccosh(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.arccosh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='arccosh')
def zl_arccosh_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad / np.sqrt(a_.values ** 2 - 1)
        input.assign_grad(values)


@ctx_register(op_name='arctanh')
def zl_arctanh(a):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.arctanh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='arctanh')
def zl_arctanh_grad(input, output):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = output.grad / (1. - a_.values ** 2)
        input.assign_grad(values)


@ctx_register(op_name='squeeze')
def zl_squeeze(a, dim=None):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    values = np.squeeze(a_.values, dim=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='squeeze')
def zl_squeeze_grad(input, output, **kwargs):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = np.reshape(output.grad, input.shape)
        input.assign_grad(values)


@ctx_register(op_name='unsqueeze')
def zl_unsqueeze(a, dim):
    local_requires_grad = is_zhangliang_requires_grad(a)
    a_ = Zhangliang(a)
    old_shape = a.shape
    new_shape = list(old_shape)
    new_shape.insert(dim, 1)
    values = np.reshape(a_.values, newshape=new_shape)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enable())


@grad_register(op_name='unsqueeze')
def zl_unsqueeze_grad(input, output, **kwargs):
    if isinstance(input, Zhangliang) and input.requires_grad:
        values = np.reshape(output.grad, input.shape)
        input.assign_grad(values)


@ctx_register(op_name='concat')
def zl_concat(inputs, dim):
    local_requires_grad = is_zhangliang_requires_grad(a)
    pass


@ctx_register(op_name='stack')
def zl_stack(inputs, dim):
    pass


@ctx_register(op_name='hstack')
def zl_hstack(inputs):
    pass


@ctx_register(op_name='vstack')
def zl_vstack(inputs):
    pass


@ctx_register(op_name='repeat')
def zl_repeat(inputs, repeat_size=None):
    pass


@ctx_register(op_name='reparray')
def zl_reparray(inputs, repeat_size=None):
    pass


if __name__ == '__main__':
    from utils.tracer import graph
    from utils.register import func_lib, grad_lib
    a = Zhangliang([2,3])
    a_ = Zhangliang(a)
    b = Zhangliang([-1,0])

    z1 = a + b + 2
    graph.toposort()
    print(z1)
