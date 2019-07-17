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
    def __init__(self, data, dtype=np.float64, requires_grad=False):
        if isinstance(data, Zhangliang):
            data = data.values
        elif np.isscalar(data):
            data = [data]
        super(Zhangliang, self).__init__(data, dtype, requires_grad)

    @classmethod
    def zeros(cls, shape, dtype=np.float64, requires_grad=False):
        zeros_ = np.zeros(shape, dtype=dtype)
        return cls(zeros_, requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape, dtype=np.float64, requires_grad=False):
        ones_ = np.ones(shape, dtype=dtype)
        return cls(ones_, requires_grad=requires_grad)

    @classmethod
    def zeros_like(cls, data, dtype=np.float64, requires_grad=False):
        shape = data.shape
        zeros_ = np.zeros(shape, dtype=dtype)
        return cls(zeros_, requires_grad=requires_grad)

    @classmethod
    def ones_like(cls, data, dtype=np.float64, requires_grad=False):
        shape = data.shapes
        ones_ = np.ones(shape, dtype=dtype)
        return cls(ones_, requires_grad=requires_grad)

    @classmethod
    def array(cls, data, requires_grad=False):
        if isinstance(data, Zhangliang):
            return cls(data.values, dtype=data.dtype, requires_grad=requires_grad)
        elif np.isscalar(data):
            return cls([data], dtype=np.int32, requires_grad=requires_grad)
        elif isinstance(data, (list, tuple)):
            return cls(data, dtype=np.float64, requires_grad=requires_grad)
        elif isinstance(data, collections.Iterable):
            data = np.array(data)
            return cls(data, dtype=np.float64, requires_grad=requires_grad)
        else:
            raise TypeError

    @classmethod
    def linspace(cls, start, stop, num):
        data = np.linspace(start, stop, num)
        return cls(data, dtype=data.dtype, requires_grad=False)

    @classmethod
    def arange(cls, start, stop=None, step=1):
        if stop is None:
            stop = start
            start = 0
        data = np.arange(start, stop, step)
        return cls(data, dtype=data.dtype, requires_grad=False)

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
        return zl_neg(self)

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


def is_zhangliang_requires_grad(x):
    if isinstance(x, (np.ndarray, numbers.Real)):
        return False
    elif isinstance(x, Zhangliang):
        return x.requires_grad
    else:
        return False


def aggregate_and_reshape_grad(grad_values, axes_to_reduce, target_shape):
    if len(axes_to_reduce) >= 0:
        aggregated_grad = np.sum(grad_values, axis=axes_to_reduce)
    else:
        aggregated_grad = grad_values
    return np.reshape(aggregated_grad, target_shape)


# ---------------------------------------------------------- #
# math ops for Zhangliang
# ---------------------------------------------------------- #

# ------------ Binary operators-------------


@ctx_register(op_name='add')
def zl_add(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    # Incase of non-zhangliang, convert the inputs to Zhangliang
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    value = a_.values + b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='add')
def zl_add_grad(output, x, y):
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        grads = aggregate_and_reshape_grad(output.grad, axes_to_reduce[0], x_.shape)
        x.assign_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = aggregate_and_reshape_grad(output.grad, axes_to_reduce[1], y_.shape)
        y.assign_grad(grads)


@ctx_register(op_name='sub')
def zl_sub(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    value = a_.values - b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='sub')
def zl_sub_grad(output, x, y):
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        grads = aggregate_and_reshape_grad(output.grad, axes_to_reduce[0], x_.shape)
        x.assign_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = aggregate_and_reshape_grad(-output.grad, axes_to_reduce[1], y_.shape)
        y.assign_grad(grads)


@ctx_register(op_name='mul')
def zl_mul(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    value = a_.values * b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='mul')
def zl_mul_grad(output, x, y):
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        # The output definitely can broadcast with each input.
        grads = output.grad * y.values
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[0], x_.shape)
        x.assign_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = output.grad * x.values
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[1], y_.shape)
        y.assign_grad(grads)


@ctx_register(op_name='div')
def zl_truediv(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    value = a_.values / b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='div')
def zl_truediv_grad(output, x, y):
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        # The output definitely can broadcast with each input.
        grads = output.grad / y.values
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[0], x_.shape)
        x.assign_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = - output.grad * x.values / (y.values ** 2)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[1], y_.shape)
        y.assign_grad(np.reshape(grads, y.shape))


@ctx_register(op_name='matmul')
def zl_matmul(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    values = np.matmul(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='matmul')
def zl_matmul_grad(output, x, y):
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


# TODO: `Zhangliang` does not seem to support for `complex` data type.
# So the inputs of the `pow` should be positive.
@ctx_register(op_name='pow')
def zl_pow(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.power(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='pow')
def zl_pow_grad(output, x, y):
    inputs_shapes = tuple([x.shape, y.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        # The output definitely can broadcast with each input.
        power = np.where(y.values, y.values-1, 1.)
        grads = output.grad * y.values * np.power(x.values, power)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[0], x.shape)
        x.assign_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        coef = np.log(np.where(x.values, x.values, 1.))
        grads = output.grad * np.power(x.values, y.values) * coef
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[1], y.shape)
        y.assign_grad(grads)


# Borrowed from
# https://github.com/HIPS/autograd/blob/387c373115ddd54cff2c8ba6a9fc619f28639cfb/autograd/numpy/numpy_vjps.py#L672
def balanced_eq(xin, yin, zout):
    return (xin == zout) / (1. + (xin == yin))


@ctx_register(op_name='maximum')
def zl_maximum(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.maximum(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='maximum')
def zl_maximum_grad(output, x, y):
    inputs_shapes = tuple([x.shape, y.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        grads = output.grad * balanced_eq(x.values, y.values, output.values)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[0], x.shape)
        x.assign_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = output.grad * balanced_eq(y.values, x.values, output.values)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[1], y.shape)
        y.assign_grad(grads)


@ctx_register(op_name='minimum')
def zl_minimum(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    values = np.minimum(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='minimum')
def zl_minimum_grad(output, x, y):
    inputs_shapes = tuple([x.shape, y.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        grads = output.grad * balanced_eq(x.values, y.values, output.values)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[0], x.shape)
        x.assign_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = output.grad * balanced_eq(y.values, x.values, output.values)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[1], y.shape)
        y.assign_grad(grads)


# Compare functions cannot backprop gradients. No need to trace them.
@func_register(op_name='ge')
def zl_ge(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.greater_equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='ge')
def zl_ge_grad(output, x, y):
    # The output of `ge` function does not require grad. So pass.
    pass


@func_register(op_name='gt')
def zl_gt(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.greater(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='gt')
def zl_gt_grad(output, x, y):
    # The output of `gt` function does not require grad. So pass.
    pass


@func_register(op_name='le')
def zl_le(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.less_equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='le')
def zl_le_grad(output, x, y):
    # The output of `le` function does not require grad. So pass.
    pass


@func_register(op_name='lt')
def zl_lt(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.less(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='lt')
def zl_lt_grad(output, x, y):
    # The output of `lt` function does not require grad. So pass.
    pass


@func_register(op_name='eq')
def zl_eq(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='eq')
def zl_eq_grad(output, x, y):
    pass


@func_register(op_name='ne')
def zl_ne(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.not_equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='ne')
def zl_ne_grad(output, x, y):
    pass


@func_register(op_name='elt_and')
def zl_elt_and(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.logical_and(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_and')
def zl_elt_and_grad(output, x, y):
    pass


@func_register(op_name='elt_or')
def zl_elt_or(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.logical_or(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_or')
def zl_elt_or_grad(output, x, y):
    pass


@func_register(op_name='elt_xor')
def zl_elt_xor(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.logical_xor(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_xor')
def zl_elt_xor_grad(output, x, y):
    pass


# --------------- Unary operators ---------------


@ctx_register(op_name='neg')
def zl_neg(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = - a_.values
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='neg')
def zl_neg_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        x.assign_grad(-output.grad)


@ctx_register(op_name='reduce_mean')
def zl_reduce_mean(x, dim=None, keepdims=False):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.mean(a_.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='reduce_mean')
def zl_reduce_mean_grad(output, x, dim=None, keepdims=False):
    inputs_shapes = x.shape
    reduced_shapes = list(inputs_shapes)

    if dim is None:
        dim = list(range(x.ndim))
    reduced_scale = 1
    for i in dim:
        reduced_scale *= inputs_shapes[i]
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        reduced_shapes[i] = 1

    if isinstance(x, Zhangliang) and x.requires_grad:
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        x.assign_grad(np.reshape(output.grad / reduced_scale, reduced_shapes))


@ctx_register(op_name='reduce_sum')
def zl_reduce_sum(x, dim=None, keepdims=False):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.sum(a_.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='reduce_sum')
def zl_reduce_sum_grad(output, x, dim=None, keepdims=False):
    inputs_shapes = x.shape
    reduced_shapes = list(inputs_shapes)
    if dim is None:
        dim = list(range(x.ndim))

    for i in dim:
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        reduced_shapes[i] = 1

    if isinstance(x, Zhangliang) and x.requires_grad:
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        x.assign_grad(np.reshape(output.grad, reduced_shapes))


@ctx_register(op_name='reshape')
def zl_reshape(x, new_shape):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.reshape(a_.values, new_shape)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='reshape')
def zl_reshape_grad(output, x, new_shape):
    old_shape = x.shape
    if isinstance(x, Zhangliang) and x.requires_grad:
        x.assign_grad(np.reshape(output.grad, old_shape))


@ctx_register(op_name='abs')
def zl_abs(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.abs(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='abs')
def zl_abs_grad(output, x):
    values = np.where(x.values > 0, 1., -1.)
    values = np.where(x.values, values, 0.)
    if isinstance(x, Zhangliang) and x.requires_grad:
        x.assign_grad(output.grad * values)


@ctx_register(op_name='log')
def zl_log(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.log(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='log')
def zl_log_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / x.values
        x.assign_grad(values)


@ctx_register(op_name='log2')
def zl_log2(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.log2(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='log2')
def zl_log2_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / x.values / np.log(2)
        x.assign_grad(values)


@ctx_register(op_name='log10')
def zl_log10(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.log10(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='log10')
def zl_log10_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / x.values / np.log(10)
        x.assign_grad(values)


@ctx_register(op_name='log1p')
def zl_log1p(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.log1p(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='log1p')
def zl_log1p_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (1. + x.values)
        x.assign_grad(values)


def grad_minmax(xin, zout, grad, dim=None, keepdims=False):
    inputs_shapes = xin.shape
    reduced_shapes = list(inputs_shapes)
    if dim is None:
        dim = list(range(xin.ndim))

    for i in dim:
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        reduced_shapes[i] = 1

    zout = np.reshape(zout, newshape=reduced_shapes)
    max_value_map = xin == zout
    nmax = np.sum(max_value_map, axis=tuple(dim))
    values = grad * max_value_map / nmax
    return values


@ctx_register(op_name='max')
def zl_max(x, dim=None, keepdims=False):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.max(a_.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='max')
def zl_max_grad(output, x, dim=None, keepdims=False):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = grad_minmax(x.values, output.values, output.grad,
                             dim=dim, keepdims=keepdims)
        x.assign_grad(values)


@ctx_register(op_name='min')
def zl_min(x, dim=None, keepdims=False):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.min(a_.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='min')
def zl_min_grad(output, x, dim=None, keepdims=False):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = grad_minmax(x.values, output.values, output.grad,
                             dim=dim, keepdims=keepdims)
        x.assign_grad(values)


@ctx_register(op_name='argmax')
def zl_argmax(x, dim=None):
    a_ = Zhangliang(x)
    values = np.argmax(a_.values, axis=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='argmax')
def zl_argmax_grad(output, x, dim=None):
    pass


@ctx_register(op_name='argmin')
def zl_argmin(x, dim=None):
    a_ = Zhangliang(x)
    values = np.argmin(a_.values, axis=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='argmin')
def zl_argmin_grad(output, x, dim=None):
    pass


@ctx_register(op_name='clamp')
def zl_clamp(x, xmin=0., xmax=1.):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.clip(a_.values, a_max=xmax, a_min=xmin)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='clamp')
def zl_clamp_grad(output, x, xmin=0., xmax=1.):
    if isinstance(x, Zhangliang) and x.requires_grad:
        valid_region = np.logical_and(x.values >= xmin, x.values <= xmax)
        values = output.grad * valid_region
        x.assign_grad(values)


@func_register(op_name='elt_not')
def zl_elt_not(x):
    a_ = Zhangliang(x)
    values = np.logical_not(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_not')
def zl_elt_not_grad(output, x):
    pass


@ctx_register(op_name='sin')
def zl_sin(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.sin(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='sin')
def zl_sin_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad * np.cos(x.values)
        x.assign_grad(values)


@ctx_register(op_name='cos')
def zl_cos(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.cos(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='cos')
def zl_cos_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = - output.grad * np.sin(x.values)
        x.assign_grad(values)


@ctx_register(op_name='tan')
def zl_tan(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.tan(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='tan')
def zl_tan_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (np.cos(x.values) ** 2)
        x.assign_grad(values)


# numpy package has no `cot` function. So we skip `cot`, as well as `arccot`.
@ctx_register(op_name='arcsin')
def zl_arcsin(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arcsin(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arcsin')
def zl_arcsin_grad(output, x):
    # TODO: the error becomes significant when the `x.values` are close to -1 and 1.
    # How to make it stable?
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (np.sqrt(1 - x.values ** 2))
        x.assign_grad(values)


@ctx_register(op_name='arccos')
def zl_arccos(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arccos(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arccos')
def zl_arccos_grad(output, x):
    # TODO: the error becomes significant when the `x.values` are close to -1 and 1.
    # How to make it stable?
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = - output.grad / (np.sqrt(1 - x.values ** 2))
        x.assign_grad(values)


@ctx_register(op_name='arctan')
def zl_arctan(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arctan(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arctan')
def zl_arctan_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (1 + x.values ** 2)
        x.assign_grad(values)


@ctx_register(op_name='sinh')
def zl_sinh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.sinh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='sinh')
def zl_sinh_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad * np.cosh(x.values)
        x.assign_grad(values)


@ctx_register(op_name='cosh')
def zl_cosh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.cosh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='cosh')
def zl_cosh_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad * np.sinh(x.values)
        x.assign_grad(values)


@ctx_register(op_name='tanh')
def zl_tanh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.tanh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='tanh')
def zl_tanh_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (np.cosh(x.values) ** 2)
        x.assign_grad(values)


@ctx_register(op_name='arcsinh')
def zl_arcsinh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arcsinh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arcsinh')
def zl_arcsinh_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / np.sqrt(x.values ** 2 + 1)
        x.assign_grad(values)


@ctx_register(op_name='arccosh')
def zl_arccosh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arccosh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arccosh')
def zl_arccosh_grad(output, x):
    # TODO: the error becomes significant when the `x.values` are close to 1.
    # How to make it stable?
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / np.sqrt(x.values ** 2 - 1)
        x.assign_grad(values)


@ctx_register(op_name='arctanh')
def zl_arctanh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arctanh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arctanh')
def zl_arctanh_grad(output, x):
    # TODO: the error becomes significant when the `x.values` are close to -1 and 1.
    # How to make it stable?
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (1. - x.values ** 2)
        x.assign_grad(values)


@ctx_register(op_name='squeeze')
def zl_squeeze(x, dim=None):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.squeeze(a_.values, dim=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='squeeze')
def zl_squeeze_grad(output, x, dim=None):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = np.reshape(output.grad, x.shape)
        x.assign_grad(values)


@ctx_register(op_name='unsqueeze')
def zl_unsqueeze(x, dim):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    old_shape = x.shape
    new_shape = list(old_shape)
    new_shape.insert(dim, 1)
    values = np.reshape(a_.values, newshape=new_shape)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='unsqueeze')
def zl_unsqueeze_grad(output, x, dim):
    if isinstance(input, Zhangliang) and x.requires_grad:
        values = np.reshape(output.grad, x.shape)
        x.assign_grad(values)


# ---------------- array-like functions -----------------


@ctx_register(op_name='concat')
def zl_concat(inputs, dim=-1):
    local_requires_grad = False
    encap_inputs = []
    for a_input in inputs:
        local_requires_grad = local_requires_grad or is_zhangliang_requires_grad(a_input)
        encap_inputs.append(Zhangliang(a_input).values)

    values = np.concatenate(encap_inputs, axis=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='concat')
def zl_concat_grad(output, inputs, dim=-1):
    nsize = []
    for a_input in inputs:
        if not isinstance(a_input, Zhangliang):
            raise TypeError
        nsize.append(a_input.shape[dim])

    split_grads = np.split(output.grad, nsize, axis=dim)
    for i, a_input in enumerate(inputs):
        if a_input.requires_grad:
            a_input.assign_grad(split_grads[i])


@ctx_register(op_name='hstack')
def zl_hstack(inputs):
    local_requires_grad = False
    encap_inputs = []
    for a_input in inputs:
        local_requires_grad = local_requires_grad or is_zhangliang_requires_grad(a_input)
        encap_inputs.append(Zhangliang(a_input).values)

    values = np.concatenate(encap_inputs, axis=1)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='hstack')
def zl_hstack_grad(output, inputs):
    nsize = []
    for a_input in inputs:
        if not isinstance(a_input, Zhangliang):
            raise TypeError
        nsize.append(a_input.shape[1])

    split_grads = np.split(output.grad, nsize, axis=1)
    for i, a_input in enumerate(inputs):
        if a_input.requires_grad:
            a_input.assign_grad(split_grads[i])


@ctx_register(op_name='vstack')
def zl_vstack(inputs):
    local_requires_grad = False
    encap_inputs = []
    for a_input in inputs:
        local_requires_grad = local_requires_grad or is_zhangliang_requires_grad(a_input)
        encap_inputs.append(Zhangliang(a_input).values)

    values = np.concatenate(encap_inputs, axis=0)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='vstack')
def zl_vstack_grad(output, inputs):
    nsize = []
    for a_input in inputs:
        if not isinstance(a_input, Zhangliang):
            raise TypeError
        nsize.append(a_input.shape[0])

    split_grads = np.split(output.grad, nsize, axis=0)
    for i, a_input in enumerate(inputs):
        if a_input.requires_grad:
            a_input.assign_grad(split_grads[i])


@ctx_register(op_name='tile')
def zl_tile(x, reps):
    local_requires_grad = is_zhangliang_requires_grad(x)
    values = np.tile(x.values, reps)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='tile')
def zl_tile_grad(output, x, reps):
    xdim = x.ndim
    reps = [reps] if np.isscalar(reps) else reps
    d = len(reps)
    if d < xdim:
        reps = list([1] * (xdim - d)) + reps
    g = output.grad
    for axis, rep in enumerate(reps):
        g = sum(np.split(g, rep, axis))
    x.assign_grad(g)


if __name__ == '__main__':
    from utils.tracer import graph
    from utils.register import func_lib, grad_lib
    a = Zhangliang([2, 3])
    a_ = Zhangliang(a)
    b = Zhangliang([-1, 0])
    c = -a

    z1 = a + b + 2
    graph.toposort()
    print(z1)
