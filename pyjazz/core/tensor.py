from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import collections
import numbers
import numpy as np

from pyjazz.core.base import BaseZhangliang
from pyjazz.utils.tracer import ctx_register, graph
from pyjazz.utils.register import grad_register, func_register

from pyjazz.utils.misc import additive_broadcast_analysis, multiplicative_broadcast_analysis


class Zhangliang(BaseZhangliang):
    def __init__(self, data, dtype=np.float64, requires_grad=False):
        if isinstance(data, Zhangliang):
            data = data.values
        # elif np.isscalar(data):
        #     data = [data]
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

    @classmethod
    def rand(cls, size, dtype=np.float64, requires_grad=False):
        return cls(np.random.rand(*size), dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def randint(cls, low, dtype=np.float64, requires_grad=False, **kwargs):
        return cls(np.random.randint(low, **kwargs), dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def randn(cls, size, dtype=np.float64, requires_grad=False):
        return cls(np.random.randn(*size), dtype=dtype, requires_grad=requires_grad)

    def backward(self, retain_graph=False):
        if not graph.is_initialized():
            graph.toposort()

        if graph.is_leaf(self) and self.requires_grad:
            self.update_grad(1.)
        elif graph.is_leaf(self) and (not self.requires_grad):
            raise AttributeError('Zhangliang does not requires grad.')
        elif (not graph.is_leaf(self)) and (not self.requires_grad):
            return
        else:
            # Proceeds only when the tensor is not a leaf tensor and requires grad
            pass

        node = graph.get_node_by_output_tensor(self)
        node.backprop()

        # Delete the gradient after backprop.
        # This is omitted when the Zhangliang is a parameter.
        # See Parameter class
        if not retain_graph:
            self.release()

        parents = graph.get_parents(node)
        # Recursive for the parent nodes
        for node_in in parents:
            output_tuple = node_in.output
            for o in output_tuple:
                if isinstance(o, Zhangliang):
                    o.backward(retain_graph)

        if graph.is_leaf(self):
            graph.clear_graph()

    def detach(self):
        return Zhangliang(self._zhi, dtype=self.dtype, requires_grad=False)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return truediv(self, other)

    def __rtruediv__(self, other):
        return truediv(other, self)

    def __abs__(self):
        return abs(self)

    def __iadd__(self, other):
        return add(self, other)

    def __isub__(self, other):
        return sub(self, other)

    def __imul__(self, other):
        return mul(self, other)

    def __imatmul__(self, other):
        return matmul(self, other)

    def __itruediv__(self, other):
        return truediv(self, other)

    def __pow__(self, power, modulo=None):
        return pow(self, power)

    def __rpow__(self, other):
        return pow(other, self)

    def __ge__(self, other):
        return ge(self, other)

    def __gt__(self, other):
        return gt(self, other)

    def __le__(self, other):
        return le(self, other)

    def __lt__(self, other):
        return lt(self, other)

    def __eq__(self, other):
        return eq(self, other)

    def __ne__(self, other):
        return ne(self, other)

    def __and__(self, other):
        return elt_and(self, other)

    def __or__(self, other):
        return elt_or(self, other)

    def __xor__(self, other):
        return elt_xor(self, other)

    def __neg__(self):
        return neg(self)

    def sum(self, dim=None, keepdims=False):
        return reduce_sum(self, dim, keepdims)

    def mean(self, dim=None, keepdims=False):
        return reduce_mean(self, dim, keepdims)

    def reshape(self, new_shape):
        return reshape(self, new_shape)

    def squeeze(self, dim=None):
        return squeeze(self, dim=dim)

    def unsqueeze(self, dim):
        return unsqueeze(self, dim)


class Parameters(Zhangliang):
    def __init__(self, data, dtype=np.float64, **kwargs):
        if isinstance(data, Zhangliang):
            data = data.values
        super(Parameters, self).__init__(data, dtype, requires_grad=True)

    def backward(self, retain_graph=False):
        if not graph.is_initialized():
            graph.toposort()

        if graph.is_leaf(self) and self.requires_grad:
            self.update_grad(1.)
        elif graph.is_leaf(self) and (not self.requires_grad):
            raise AttributeError('Zhangliang does not requires grad.')
        elif (not graph.is_leaf(self)) and (not self.requires_grad):
            return
        else:
            # Proceeds only when the tensor is not a leaf tensor and requires grad
            pass

        node = graph.get_node_by_output_tensor(self)
        node.backprop()

        parents = graph.get_parents(node)
        # Recursive for the parent nodes
        for node_in in parents:
            output_tuple = node_in.output
            for o in output_tuple:
                if isinstance(o, Zhangliang):
                    o.backward(retain_graph)

        if graph.is_leaf(self):
            graph.clear_graph()

    def add_(self, x):
        # Only used to update the parameters
        if isinstance(x, Zhangliang):
            self._zhi += x.values
        elif isinstance(x, np.ndarray) or np.isscalar(x):
            self._zhi += x
        else:
            raise TypeError('Unrecognized data type.')


def is_zhangliang_requires_grad(x):
    if isinstance(x, (np.ndarray, numbers.Real)):
        return False
    elif isinstance(x, Zhangliang):
        return x.requires_grad
    else:
        return False


def aggregate_and_reshape_grad(grad_values, axes_to_reduce, target_shape):
    if len(axes_to_reduce) >= 0:
        aggregated_grad = np.sum(grad_values, axis=tuple(axes_to_reduce))
    else:
        aggregated_grad = grad_values
    return np.reshape(aggregated_grad, target_shape)


# ---------------------------------------------------------- #
# math ops for Zhangliang
# ---------------------------------------------------------- #

# ------------ Binary operators-------------


@ctx_register(op_name='add')
def add(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    # Incase of non-zhangliang, convert the inputs to Zhangliang
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    value = a_.values + b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='add')
def add_grad(output_tuple, x, y):
    # output_tuple is a tuple, unpack it first
    output = output_tuple[0]
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        grads = aggregate_and_reshape_grad(output.grad, axes_to_reduce[0], x_.shape)
        x.update_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = aggregate_and_reshape_grad(output.grad, axes_to_reduce[1], y_.shape)
        y.update_grad(grads)


@ctx_register(op_name='sub')
def sub(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    value = a_.values - b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='sub')
def sub_grad(output_tuple, x, y):
    # output_tuple is a tuple, unpack it first
    output = output_tuple[0]
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        grads = aggregate_and_reshape_grad(output.grad, axes_to_reduce[0], x_.shape)
        x.update_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = aggregate_and_reshape_grad(-output.grad, axes_to_reduce[1], y_.shape)
        y.update_grad(grads)


@ctx_register(op_name='mul')
def mul(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    value = a_.values * b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='mul')
def mul_grad(output_tuple, x, y):
    # output_tuple is a tuple, unpack it first
    output = output_tuple[0]
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        # The output definitely can broadcast with each input.
        grads = output.grad * y_.values
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[0], x_.shape)
        x.update_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = output.grad * x_.values
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[1], y_.shape)
        y.update_grad(grads)


@ctx_register(op_name='div')
def truediv(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    value = a_.values / b_.values
    return Zhangliang(value, dtype=value.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='div')
def truediv_grad(output_tuple, x, y):
    # output_tuple is a tuple, unpack it first
    output = output_tuple[0]
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        # The output definitely can broadcast with each input.
        grads = output.grad / y_.values
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[0], x_.shape)
        x.update_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = - output.grad * x_.values / (y.values ** 2)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[1], y_.shape)
        y.update_grad(np.reshape(grads, y.shape))


@ctx_register(op_name='matmul')
def matmul(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    values = np.matmul(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='matmul')
def matmul_grad(output_tuple, x, y):
    # output_tuple is a tuple, unpack it first
    output = output_tuple[0]
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = multiplicative_broadcast_analysis(inputs_shapes, output_shape)

    a_dim, b_dim = x_.shape, y_.shape

    # In case of (m, n) X (n, ) = (m, ).
    # (m, ) X (1, n) is impossible in forward mode. So maybe only inputs[1] needs to be checked.
    if len(b_dim) == 1:
        b_transposed = y_.values[np.newaxis, :]
        output_grad = output.grad[..., np.newaxis]
    else:
        b_transposed = np.swapaxes(y_.values, -1, -2)
        output_grad = output.grad

    a_grad = np.matmul(output_grad, b_transposed)
    if isinstance(x, Zhangliang) and x.requires_grad:
        x.update_grad(np.sum(a_grad, axis=axes_to_reduce[0]))

    a_transposed = np.swapaxes(x.values, -1, -2)
    b_grad = np.matmul(a_transposed, output_grad)
    if isinstance(y, Zhangliang) and y.requires_grad:
        y.update_grad(np.sum(b_grad, axis=axes_to_reduce[1]))


# TODO: `Zhangliang` does not seem to support for `complex` data type.
# So the inputs of the `pow` should be positive.
@ctx_register(op_name='pow')
def pow(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.power(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='pow')
def pow_grad(output_tuple, x, y):
    # output_tuple is a tuple, unpack it first
    output = output_tuple[0]
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        # The output definitely can broadcast with each input.
        power = np.where(y_.values, y_.values-1, 1.)
        grads = output.grad * y_.values * np.power(x.values, power)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[0], x.shape)
        x.update_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        coef = np.log(np.where(x_.values, x_.values, 1.))
        grads = output.grad * np.power(x_.values, y.values) * coef
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[1], y.shape)
        y.update_grad(grads)


@ctx_register(op_name='square')
def square(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.square(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='square')
def square_grad(output_tuple, x):
    # output_tuple is a tuple, unpack it first
    output = output_tuple[0]
    x_ = Zhangliang(x)
    if isinstance(x, Zhangliang) and x.requires_grad:
        # The output definitely can broadcast with each input.
        grads = 2 * output.grad * x_.values
        x.update_grad(grads)


# Borrowed from
# https://github.com/HIPS/autograd/blob/387c373115ddd54cff2c8ba6a9fc619f28639cfb/autograd/numpy/numpy_vjps.py#L672
def balanced_eq(xin, yin, zout):
    return (xin == zout) / (1. + (xin == yin))


@ctx_register(op_name='maximum')
def maximum(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.maximum(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='maximum')
def maximum_grad(output_tuple, x, y):
    output = output_tuple[0]
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        grads = output.grad * balanced_eq(x.values, y_.values, output.values)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[0], x.shape)
        x.update_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = output.grad * balanced_eq(y.values, x_.values, output.values)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[1], y.shape)
        y.update_grad(grads)


@ctx_register(op_name='minimum')
def minimum(x, y):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(y)
    a_ = Zhangliang.array(x)
    b_ = Zhangliang.array(y)
    values = np.minimum(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='minimum')
def minimum_grad(output_tuple, x, y):
    output = output_tuple[0]
    x_ = Zhangliang(x)
    y_ = Zhangliang(y)
    inputs_shapes = tuple([x_.shape, y_.shape])
    output_shape = output.shape
    axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
    if isinstance(x, Zhangliang) and x.requires_grad:
        grads = output.grad * balanced_eq(x.values, y_.values, output.values)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[0], x.shape)
        x.update_grad(grads)
    if isinstance(y, Zhangliang) and y.requires_grad:
        grads = output.grad * balanced_eq(y.values, x_.values, output.values)
        grads = aggregate_and_reshape_grad(grads, axes_to_reduce[1], y.shape)
        y.update_grad(grads)


# Compare functions cannot backprop gradients. No need to trace them.
@func_register(op_name='ge')
def ge(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.greater_equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='ge')
def ge_grad(output_tuple, x, y):
    # The output of `ge` function does not require grad. So pass.
    pass


@func_register(op_name='gt')
def gt(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.greater(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='gt')
def gt_grad(output_tuple, x, y):
    # The output of `gt` function does not require grad. So pass.
    pass


@func_register(op_name='le')
def le(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.less_equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='le')
def le_grad(output_tuple, x, y):
    # The output of `le` function does not require grad. So pass.
    pass


@func_register(op_name='lt')
def lt(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.less(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='lt')
def lt_grad(output_tuple, x, y):
    # The output of `lt` function does not require grad. So pass.
    pass


@func_register(op_name='eq')
def eq(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='eq')
def eq_grad(output_tuple, x, y):
    pass


@func_register(op_name='ne')
def ne(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.not_equal(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='ne')
def ne_grad(output_tuple, x, y):
    pass


@func_register(op_name='elt_and')
def elt_and(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.logical_and(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_and')
def elt_and_grad(output_tuple, x, y):
    pass


@func_register(op_name='elt_or')
def elt_or(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.logical_or(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_or')
def elt_or_grad(output_tuple, x, y):
    output = output_tuple[0]
    pass


@func_register(op_name='elt_xor')
def elt_xor(x, y):
    a_ = Zhangliang(x)
    b_ = Zhangliang(y)
    values = np.logical_xor(a_.values, b_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_xor')
def elt_xor_grad(output_tuple, x, y):
    pass


# --------------- Unary operators ---------------


@ctx_register(op_name='neg')
def neg(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = - a_.values
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='neg')
def neg_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        x.update_grad(-output.grad)


@ctx_register(op_name='exp')
def exp(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.exp(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='exp')
def neg_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        x.update_grad(output.grad * x.values)


@ctx_register(op_name='reduce_mean')
def reduce_mean(x, dim=None, keepdims=False):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.mean(a_.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='reduce_mean')
def reduce_mean_grad(output_tuple, x, dim=None, keepdims=False):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        inputs_shapes = x.shape
        reduced_shapes = list(inputs_shapes)

        if dim is None:
            dim = list(range(x.ndim))
        reduced_scale = 1
        for i in dim:
            reduced_scale *= inputs_shapes[i]
            # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
            reduced_shapes[i] = 1
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        x.update_grad(np.reshape(output.grad / reduced_scale, reduced_shapes))


@ctx_register(op_name='reduce_sum')
def reduce_sum(x, dim=None, keepdims=False):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.sum(a_.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='reduce_sum')
def reduce_sum_grad(output_tuple, x, dim=None, keepdims=False):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        inputs_shapes = x.shape
        reduced_shapes = list(inputs_shapes)
        if dim is None:
            dim = list(range(x.ndim))

        for i in dim:
            # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
            reduced_shapes[i] = 1
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        x.update_grad(np.reshape(output.grad, reduced_shapes))


@ctx_register(op_name='reshape')
def reshape(x, new_shape):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.reshape(a_.values, new_shape)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='reshape')
def reshape_grad(output_tuple, x, new_shape):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        old_shape = x.shape
        x.update_grad(np.reshape(output.grad, old_shape))


@ctx_register(op_name='abs')
def abs(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.abs(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='abs')
def abs_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = np.where(x.values > 0, 1., -1.)
        values = np.where(x.values, values, 0.)
        x.update_grad(output.grad * values)


@ctx_register(op_name='log')
def log(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.log(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='log')
def log_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / x.values
        x.update_grad(values)


@ctx_register(op_name='log2')
def log2(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.log2(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='log2')
def log2_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / x.values / np.log(2)
        x.update_grad(values)


@ctx_register(op_name='log10')
def log10(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.log10(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='log10')
def log10_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / x.values / np.log(10)
        x.update_grad(values)


@ctx_register(op_name='log1p')
def log1p(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.log1p(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='log1p')
def log1p_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (1. + x.values)
        x.update_grad(values)


def grad_minmax(xin, zout, grad, dim=None, keepdims=False):
    inputs_shapes = xin.shape
    reduced_shapes = list(inputs_shapes)
    if dim is None:
        dim = list(range(xin.ndim))

    for i in dim:
        # We do not need the grad has the same shape as the values, but the broadcast shape is necessary.
        reduced_shapes[i] = 1

    zout = np.reshape(zout, newshape=reduced_shapes)
    grad = np.reshape(grad, newshape=reduced_shapes)
    max_value_map = xin == zout
    nmax = np.sum(max_value_map, axis=tuple(dim), keepdims=True)
    values = grad * max_value_map / nmax
    return values


@ctx_register(op_name='max')
def max(x, dim=None, keepdims=False):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.max(a_.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='max')
def max_grad(output_tuple, x, dim=None, keepdims=False):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = grad_minmax(x.values, output.values, output.grad,
                             dim=dim, keepdims=keepdims)
        x.update_grad(values)


@ctx_register(op_name='min')
def min(x, dim=None, keepdims=False):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.min(a_.values, axis=dim, keepdims=keepdims)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='min')
def min_grad(output_tuple, x, dim=None, keepdims=False):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = grad_minmax(x.values, output.values, output.grad,
                             dim=dim, keepdims=keepdims)
        x.update_grad(values)


@ctx_register(op_name='argmax')
def argmax(x, dim=None):
    a_ = Zhangliang(x)
    values = np.argmax(a_.values, axis=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='argmax')
def argmax_grad(output_tuple, x, dim=None):
    pass


@ctx_register(op_name='argmin')
def argmin(x, dim=None):
    a_ = Zhangliang(x)
    values = np.argmin(a_.values, axis=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='argmin')
def argmin_grad(output_tuple, x, dim=None):
    pass


@ctx_register(op_name='clamp')
def clamp(x, xmin=0., xmax=1.):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.clip(a_.values, a_max=xmax, a_min=xmin)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='clamp')
def clamp_grad(output_tuple, x, xmin=0., xmax=1.):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        valid_region = np.logical_and(x.values >= xmin, x.values <= xmax)
        values = output.grad * valid_region
        x.update_grad(values)


@func_register(op_name='elt_not')
def elt_not(x):
    a_ = Zhangliang(x)
    values = np.logical_not(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=False)


@grad_register(op_name='elt_not')
def elt_not_grad(output_tuple, x):
    pass


@ctx_register(op_name='sin')
def sin(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.sin(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='sin')
def sin_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad * np.cos(x.values)
        x.update_grad(values)


@ctx_register(op_name='cos')
def cos(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.cos(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='cos')
def cos_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = - output.grad * np.sin(x.values)
        x.update_grad(values)


@ctx_register(op_name='tan')
def tan(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.tan(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='tan')
def tan_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (np.cos(x.values) ** 2)
        x.update_grad(values)


# numpy package has no `cot` function. So we skip `cot`, as well as `arccot`.
@ctx_register(op_name='arcsin')
def arcsin(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arcsin(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arcsin')
def arcsin_grad(output_tuple, x):
    output = output_tuple[0]
    # TODO: the error becomes significant when the `x.values` are close to -1 and 1.
    # How to make it stable?
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (np.sqrt(1 - x.values ** 2))
        x.update_grad(values)


@ctx_register(op_name='arccos')
def arccos(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arccos(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arccos')
def arccos_grad(output_tuple, x):
    output = output_tuple[0]
    # TODO: the error becomes significant when the `x.values` are close to -1 and 1.
    # How to make it stable?
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = - output.grad / (np.sqrt(1 - x.values ** 2))
        x.update_grad(values)


@ctx_register(op_name='arctan')
def arctan(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arctan(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arctan')
def arctan_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (1 + x.values ** 2)
        x.update_grad(values)


@ctx_register(op_name='sinh')
def sinh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.sinh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='sinh')
def sinh_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad * np.cosh(x.values)
        x.update_grad(values)


@ctx_register(op_name='cosh')
def cosh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.cosh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='cosh')
def cosh_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad * np.sinh(x.values)
        x.update_grad(values)


@ctx_register(op_name='tanh')
def tanh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.tanh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='tanh')
def tanh_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (np.cosh(x.values) ** 2)
        x.update_grad(values)


@ctx_register(op_name='arcsinh')
def arcsinh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arcsinh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arcsinh')
def arcsinh_grad(output_tuple, x):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / np.sqrt(x.values ** 2 + 1)
        x.update_grad(values)


@ctx_register(op_name='arccosh')
def arccosh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arccosh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arccosh')
def arccosh_grad(output_tuple, x):
    output = output_tuple[0]
    # TODO: the error becomes significant when the `x.values` are close to 1.
    # How to make it stable?
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / np.sqrt(x.values ** 2 - 1)
        x.update_grad(values)


@ctx_register(op_name='arctanh')
def arctanh(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.arctanh(a_.values)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='arctanh')
def arctanh_grad(output_tuple, x):
    output = output_tuple[0]
    # TODO: the error becomes significant when the `x.values` are close to -1 and 1.
    # How to make it stable?
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.grad / (1. - x.values ** 2)
        x.update_grad(values)


@ctx_register(op_name='squeeze')
def squeeze(x, dim=None):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    values = np.squeeze(a_.values, axis=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='squeeze')
def squeeze_grad(output_tuple, x, dim=None):
    output = output_tuple[0]
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = np.reshape(output.grad, x.shape)
        x.update_grad(values)


@ctx_register(op_name='unsqueeze')
def unsqueeze(x, dim):
    local_requires_grad = is_zhangliang_requires_grad(x)
    a_ = Zhangliang(x)
    old_shape = x.shape
    new_shape = list(old_shape)
    new_shape.insert(dim, 1)
    values = np.reshape(a_.values, newshape=new_shape)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='unsqueeze')
def unsqueeze_grad(output_tuple, x, dim=None):
    output = output_tuple[0]
    if isinstance(input, Zhangliang) and x.requires_grad:
        values = np.reshape(output.grad, x.shape)
        x.update_grad(values)


# ---------------- array-like functions -----------------


@ctx_register(op_name='concat')
def concat(inputs, dim=-1):
    local_requires_grad = False
    encap_inputs = []
    for a_input in inputs:
        local_requires_grad = local_requires_grad or is_zhangliang_requires_grad(a_input)
        encap_inputs.append(Zhangliang(a_input).values)

    values = np.concatenate(encap_inputs, axis=dim)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='concat')
def concat_grad(output_tuple, inputs, dim=-1):
    output = output_tuple[0]
    nsize = []
    for a_input in inputs:
        if not isinstance(a_input, Zhangliang):
            raise TypeError
        nsize.append(a_input.shape[dim])

    split_grads = np.split(output.grad, nsize, axis=dim)
    for i, a_input in enumerate(inputs):
        if a_input.requires_grad:
            a_input.update_grad(split_grads[i])


@ctx_register(op_name='hstack')
def hstack(inputs):
    local_requires_grad = False
    encap_inputs = []
    for a_input in inputs:
        local_requires_grad = local_requires_grad or is_zhangliang_requires_grad(a_input)
        encap_inputs.append(Zhangliang(a_input).values)

    values = np.concatenate(encap_inputs, axis=1)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='hstack')
def hstack_grad(output_tuple, inputs):
    output = output_tuple[0]
    nsize = []
    for a_input in inputs:
        if not isinstance(a_input, Zhangliang):
            raise TypeError
        nsize.append(a_input.shape[1])

    split_grads = np.split(output.grad, nsize, axis=1)
    for i, a_input in enumerate(inputs):
        if a_input.requires_grad:
            a_input.update_grad(split_grads[i])


@ctx_register(op_name='vstack')
def vstack(inputs):
    local_requires_grad = False
    encap_inputs = []
    for a_input in inputs:
        local_requires_grad = local_requires_grad or is_zhangliang_requires_grad(a_input)
        encap_inputs.append(Zhangliang(a_input).values)

    values = np.concatenate(encap_inputs, axis=0)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='vstack')
def vstack_grad(output_tuple, inputs):
    output = output_tuple[0]
    nsize = []
    for a_input in inputs:
        if not isinstance(a_input, Zhangliang):
            raise TypeError
        nsize.append(a_input.shape[0])

    split_grads = np.split(output.grad, nsize, axis=0)
    for i, a_input in enumerate(inputs):
        if a_input.requires_grad:
            a_input.update_grad(split_grads[i])


@ctx_register(op_name='tile')
def tile(x, reps):
    local_requires_grad = is_zhangliang_requires_grad(x)
    values = np.tile(x.values, reps)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='tile')
def tile_grad(output_tuple, x, reps):
    output = output_tuple[0]
    xdim = x.ndim
    reps = [reps] if np.isscalar(reps) else reps
    d = len(reps)
    if d < xdim:
        reps = list([1] * (xdim - d)) + reps
    g = output.grad
    for axis, rep in enumerate(reps):
        g = sum(np.split(g, rep, axis))
    x.update_grad(g)


if __name__ == '__main__':
    from pyjazz.utils.tracer import graph
    a = Zhangliang([2, 3])
    a_ = Zhangliang(a)
    b = Zhangliang([-1, 0])
    c = -a
    d = Zhangliang(a)
    d += a

    z1 = a + b + 2
    graph.toposort()
    print(z1)
