from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from core.grad_mode import no_grad
from core.tensor import *


@ctx_register(op_name='biased_fc')
def x_matmul_w_plus_b(x, w, b):
    # Register the whole layer as an op, rather than decompose it into several base ops.
    # Therefore inside the layers, we set no_grad() not to add the interleave nodes into the graph.
    # Note: `x` `w` and `b` should be a Zhangliang.
    # TODO: add type check.
    with no_grad():
        xw = zl_matmul(x, w)
        y = xw + b
    return y


@grad_register(op_name='biased_fc')
def x_matmul_w_plus_b_grad(output, x, w, b):
    x_ = Zhangliang(x)
    w_ = Zhangliang(w)
    b_ = Zhangliang(b)

    inputs_shapes = tuple([x_.shape, w_.shape])
    output_shape = output.shape
    matmul_to_reduce = multiplicative_broadcast_analysis(inputs_shapes, output_shape)
    plus_to_reduce = additive_broadcast_analysis([b_.shape], output_shape)

    x_dim, w_dim = x_.shape, w_.shape

    # In case of (m, n) X (n, ) = (m, ).
    # (m, ) X (1, n) is impossible in forward mode. So maybe only inputs[1] needs to be checked.
    if len(w_dim) == 1:
        w_transposed = w_.values[np.newaxis, :]
        output_grad = output.grad[..., np.newaxis]
    else:
        w_transposed = np.swapaxes(w_.values, -1, -2)
        output_grad = output.grad

    if isinstance(x, Zhangliang) and x.requires_grad:
        x_grad = np.matmul(output_grad, w_transposed)
        grads = aggregate_and_reshape_grad(x_grad, matmul_to_reduce[0], x.shape)
        x.update_grad(grads)

    if isinstance(w, Zhangliang) and w.requires_grad:
        x_transposed = np.swapaxes(x.values, -1, -2)
        w_grad = np.matmul(x_transposed, output_grad)
        grads = aggregate_and_reshape_grad(w_grad, matmul_to_reduce[1], w.shape)
        w.update_grad(grads)

    if isinstance(b, Zhangliang) and b.requires_grad:
        grads = aggregate_and_reshape_grad(output.grad, plus_to_reduce[0], b.shape)
        b.update_grad(grads)


@ctx_register(op_name='unbiased_fc')
def x_matmul_w(x, w):
    # Register the whole layer as an op, rather than decompose it into several base ops.
    # Therefore inside the layers, we set no_grad() not to add the interleave nodes into the graph.
    # Note: `x` `w` and `b` should be a Zhangliang.
    # TODO: add type check.
    with no_grad():
        y = zl_matmul(x, w)
    return y


@grad_register(op_name='unbiased_fc')
def x_matmul_w_grad(output, x, w):
    zl_matmul_grad(output, x, w)


@ctx_register(op_name='sigmoid')
def sigmoid(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    # Two cases:
    #    `x` > 0: If `x` is very large, exp(x) may overflow. Then we use 1 / (1 + exp(-x)).
    #    `x` <= 0: If abs(x) is very large, exp(-x) may overflow. Then we use exp(x) / (1 + exp(x))
    values = x.values

    signs = x.values > 0
    a = np.exp(-values[signs])
    values[signs] = 1. / (1. + a)

    signs = x.values <= 0
    a = np.exp(values[signs])
    values[signs] = a / (1. + a)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='sigmoid')
def sigmoid_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.values * (1. - output.values)
        values *= output.grad
        x.update_grad(values)


@ctx_register(op_name='relu')
def relu(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    with no_grad():
        x_ = Zhangliang(x)
        values = np.maximum(x_.values, 0.)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='relu')
def relu_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        ones = x.values > 0
        x.update_grad(output.grad * ones)


@ctx_register(op_name='leaky_relu')
def lrelu(x, alpha=.2):
    local_requires_grad = is_zhangliang_requires_grad(x)
    with no_grad():
        x_ = Zhangliang(x)
        values = x_.values
        values[values < 0] *= alpha
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='leaky_relu')
def lrelu_grad(output, x, alpha=.2):
    if isinstance(x, Zhangliang) and x.requires_grad:
        rate = x.values > 0
        rate[rate == 0] = alpha
        x.update_grad(output.grad * rate)


@ctx_register(op_name='softmax')
def softmax(x, dim=-1):
    local_requires_grad = is_zhangliang_requires_grad(x)
    # Common practise:
    #     x = x - max(x, dim)
    #     y = exp(x)
    #     z = y / sum(y, dim)
    values = x.values

    vmax = np.max(values, axis=dim, keepdims=True)
    values = values - vmax
    values = np.exp(values)
    values = values / np.sum(values, axis=dim, keepdims=True)
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='softmax')
def softmax_grad(output, x, dim=-1):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = output.values * (1. - output.values)
        values *= output.grad
        x.update_grad(values)


@ctx_register(op_name='softplus')
def softplus(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    with no_grad():
        y = zl_exp(x)
        z = zl_log1p(y)
    return Zhangliang(z.values, dtype=z.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='softplus')
def softplus_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = np.exp(x.values)
        values = output.grad * values / (1. + values)
        x.update_grad(values)

