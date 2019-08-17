from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
from core.grad_mode import no_grad
from core.tensor import *
from core.tensor_utils import im2col, get_conv_size, col2im_backward, get_convtr_size


@ctx_register(op_name='linear')
def linear(x, w):
    # Register the whole layer as an op, rather than decompose it into several base ops.
    # Therefore inside the layers, we set no_grad() not to add the interleave nodes into the graph.
    # Note: `x` `w` and `b` should be a Zhangliang.
    # TODO: add type check.
    with no_grad():
        y = zl_matmul(x, w)
    return y


@grad_register(op_name='linear')
def linear_grad(output, x, w):
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


@ctx_register(op_name='conv2d')
def conv2d(x, k, stride=1, padding=0, dilation=1):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(k)
    x_ = Zhangliang(x)
    k_ = Zhangliang(k)
    cout, cin, kh, kw = k_.shape
    assert cin == x_.shape[1], "Expected feature dimension {}, but got {}.". \
        format(cin, x_.shape[1])

    # Cover each region into a column
    x_col, target_size = im2col(x_.values, (kh, kw), stride, padding, dilation)
    # Reshape the kernel into rows
    k_row = np.reshape(k_.values, (cout, cin*kh*kw))
    # Apply convolution
    y = np.matmul(k_row, x_col)
    y_old_shape = y.shape
    y_new_shape = y_old_shape[:-1] + target_size
    y = np.reshape(y, y_new_shape)
    return Zhangliang(y, dtype=y.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='conv2d')
def conv2d_grad(output, x, k, stride=1, padding=0, dilation=1):
    x_ = Zhangliang(x)
    k_ = Zhangliang(k)
    n, cin, hin, win = x_.shape
    _, cout, hout, wout = output.shape
    kcout, kcin, kh, kw = k_.shape

    # Reshape the kernel into rows
    k_row = np.reshape(k_.values, (cout, cin * kh * kw))
    output_grad = np.reshape(output.grad, (n, cout, -1))  # [n,cout,hout*wout]

    if isinstance(x, Zhangliang) and x.requires_grad:
        # Apply convolution transpose
        x_grad = np.matmul(k_row.T, output_grad)
        x_grad = np.reshape(x_grad, newshape=(n,cin,kh,kw,hout,wout))
        x_grad = col2im_backward(x_grad, hin, win, stride, padding, dilation)
        x.update_grad(x_grad)

    if isinstance(k, Zhangliang) and k.requires_grad:
        # Apply convolution transpose
        x_col, _ = im2col(x_.values, (kh, kw), stride, padding, dilation)  # [n,cin*kh*kw,hout*wout]
        x_col = np.transpose(x_col, (0,2,1))                # [n,hout*wout,cin*kh*kw]
        k_grad = np.matmul(output_grad, x_col)              # [n,cout,cin*kh*kw]
        k_grad = np.sum(k_grad, axis=0)                     # [cout, cin*kh*kw]
        k_grad = np.reshape(k_grad, k.shape)
        k.update_grad(k_grad)


@ctx_register(op_name='conv2d_transpose')
def conv2d_transpose(x, k, stride=1, padding=0, dilation=1):
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(k)
    x_ = Zhangliang(x)
    k_ = Zhangliang(k)

    cout, cin, kh, kw = k_.shape
    n, _, hout, wout = x_.shape
    assert cout == x_.shape[1], "Expected feature dimension {}, but got {}.". \
        format(cout, x_.shape[1])

    k_row = np.reshape(k_.values, (cout, cin * kh * kw))
    x_rows = np.reshape(x_.values, (n,cout,-1))

    hin, win = get_convtr_size(x_.shape[-2:], k_.shape[-2:], stride, padding, dilation)
    y = np.matmul(k_row.T, x_rows)  # [n, cin*kh*kw, hout*wout]
    y = np.reshape(y, newshape=(n, cin, kh, kw, hout, wout))
    y = col2im_backward(y, hin, win, stride, padding, dilation)

    return Zhangliang(y, dtype=y.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='conv2d_transpose')
def conv2d_transpose_grad(output, x, k, stride=1, padding=0, dilation=1):
    x_ = Zhangliang(x)
    k_ = Zhangliang(k)
    n, cout, hout, wout = x_.shape
    _, cin, hin, win = output.shape
    kcout, kcin, kh, kw = k_.shape

    # Cover each region into a column
    grad_col, _ = im2col(output.grad, (kh, kw), stride, padding, dilation)  # [n,cin*kh*kw, hout*wout]

    if isinstance(x, Zhangliang) and x.requires_grad:
        # Reshape the kernel into rows
        k_row = np.reshape(k_.values, (cout, cin * kh * kw))  # [cout, cin*kh*kw]
        x_grad = np.matmul(k_row, grad_col)                   # [n, cout, hout*wout]
        x_grad = np.reshape(x_grad, x.shape)
        x.update_grad(x_grad)

    if isinstance(k, Zhangliang) and k.requires_grad:
        # Reshape x into rows
        x_row = np.reshape(x_.values, (n, cout, hout * wout))  # [n, cout, hout*wout]
        k_grad = np.matmul(x_row, grad_col.T)                  # [n, cout, cin*kh*kw]
        k_grad = np.sum(k_grad, axis=0)
        k_grad = np.reshape(k_grad, k.shape)
        k.update_grad(k_grad)

