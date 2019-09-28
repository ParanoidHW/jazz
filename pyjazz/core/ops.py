from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from pyjazz.core.grad_mode import no_grad
from pyjazz.core.tensor import *
from pyjazz.core.tensor_utils import im2col, col2im_backward, get_convtr_size, get_op_settings

EPS = 1e-8


@ctx_register(op_name='linear')
def linear(x, w):
    # Register the whole layer as an op, rather than decompose it into several base ops.
    # Therefore inside the layers, we set no_grad() not to add the interleave nodes into the graph.
    # Note: `x` `w` and `b` should be a Zhangliang.
    # TODO: add type check.
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(w)
    with no_grad():
        y = matmul(x, w)
    return Zhangliang(y.values, dtype=y.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='linear')
def linear_grad(output, x, w):
    matmul_grad(output, x, w)


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


@ctx_register(op_name='lrelu')
def lrelu(x, alpha=.2):
    local_requires_grad = is_zhangliang_requires_grad(x)
    with no_grad():
        x_ = Zhangliang(x)
        values = x_.values
        values[values < 0] *= alpha
    return Zhangliang(values, dtype=values.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='lrelu')
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
        dz_sum = np.sum(output.values * output.grad, axis=dim, keepdims=True)
        values = output.values * (output.grad - dz_sum)
        x.update_grad(values)


@ctx_register(op_name='softplus')
def softplus(x):
    local_requires_grad = is_zhangliang_requires_grad(x)
    with no_grad():
        y = exp(x)
        z = log1p(y)
    return Zhangliang(z.values, dtype=z.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='softplus')
def softplus_grad(output, x):
    if isinstance(x, Zhangliang) and x.requires_grad:
        values = np.exp(x.values)
        values = output.grad * values / (1. + values)
        x.update_grad(values)


@ctx_register(op_name='conv2d')
def conv2d(x, k, bias=None, stride=1, padding=0, dilation=1):
    stride, padding, dilation = get_op_settings(stride, padding, dilation)
    local_requires_grad = is_zhangliang_requires_grad(x) or is_zhangliang_requires_grad(k)
    x_ = Zhangliang(x)
    k_ = Zhangliang(k)
    cout, cin, kh, kw = k_.shape
    assert cin == x_.shape[1], "Expected feature dimension {}, but got {}.". \
        format(cin, x_.shape[1])

    # Convert each region into a column
    x_col, target_size = im2col(x_.values, (kh, kw), stride, padding, dilation)
    # Reshape the kernel into rows
    k_row = np.reshape(k_.values, (cout, cin*kh*kw))
    # Apply convolution
    y = np.matmul(k_row, x_col)
    y_old_shape = y.shape
    y_new_shape = y_old_shape[:-1] + target_size
    y = np.reshape(y, y_new_shape)
    if bias is not None:
        bias_ = Zhangliang(bias)
        y = y + np.reshape(bias_.values, (1,cout,1,1))

    return Zhangliang(y, dtype=y.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='conv2d')
def conv2d_grad(output, x, k, bias=None, stride=1, padding=0, dilation=1):
    stride, padding, dilation = get_op_settings(stride, padding, dilation)
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

    if bias is not None and isinstance(bias, Zhangliang) and bias.requires_grad:
        output_shape = output.shape
        axes_to_reduce = [i for i in range(len(output_shape)) if i != 1]
        grads = aggregate_and_reshape_grad(output.grad, axes_to_reduce, bias.shape)
        bias.update_grad(grads)


@ctx_register(op_name='conv2d_transpose')
def conv2d_transpose(x, k, bias=None, stride=1, padding=0, dilation=1):
    stride, padding, dilation = get_op_settings(stride, padding, dilation)
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

    if bias is not None:
        bias_ = Zhangliang(bias)
        y = y + np.reshape(bias_.values, (1,cin,1,1))

    return Zhangliang(y, dtype=y.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='conv2d_transpose')
def conv2d_transpose_grad(output, x, k, bias=None, stride=1, padding=0, dilation=1):
    stride, padding, dilation = get_op_settings(stride, padding, dilation)
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

    if bias is not None and isinstance(bias, Zhangliang) and bias.requires_grad:
        inputs_shapes = tuple([bias.shape])
        output_shape = output.shape
        axes_to_reduce = additive_broadcast_analysis(inputs_shapes, output_shape)
        grads = aggregate_and_reshape_grad(output.grad, axes_to_reduce[0], bias.shape)
        bias.update_grad(grads)


def _pooling_prepare(x, kernel_size, stride, dilation):
    stride, _, dilation = get_op_settings(stride, 0, dilation)

    x_ = Zhangliang(x)
    b, c, h, w = x_.shape

    kh, kw = kernel_size

    x_col, target_size = im2col(x_.values, (kh, kw), stride, 0, dilation)
    x_col = np.reshape(x_col, (b, c, kh * kw) + target_size)

    return x_col


@ctx_register(op_name='max_pool2d')
def max_pool2d(x, kernel_size=2, stride=1, dilation=1, return_indices=False):
    local_requires_grad = is_zhangliang_requires_grad(x)
    x_col = _pooling_prepare(x, kernel_size, stride, dilation)

    y = np.max(x_col, axis=2, keepdims=False)
    if return_indices:
        y_indices = np.argmax(x_col, axis=2)
        return Zhangliang(y, dtype=y.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled()), \
            Zhangliang(y_indices, dtype=np.int32, requires_grad=False)
    else:
        return Zhangliang(y, dtype=y.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='max_pool2d')
def max_pool2d_grad(output, indices, x, kernel_size=2, stride=1, dilation=1, return_indices=False):
    stride, padding, dilation = get_op_settings(stride, 0, dilation)

    x_ = Zhangliang(x)
    b, c, h, w = x_.shape
    kh, kw = kernel_size

    if isinstance(x, Zhangliang) and x.requires_grad:
        output_grad = np.reshape(output.grad, (b, c, 1, h, w))  # [n,cout,1, hout*wout]
        output_values = np.reshape(output.values, (b, c, 1, h, w))
        x_col, target_size = im2col(x_.values, (kh, kw), stride, 0, dilation)
        x_col = np.reshape(x_col, (b, c, kh * kw) + target_size)

        # We do not use the returned indices during the forward, since the max value may
        # may occur in several places in each conv region.
        max_indices = output_values == x_col
        norm_indices = max_indices / np.sum(max_indices, axis=2, keepdims=True)
        x_grad = output_grad * norm_indices
        x_grad = col2im_backward(x_grad, h, w, stride, padding, dilation)

        x.update_grad(x_grad)


@ctx_register(op_name='avg_pool2d')
def avg_pool2d(x, kernel_size=2, stride=1, padding=0, dilation=1):
    local_requires_grad = is_zhangliang_requires_grad(x)
    x_col = _pooling_prepare(x, kernel_size, stride, dilation)

    y = np.mean(x_col, axis=2, keepdims=False)
    return Zhangliang(y, dtype=y.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='avg_pool2d')
def avg_pool2d_grad(outputs, x, kernel_size=2, stride=1, dilation=1):
    stride, padding, dilation = get_op_settings(stride, 0, dilation)

    x_ = Zhangliang(x)
    b, c, h, w = x_.shape
    kh, kw = kernel_size

    if isinstance(x, Zhangliang) and x.requires_grad:
        output_grad = np.reshape(outputs.grad, (b, c, 1, h, w))  # [n,cout,1, hout*wout]
        norm_indices = 1./ (kw * kh)
        x_grad = output_grad * norm_indices
        x_grad = col2im_backward(x_grad, h, w, stride, padding, dilation)
        x.update_grad(x_grad)


@func_register(op_name='one_hot')
def one_hot(x, depth, dim=-1):
    x_ = Zhangliang(x)
    indices = np.expand_dims(x_.values, axis=dim).astype(np.int32)
    target_shape = list(indices.shape)
    target_shape[dim] = depth

    flatten_ind = indices.reshape(-1)
    one_hot_target = np.eye(depth)[flatten_ind]
    one_hot_target = np.reshape(one_hot_target, target_shape)

    return Zhangliang(one_hot_target, dtype=np.float32, requires_grad=False)


@grad_register(op_name='one_hot')
def one_hot_grad(output, x, depth, dim=-1):
    pass


@ctx_register(op_name='cross_entropy')
def cross_entropy(act, label, dim=-1):
    local_requires_grad = is_zhangliang_requires_grad(act) or is_zhangliang_requires_grad(label)
    x_ = Zhangliang(act)
    y_ = Zhangliang(label)

    logits = np.log(x_.values + EPS)
    l = - np.sum(y_.values * logits, axis=dim, keepdims=False)
    return Zhangliang(l, dtype=l.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='cross_entropy')
def cross_entropy_grad(output, act, label, dim=-1):
    x_ = Zhangliang(act)
    y_ = Zhangliang(label)
    if isinstance(act, Zhangliang) and act.requires_grad:
        grad = np.expand_dims(output.grad, axis=dim)
        grad = - grad * y_.values / (act.values + EPS)
        act.update_grad(grad)

    if isinstance(label, Zhangliang) and label.requires_grad:
        grad = np.expand_dims(output.grad, axis=dim)
        logits = np.log(x_.values + EPS)
        grad = - logits * grad
        label.update_grad(grad)


@ctx_register(op_name='cross_entropy_with_logit')
def cross_entropy_with_logits(logit, label, dim=-1):
    local_requires_grad = is_zhangliang_requires_grad(logit) or is_zhangliang_requires_grad(label)
    x_ = Zhangliang(logit)
    y_ = Zhangliang(label)

    l = - np.sum(y_.values * x_.values, axis=dim, keepdims=False)
    return Zhangliang(l, dtype=l.dtype, requires_grad=local_requires_grad and graph.is_grad_enabled())


@grad_register(op_name='cross_entropy_with_logit')
def cross_entropy_with_logits_grad(output, logit, label, dim=-1):
    x_ = Zhangliang(logit)
    y_ = Zhangliang(label)
    if isinstance(logit, Zhangliang) and logit.requires_grad:
        grad = np.expand_dims(output.grad, axis=dim)
        grad = - grad * y_.values
        logit.update_grad(grad)

    if isinstance(label, Zhangliang) and label.requires_grad:
        grad = np.expand_dims(output.grad, axis=dim)
        grad = - x_.values * grad
        label.update_grad(grad)
