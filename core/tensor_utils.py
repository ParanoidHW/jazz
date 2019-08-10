from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
from core.tensor import Zhangliang


def im2col(im, kernel, stride, padding, dilation):
    # TODO: remember to define the data format. Default to NCHW
    n, cin, hin, win = im.shape
    kh, kw = kernel
    sh, sw = stride
    pl, pr, pu, pd = padding
    dh, dw = dilation

    hout = int((hin + pl + pr - (kh - 1)*dh - 1) / sh) + 1
    wout = int((win + pu + pd - (kw - 1)*dw - 1) / sw) + 1

    data = np.pad(im.values, pad_width=((pl, pr), (pu, pd)))
    values = np.zeros((n, kh*kw, cin, hout*wout), dtype=im.dtype)
    for y in np.arange(hout):
        ys = sh*y
        ye = ys + (kh - 1)*dh + 1
        yind = np.arange(ys, ye, dh)
        for x in np.arange(wout):
            xs = sw * x
            xe = xs + (kw - 1) * dw + 1
            xind = np.arange(xs, xe, dw)
            ind = y*wout+x
            values[:, :, :, ind] = np.reshape(data[:, :, yind, xind], newshape=(n, cin, kh*kw))
    values = np.reshape(values, (kh*kw*cin, hout*wout))
    return Zhangliang(values, dtype=im.dtype, requires_grad=False)


def im2col_backward(im, num_filters, kernel, stride, padding, dilation):
    pass


def col2im(im, num_filters, kernel, stride, padding, dilation):
    pass


def col2im_backward(im, num_filters, kernel, stride, padding, dilation):
    pass
