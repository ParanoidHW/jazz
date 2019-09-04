from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numbers
import collections
import numpy as np
from core.tensor import Zhangliang


def get_conv_size(im_size, k, s, p, d):
    h, w = im_size
    kh, kw = k
    sh, sw = s
    pl, pr, pu, pd = p
    dh, dw = d

    hout = int((h + pl + pr - (kh - 1) * dh - 1) / sh) + 1
    wout = int((w + pu + pd - (kw - 1) * dw - 1) / sw) + 1
    return hout, wout

def get_convtr_size(im_size, k, s, p, d):
    h, w = im_size
    kh, kw = k
    sh, sw = s
    pl, pr, pu, pd = p
    dh, dw = d

    hin = (h - 1)*sh + 1 + (kh-1)*dh - (pl + pr)
    win = (w - 1)*sw + 1 + (kw-1)*dw - (pu + pd)
    return hin, win


def im2col(im, ksize, stride, padding, dilation):
    # TODO: define the data format. Default to NCHW
    n, cin, hin, win = im.shape
    kh, kw = ksize
    sh, sw = stride
    pl, pr, pu, pd = padding
    dh, dw = dilation

    hout, wout = get_conv_size((hin, win), ksize, stride, padding, dilation)

    data = np.pad(im, pad_width=((0, 0), (0, 0), (pl, pr), (pu, pd)), mode='constant', constant_values=0)
    values = np.zeros((n, cin, kh*kw, hout*wout), dtype=im.dtype)
    for y in np.arange(hout):
        ys = sh*y
        ye = ys + (kh - 1)*dh + 1
        yind = np.arange(ys, ye, dh)
        for x in np.arange(wout):
            xs = sw * x
            xe = xs + (kw - 1) * dw + 1
            xind = np.arange(xs, xe, dw)
            ind = y*wout+x
            values[:, :, :, ind] = np.reshape(data[:, :, yind[:, None], xind[None, :]], newshape=(n, cin, kh*kw))
    values = np.reshape(values, (n, cin*kh*kw, hout*wout))
    return Zhangliang(values, dtype=im.dtype, requires_grad=False), (hout, wout)


def im2col_backward(im, num_filters, kernel, stride, padding, dilation):
    pass


def col2im(im, cin, hin, win, ksize, stride, padding, dilation):
    pass



def col2im_backward(im, hin, win, stride, padding, dilation):
    n, cin, kh, kw, hout, wout = im.shape
    sh, sw = stride
    pl, pr, pu, pd = padding
    dh, dw = dilation

    new_im = np.zeros((n, cin, hin, win), dtype=im.dtype)
    new_im = np.pad(new_im, pad_width=((0, 0), (0, 0), (pl, pr), (pu, pd)), mode='constant', constant_values=0)
    for y in np.arange(hout):
        ys = sh*y
        ye = ys + (kh - 1)*dh + 1
        yind = np.arange(ys, ye, dh)
        for x in np.arange(wout):
            xs = sw * x
            xe = xs + (kw - 1) * dw + 1
            xind = np.arange(xs, xe, dw)
            # new_im[:,:,yind[:, None], xind[None, :]] += np.reshape(im[:,:,:,:,y,x], (n, cin, kh*kw))
            new_im[:,:,yind[:, None], xind[None, :]] += im[:,:,:,:,y,x]
    n, cin, hin_pad, win_pad = new_im.shape
    return new_im[:,:, pl:hin_pad-pr, pu:win_pad-pd]


def get_op_settings(stride, padding, dilation):
    if isinstance(stride, numbers.Integral):
        stride_ = (stride, ) * 2
    elif isinstance(stride, (tuple, list)):
        if len(stride) == 2:
            stride_ = tuple(stride)
        elif len(stride) == 1:
            stride_ = tuple(stride) * 2
        else:
            raise ValueError
    else:
        raise TypeError

    if isinstance(padding, numbers.Integral):
        padding_ = (padding, padding, padding, padding)
    elif isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            padding_ = tuple(padding) + (0, 0)
        elif len(padding) == 4:
            padding_ = tuple(padding)
        elif len(padding) == 1:
            padding_ = tuple(padding) * 4
        else:
            raise ValueError
    else:
        raise TypeError

    if isinstance(dilation, numbers.Integral):
        dilation_ = (dilation, ) * 2
    elif isinstance(dilation, (tuple, list)):
        if len(dilation) == 2:
            dilation_ = tuple(dilation)
        elif len(dilation) == 1:
            dilation_ = tuple(dilation) * 2
        else:
            raise ValueError
    else:
        raise TypeError

    return stride_, padding_, dilation_
