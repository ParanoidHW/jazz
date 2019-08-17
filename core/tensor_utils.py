from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

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

    data = np.pad(im, pad_width=((pl, pr), (pu, pd)))
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
            values[:, :, :, ind] = np.reshape(data[:, :, yind, xind], newshape=(n, cin, kh*kw))
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
    new_im = np.pad(new_im, pad_width=((pl, pr), (pu, pd)))
    for y in np.arange(hout):
        ys = sh*y
        ye = ys + (kh - 1)*dh + 1
        yind = np.arange(ys, ye, dh)
        for x in np.arange(wout):
            xs = sw * x
            xe = xs + (kw - 1) * dw + 1
            xind = np.arange(xs, xe, dw)
            new_im[:,:,yind,xind] += np.reshape(im[:,:,:,:,y,x], (n, cin, kh*kw))
    n, cin, hin_pad, win_pad = new_im.shape
    return new_im[:,:, pl:hin_pad-pr, pu:win_pad-pd]
