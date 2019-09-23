from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numbers
import collections


def expand_dim_as(old_shape, ref_shape):
    ndim_a = len(old_shape)
    ndim_ref = len(ref_shape)
    new_shape = list([1] * (ndim_ref - ndim_a)) + list(old_shape)
    return new_shape


def check_elementwise_dim_compatible(a_shape, ref_shape):
    ndim_a = len(a_shape)
    ndim_ref = len(ref_shape)
    assert ndim_a == ndim_ref, 'Dimension not equal: {} vs {}.'.format(ndim_a, ndim_ref)
    compatible = True
    aggregated_axes = []
    for i, (dim_a, dim_ref) in enumerate(zip(a_shape, ref_shape)):
        # `ref_shape` is supposed to be the shape of the operator result.
        # So ref_shape[i] >= a_shape[i] at every where
        if dim_a != dim_ref and dim_a != 1:
            compatible = False
            return compatible, aggregated_axes
        elif dim_a == 1:
            aggregated_axes.append(i)
    return compatible, aggregated_axes


def check_matrixwise_dim_compatible(a_shape, ref_shape):
    ndim_a = len(a_shape)
    ndim_ref = len(ref_shape)
    assert ndim_a == ndim_ref, 'Dimension not equal: {} vs {}.'.format(ndim_a, ndim_ref)
    # The length of both input shapes must greater than or equal with 2.
    # Otherwise the operator cannot perform.
    # One of the last dim in `a_shape` should be the same with the corresponding dim of `ref_shape`.
    compatible = (a_shape[-1] == ref_shape[-1]) or (a_shape[-2] == ref_shape[-2])
    aggregated_axes = []
    for i, (dim_a, dim_ref) in enumerate(zip(a_shape[:-2], ref_shape[:-2])):
        # `ref_shape` is supposed to be the shape of the operator result.
        # So ref_shape[i] >= a_shape[i] at every where
        if dim_a != dim_ref and dim_a != 1:
            compatible = False
            return compatible, aggregated_axes
        elif dim_a == 1:
            aggregated_axes.append(i)
    return compatible, aggregated_axes


def additive_broadcast_analysis(input_shapes, output_shape):
    """
    Analyze and determine the broadcast method. Return a tuple of the axes to be reduced
    of the output w.r.t. each input.
    :param input_shapes: a list or tuple of input data shapes
    :param output_shape: the shape of the output
    :return:
    """
    reduced_axes = []
    for input_shape in input_shapes:
        # red_ax, invalid = data_shape_comp(input_shape, output_shape)
        new_shape = expand_dim_as(input_shape, output_shape)
        valid, axes_aggregated = check_elementwise_dim_compatible(new_shape, output_shape)
        if valid:
            reduced_axes.append(tuple(axes_aggregated))
        else:
            raise ArithmeticError

    return reduced_axes


def multiplicative_broadcast_analysis(input_shapes, output_shape):
    """
        Analyze and determine the broadcast method. Return a tuple of the axes to be reduce
        of the output w.r.t. each input.
        :param input_shapes: a list or tuple of input data shapes
        :param output_shape: the shape of the output
        :return:
        """
    # (m_1, m_2, ..., m_k, n) X (n, ) = (m_1, m_2, ..., m_k) happens.
    input_shapes_ = [list(a) for a in input_shapes]
    output_shape_ = list(output_shape)
    if len(input_shapes_[1]) == 1:
        input_shapes_[1] += [1]
        output_shape_ += [1]

    reduced_axes = []
    for input_shape in input_shapes_:
        # red_ax, invalid = data_shape_comp(input_shape, output_shape)
        new_shape = expand_dim_as(input_shape, output_shape_)
        valid, axes_aggregated = check_matrixwise_dim_compatible(new_shape, output_shape_)
        if valid:
            reduced_axes.append(tuple(axes_aggregated))
        else:
            raise ArithmeticError

    return reduced_axes


def recover_dim(ori_shape, tar_shape, dim=None, keepdims=False):
    new_shape = list(ori_shape)
    if keepdims:
        return new_shape

    if dim is not None:
        if isinstance(dim, (list, tuple)) or isinstance(dim, collections.Iterable):
            dim = sorted(dim)
        elif isinstance(dim, numbers.Integral):
            dim = [dim]
        else:
            raise ValueError
    else:
        dim = list(range(len(ori_shape)))
    for d in dim:
        new_shape[d] = 1
    return new_shape


# Monotonically checking
def strictly_increasing(l):
    """
    :param l: l should be iterable
    :return:
    """
    return all(x<y for x,y in zip(l, l[1:]))


def strictly_decreasing(l):
    return all(x>y for x,y in zip(l, l[1:]))


def non_increasing(l):
    return all(x<=y for x,y in zip(l, l[1:]))


def non_decreasing(l):
    return all(x>=y for x,y in zip(l, l[1:]))

