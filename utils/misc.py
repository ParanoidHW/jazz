from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numbers
import collections


def additive_broadcast_analysis(input_shapes, output_shape):
    """
    Analyze and determine the broadcast method. Return a tuple of the axes to be reduced
    of the output w.r.t. each input.
    :param input_shapes: a list or tuple of input data shapes
    :param output_shape: the shape of the output
    :return:
    """

    def data_shape_comp(a_shape, b_shape):
        """
        Compare a single shape `a_shape` and another shape `b_shape`， determine whether can be broadcast.
        :param a_shape:
        :param b_shape:
        :return:
        """
        # First check whether all axes reduced.
        axes_to_reduce = []
        incompatible = False
        if len(a_shape) == 1:
            if a_shape[0] == 1:
                axes_to_reduce = tuple(range(len(b_shape)))
                incompatible = False
            elif len(b_shape) > 1:
                axes_to_reduce = []
                incompatible = True
        # Check whether the two shapes have the same length.
        # If not, the broadcast is not possible.
        elif len(a_shape) != len(b_shape):
            axes_to_reduce = []
            incompatible = True
        # Check for each axis in both shapes. If broadcast is satisfied, the different
        # axes in `a_shape` should have the dimension 1.
        else:
            axes_to_reduce = []
            incompatible = False
            for i, (a_axis, b_axis) in enumerate(zip(a_shape, b_shape)):
                if a_axis != b_axis:
                    incompatible = incompatible or (a_axis != 1)
                    axes_to_reduce.append(i)
        return tuple(axes_to_reduce), incompatible

    reduced_axes = []
    for input_shape in input_shapes:
        red_ax, invalid = data_shape_comp(input_shape, output_shape)
        if not invalid:
            reduced_axes.append(red_ax)
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

    def data_shape_comp(a_shape, b_shape):
        """
        Compare a single shape `a_shape` and another shape `b_shape` can be broadcast.
        :param a_shape:
        :param b_shape:
        :return:
        """
        a_dim = len(a_shape)
        b_dim = len(b_shape)
        # In case of (m, n) X (n, ) = (m, )
        if a_dim > b_dim:
            b_shape = b_shape + (1, )

        # First check whether all axes reduced.
        if a_dim == 1:
            axes_to_reduce = tuple(range(b_dim))
            incompatible = False
        else:
            axes_to_reduce = []
            incompatible = False

            if b_dim > a_dim:
                axes_to_reduce = list(range(b_dim-a_dim))
            for i in range(a_dim-2):
                neg_idx = i - a_dim
                if a_shape[neg_idx] != b_shape[neg_idx]:
                    incompatible = a_shape[neg_idx] != 1
                    if a_shape[neg_idx] == 1:
                        axes_to_reduce.append(i)
        return tuple(axes_to_reduce), incompatible

    reduced_axes = []
    for input_shape in input_shapes:
        red_ax, invalid = data_shape_comp(input_shape, output_shape)
        if not invalid:
            reduced_axes.append(red_ax)
        else:
            raise ArithmeticError

    return reduced_axes


def recover_dim(ori_shape, tar_shape, dim=None, keepdims=False):
    new_shape = list(ori_shape)
    if keepdims:
        return new_shape

    if dim is not None:
        if isinstance(dim, (list, tuple)) or isinstance(dim ,collections.Iterable):
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


if __name__ == '__main__':
    import numpy as np

    def eltwise(ashape, bshape):
        a = np.ones(ashape)
        b = np.ones(bshape)
        c = a + b
        return c.shape

    def mult(ashape, bshape):
        a = np.ones(ashape)
        b = np.ones(bshape)
        c = np.matmul(a, b)
        return c.shape

    def check_elt_shape(ashape, bshape):
        try:
            print('-----------------------------------------------------------------------------------')
            elt_shape = eltwise(ashape, bshape)
            print('NP allowed elt operator for {} and {}.'.format(ashape, bshape))
            axes_to_reduce = additive_broadcast_analysis((a_shape, b_shape), elt_shape)
            print('{} + {} = {}'.format(ashape, bshape, elt_shape))
            output_info_pattern = '\tShape A: {}, axes to reduce: {}'.format(ashape, axes_to_reduce[0])
            print(output_info_pattern)
            output_info_pattern = '\tShape B: {}, axes to reduce: {}'.format(bshape, axes_to_reduce[1])
            print(output_info_pattern)
        except ValueError:
            print('NP elt operator for {} and {} failed.'.format(ashape, bshape))
        except ArithmeticError:
            print('Something wrong with analysis elt operator for {} and {}.'.format(ashape, bshape))


    def check_prod_shape(ashape, bshape):
        try:
            print('-----------------------------------------------------------------------------------')
            prod_shape = mult(ashape, bshape)
            print('NP allowed prod operator for {} and {}.'.format(ashape, bshape))
            axes_to_reduce = multiplicative_broadcast_analysis((a_shape, b_shape), prod_shape)
            print('{} X {} = {}'.format(ashape, bshape, prod_shape))
            output_info_pattern = '\tShape A: {}, axes to reduce: {}'.format(ashape, axes_to_reduce[0])
            print(output_info_pattern)
            output_info_pattern = '\tShape B: {}, axes to reduce: {}'.format(bshape, axes_to_reduce[1])
            print(output_info_pattern)
        except ValueError:
            print('NP prod operator for {} and {} failed.'.format(ashape, bshape))
        except ArithmeticError:
            print('Something wrong with analysis prod operator for {} and {}.'.format(ashape, bshape))


    a_shape = (2, 3, 4, 5)

    # element-wise
    b_shape = (1, )
    check_elt_shape(a_shape, b_shape)

    b_shape = (1, 1, 1, 5)
    check_elt_shape(a_shape, b_shape)

    b_shape = (2, 1, 4, 1)
    check_elt_shape(a_shape, b_shape)

    b_shape = (2, 3, 4, 5)
    check_elt_shape(a_shape, b_shape)

    b_shape = (2, )
    check_elt_shape(a_shape, b_shape)

    # matmul
    b_shape = (1,)
    check_prod_shape(a_shape, b_shape)

    b_shape = (5,)
    check_prod_shape(a_shape, b_shape)

    b_shape = (5, 6)
    check_prod_shape(a_shape, b_shape)

    b_shape = (3, 5, 6)
    check_prod_shape(a_shape, b_shape)

    b_shape = (1, 1, 5, 6)
    check_prod_shape(a_shape, b_shape)

    b_shape = (2, 1, 5, 6)
    check_prod_shape(a_shape, b_shape)

    b_shape = (1, 3, 5, 6)
    check_prod_shape(a_shape, b_shape)

    b_shape = (2, 3, 4, 2, 3, 5, 6)
    check_prod_shape(a_shape, b_shape)

    a_shape = (2, 3, 4, 2, 3, 5, 6)
    b_shape = (2, 1, 4, 5)
    check_prod_shape(a_shape, b_shape)

    a_shape = (5, )
    b_shape = (2, 3, 1, 5)
    check_prod_shape(a_shape, b_shape)
