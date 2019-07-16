from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np


def test_broadcast_rule():
    from utils.misc import additive_broadcast_analysis, multiplicative_broadcast_analysis

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

    b_shape = (5,)
    check_elt_shape(a_shape, b_shape)

    a_shape = (1,)
    b_shape = (2,)
    check_elt_shape(a_shape, b_shape)

    # matmul
    a_shape = (2, 3, 4, 5)
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

    a_shape = (1,)
    b_shape = (2,)
    check_prod_shape(a_shape, b_shape)
