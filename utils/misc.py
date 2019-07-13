from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


def additive_broadcast_analysis(input_shapes, output_shape):
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
        # First check whether all axes reduced.
        axes_to_reduce = []
        imcompatible = False
        if len(a_shape) == 1:
            if a_shape[0] == 1:
                axes_to_reduce = tuple(range(len(b_shape)))
                imcompatible = False
            elif len(b_shape) > 1:
                axes_to_reduce = []
                imcompatible = True
        # Check whether the two shapes have the same length.
        # If not, the broadcast is not possible.
        elif len(a_shape) != len(b_shape):
            axes_to_reduce = []
            imcompatible = True
        # Check for each axis in both shapes. If broadcast is satisfied, the different
        # axes in `a_shape` should have the dimension 1.
        else:
            axes_to_reduce = []
            imcompatible = False
            for i, (a_axis, b_axis) in enumerate(zip(a_shape, b_shape)):
                if a_axis != b_axis:
                    imcompatible = imcompatible or (a_axis != 1)
                    axes_to_reduce.append(i)
        return axes_to_reduce, imcompatible

    reduced_axes = []
    for input_shape in input_shapes:
        red_ax, valid = data_shape_comp(input_shape, output_shape)
        if valid:
            reduced_axes.append(red_ax)
        else:
            raise ValueError

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
        # First check whether all axes reduced.
        axes_to_reduce = []
        imcompatible = False
        if len(a_shape) == 1:
            if a_shape[0] == 1:
                axes_to_reduce = tuple(range(len(b_shape)))
                imcompatible = False
            elif len(b_shape) > 1:
                axes_to_reduce = []
                imcompatible = True
        # Check whether the two shapes have the same length.
        # If not, the broadcast is not possible.
        elif len(a_shape) != len(b_shape):
            axes_to_reduce = []
            imcompatible = True
        # Check for each axis in both shapes. If broadcast is satisfied, the different
        # axes in `a_shape` should have the dimension 1.
        else:
            axes_to_reduce = []
            imcompatible = False
            for i, (a_axis, b_axis) in enumerate(zip(a_shape, b_shape)):
                if a_axis != b_axis:
                    imcompatible = imcompatible or (a_axis != 1)
                    axes_to_reduce.append(i)
        return axes_to_reduce, imcompatible

    reduced_axes = []
    for input_shape in input_shapes:
        red_ax, valid = data_shape_comp(input_shape, output_shape)
        if valid:
            reduced_axes.append(red_ax)
        else:
            raise ValueError

    return reduced_axes
