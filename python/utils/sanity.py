from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


class TypeCheck:
    def __init__(self, source_var, target_type_or_cls, info=None):
        if not isinstance(source_var, target_type_or_cls):
            if info is None:
                raise TypeError('Variable should be a {}'.format(target_type_or_cls))
            else:
                raise TypeError(info)


class ShapeCheck:
    def __init__(self, source_var, target_shape, info=None):
        if 'shape' not in source_var.__dict__:
            raise TypeError('Expect a Zhangliang but got {}'.format(type(source_var)))

        if tuple(source_var.shape) != target_shape:
            if info is None:
                raise ValueError('Variable shape should be {}, but got {}.'.
                                 format(target_shape, tuple(source_var.shape)))
            else:
                raise ValueError(info)


class ClosedRangeCheck:
    def __init__(self, var, xmin=None, xmax=None, info=None):
        if xmin is None and xmax is None:
            raise ValueError('One of the arguments `xmin` and `xmax` should be non-None.')
        elif xmin is None:
            valid = var <= xmax
            xr = '(-inf, {}]'.format(xmax)
        elif xmax is None:
            valid = var >= xmin
            xr = '[{}, inf)'.format(xmin)
        else:
            valid = (var <= xmax) & (var >= xmin)
            xr = '[{}, {}]'.format(xmin, xmax)

        if not valid:
            if info is None:
                raise ValueError('Variable not in range {}.'.format(xr))
            else:
                raise ValueError(info)
