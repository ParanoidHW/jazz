from .tensor import (
    zl_add as add,
    zl_sub as sub,
    zl_mul as mul,
    zl_truediv as div,
    zl_matmul as matmul,
    zl_elt_and as elt_and,
    zl_elt_not as elt_not,
    zl_elt_or as elt_or,
    zl_elt_xor as elt_xor,
    zl_abs as abs,
    zl_pow as pow,
    zl_log as log,
    zl_sin as sin,
    zl_cos as cos,
    zl_tan as tan,
    zl_eq as eq,
    zl_ne as ne,
    zl_ge as ge,
    zl_gt as gt,
    zl_le as le,
    zl_lt as lt
)
from .tensor import Zhangliang
from .ops import (
    conv2d,
    conv2d_transpose
)
