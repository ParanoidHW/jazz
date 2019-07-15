from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


def test_check_forward_func_and_backkward_func():
    from utils import func_lib, grad_lib
    forward_keys = func_lib.keys()
    backward_keys = grad_lib.keys()
    diff = set(forward_keys) - set(backward_keys)
    assert len(diff) == 0
