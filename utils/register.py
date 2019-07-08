from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


def create_register(dict_):
    def register(op_name):
        def _(fn):
            dict_[op_name] = fn
            return fn
        return _
    return register


grad_lib = {}
grad_register = create_register(grad_lib)

func_lib = {}
func_register = create_register(func_lib)
