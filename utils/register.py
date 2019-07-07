from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


def create_register(dict_):
    def register(key):
        def _(fn):
            dict_[key] = fn
            return fn
        return _
    return register


grad_lib = {}
grad_register = create_register(grad_lib)

