from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from utils.tracer import graph


class no_grad(object):
    def __init__(self):
        self.prev_state = graph.is_grad_enabled()

    def __enter__(self):
        self.prev_state = graph.is_grad_enabled()
        graph.set_grad_enable(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        graph.set_grad_enable(self.prev_state)
        return False


class has_grad(object):
    def __init__(self):
        self.prev_state = graph.is_grad_enabled()

    def __enter__(self):
        self.prev_state = graph.is_grad_enabled()
        graph.set_grad_enable(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        graph.set_grad_enable(self.prev_state)
        return False
