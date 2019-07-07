from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import weakref


class Node(object):
    ID = 0

    def __init__(self, input_args, input_kwargs, op_name, node_name=None):
        self.inputs = []
        for node in input_args:
            try:
                self.inputs.append(weakref.ref(node))
            except TypeError:
                self.inputs.append(node)
        self.inputs = tuple(self.inputs)
        self.input_kwargs = input_kwargs
        self.op_name = op_name

        if node_name is None:
            node_name = '{}_{}'.format(op_name, self.ID)
        self.ID += 1
        self.name = node_name


def create_tracer(dict_):
    def trace_with_name(name):
        def warp(fn):
            def eval_fn(*args, **kwargs):
                output = fn(*args, **kwargs)
                new_node = Node(input_args=args, input_kwargs=kwargs, op_name=name)
                if new_node.name in dict_:
                    raise ValueError('Duplicate node name {} in computation graph.'.format(new_node.name))
                # list_.append(weakref.ref(new_node))
                dict_[new_node.name] = new_node
                return output
            return eval_fn
        return warp
    return trace_with_name


graph = {}
trace = create_tracer(graph)


if __name__ == '__main__':
    a = 2
    b = 3

    @trace(name='add')
    def temp_add(x, y):
        out = x + y
        return out

    z = temp_add(a, b)
    print(z)

