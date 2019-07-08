from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import weakref
from .register import func_register
from core.base import BaseZhangliang


class Node(object):
    ID = 0

    def __init__(self, input_args, input_kwargs, output_zl, op_name, node_name=None):
        self.parents = []
        for node in input_args:
            if isinstance(node, BaseZhangliang):
                self.parents.append(weakref.ref(node))
            else:
                self.inputs.append(node)
        self.inputs = tuple(self.inputs)
        self.input_kwargs = input_kwargs
        self.op_name = op_name

        if node_name is None:
            node_name = '{}_{}'.format(op_name, self.ID)
        self.ID += 1
        self.name = node_name
        self.output_zl = output_zl


class Graph:
    def __init__(self):
        self._ops = dict()
        self._nodes = []

    def insert_node(self, node, op_type, name):
        _op_count = self._ops.setdefault(op_type, 0)
        self._ops[op_type] += 1
        self._nodes.append(node)

    def clear_graph(self):
        self._ops = dict()
        self._nodes = []


def create_tracer(dict_):
    def trace_with_name(op_name):
        @func_register(op_name=op_name)
        def warp(fn):
            def eval_fn(*args, **kwargs):
                output = fn(*args, **kwargs)
                new_node = Node(input_args=args, input_kwargs=kwargs, op_name=op_name)
                if 'name' in kwargs.keys():
                    if kwargs['name'] in dict_:
                        raise ValueError('Duplicate node name {} in computation graph.'.format(kwargs['name']))
                    else:
                        dict_[kwargs['name']] = new_node
                else:
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

    @trace(op_name='add')
    def temp_add(x, y):
        out = x + y
        return out

    z1 = temp_add(a, b)
    z2 = temp_add(z1, b)
    print(z2)

