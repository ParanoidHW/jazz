from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import weakref
from utils.register import func_register
from core.base import BaseZhangliang


class Node(object):
    def __init__(self, input_args, input_kwargs, output, op_name):
        self.inputs = []
        for node in input_args:
            self.inputs.append(node)
        self.inputs = tuple(self.inputs)
        self.input_kwargs = input_kwargs
        self.output = output
        self.op_name = op_name


class Graph:
    def __init__(self):
        self._ops = dict()
        self._nodes = []
        self.output_mapping = dict()

    def insert_node(self, node, op_type):
        _op_count = self._ops.setdefault(op_type, 0)
        node_name = '{}_{}'.format(op_type, _op_count)
        self._ops[op_type] += 1
        self._nodes.append(node)
        self.output_mapping[node.id] = node

    def clear_graph(self):
        self._ops = dict()
        self._nodes = []
        self.output_mapping = dict()


def create_tracer(graph_: Graph):
    def trace_with_name(op_name):
        @func_register(op_name=op_name)
        def warp(fn):
            def eval_fn(*args, **kwargs):
                output = fn(*args, **kwargs)
                parents_node = []
                new_node = Node(input_args=args, input_kwargs=kwargs, outputs=output, op_name=op_name)
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
    # z2 = temp_add(z1, b)
    print(z1)

