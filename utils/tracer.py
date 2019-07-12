from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import weakref
from collections import OrderedDict

from utils.register import func_register
from core.base import BaseZhangliang


class Node(object):
    def __init__(self, input_args, input_kwargs, output, op_type):
        self.input_list = tuple(input_args)
        self.input_list_id = tuple([id(an_input) for an_input in self.input_list])
        self.output = output
        self.op_type = op_type
        self.input_kwargs = input_kwargs
        self.op_id = -1

    def set_id(self, id_value):
        self.op_id = id_value

    @property
    def name(self):
        if self.op_id < 0:
            raise ValueError('Node not added to graph.')
        node_name = '{}_{}'.format(self.op_type, self.op_id)
        return node_name


class Graph:
    def __init__(self):
        self._op_count = dict()
        self._nodes_list = OrderedDict()
        self._topo = OrderedDict()

    def append_node(self, node: Node):
        node_type = node.op_type
        count = self._op_count.setdefault(node_type, 0)
        node.set_id(count)
        self._op_count[node_type] += 1
        self._nodes_list[node.name] = node

    def toposort(self):
        for k, node_ in reversed(self._nodes_list.items()):
            parents = []
            for j, node_b in reversed(self._nodes_list.items()):
                if id(node_b.output) in node_.input_list_id:
                    parents.append(j)
                if len(parents) == len(node_.input_list):
                    break
            self._topo[k] = parents

    def clear_graph(self):
        self._op_count.clear()
        self._nodes_list.clear()
        self._topo.clear()

    def check_exist(self, key):
        return key in self._nodes_list.keys()


def create_tracer(graph_: Graph):
    def trace_with_name(op_name):
        @func_register(op_name=op_name)
        def warp(fn):
            def eval_fn(*args, **kwargs):
                output = fn(*args, **kwargs)
                new_node = Node(input_args=args, input_kwargs=kwargs, output=output, op_type=op_name)
                graph_.append_node(new_node)
                return output
            return eval_fn
        return warp
    return trace_with_name


graph = Graph()
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

