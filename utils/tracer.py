from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from collections import OrderedDict

from utils.register import func_register, grad_lib


class Node(object):
    def __init__(self, input_args, input_kwargs, output, op_type):
        self.input_list = tuple(input_args)
        self.input_list_id = tuple([id(an_input) for an_input in self.input_list])
        if isinstance(output, tuple):
            self.output = output
        else:
            self.output = (output, )
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

    def backprop(self):
        grad_fn = grad_lib[self.op_type]
        grad_fn(*self.output, *self.input_list, **self.input_kwargs)


class Graph:
    def __init__(self):
        self._op_count = dict()
        self._nodes_by_name = OrderedDict()
        self._nodes_by_id = OrderedDict()
        self._topo = OrderedDict()
        self._ctx_requires_grad = True

    def is_grad_enabled(self):
        return self._ctx_requires_grad

    def set_grad_enable(self, enabled=True):
        self._ctx_requires_grad = enabled

    def is_initialized(self):
        return len(self._topo) != 0

    def is_leaf(self, tensor):
        node = self.get_node_by_output_tensor(tensor)
        return list(self._topo.items())[0][0] == node.name

    def get_node_by_output_tensor(self, tensor):
        query_id = id(tensor)
        node = self._nodes_by_id[query_id]
        return node

    def get_parents(self, node):
        if not self.is_initialized():
            self.toposort()
        parent_name = self._topo[node.name]
        parent_nodes = [self._nodes_by_name[p] for p in parent_name]
        return parent_nodes

    def append_node(self, node: Node):
        node_type = node.op_type
        count = self._op_count.setdefault(node_type, 0)
        node.set_id(count)
        self._op_count[node_type] += 1

        # Index node by the op name
        self._nodes_by_name[node.name] = node

        # Index node by the output id
        for o in node.output:
            self._nodes_by_id[id(o)] = node

    def toposort(self):
        for k, node_ in reversed(self._nodes_by_name.items()):
            parents = []
            for j, node_b in reversed(self._nodes_by_name.items()):
                output = node_b.output[0]
                if id(output) in node_.input_list_id:
                    parents.append(j)
                if len(parents) == len(node_.input_list):
                    break
            self._topo[k] = parents

    def clear_graph(self):
        self._op_count.clear()
        self._nodes_by_name.clear()
        self._topo.clear()


def create_tracer(graph_: Graph):
    def trace_with_name(op_name):
        def warp(fn):
            @func_register(op_name=op_name)
            def eval_fn(*args, **kwargs):
                output = fn(*args, **kwargs)
                new_node = Node(input_args=args, input_kwargs=kwargs, output=output, op_type=op_name)
                if graph_.is_grad_enabled():
                    graph_.append_node(new_node)
                return output
            return eval_fn
        return warp
    return trace_with_name


graph = Graph()
ctx_register = create_tracer(graph)


if __name__ == '__main__':
    a = 2
    b = 3

    @ctx_register(op_name='add')
    def temp_add(x, y):
        out = x + y
        return out

    z1 = temp_add(a, b)
    # z2 = temp_add(z1, b)
    print(z1)

