from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import os

import pyjazz
from pyjazz.core.optim import SGD
from pyjazz.data.dataset import MNISTDataset


w0 = pyjazz.Parameters(np.random.randn(784, 128) * 0.1, requires_grad=True)
w1 = pyjazz.Parameters(np.random.randn(128, 10) * 0.1, requires_grad=True)
b0 = pyjazz.Parameters(np.zeros((1, 128)), requires_grad=True)
b1 = pyjazz.Parameters(np.zeros((1, 10)), requires_grad=True)


def network(input):
    fc_fn = pyjazz.func_lib['linear']
    sf_fn = pyjazz.func_lib['softmax']
    re_fn = pyjazz.func_lib['relu']
    rs_fn = pyjazz.func_lib['reshape']

    input = rs_fn(input, new_shape=(-1, 784))
    y1 = re_fn(fc_fn(input, w0) + b0)
    # y2 = re_fn(fc_fn(y1, w1) + b1)
    logit = fc_fn(y1, w1) + b1
    pred = sf_fn(logit, dim=1)
    return pred


tr_dataset = MNISTDataset(root_dir=os.path.join('..', 'test_data'))
tr_loader = pyjazz.DataLoader(tr_dataset, batch_size=32)
te_dataset = MNISTDataset(root_dir=os.path.join('..', 'test_data'), split='test')
te_loader = pyjazz.DataLoader(te_dataset, batch_size=32)

optimizer = SGD((w0, w1, b0, b1), lr=1e-2)
oh_fn = pyjazz.func_lib['one_hot']
# sq_fn = pyjazz.func_lib['square']
armax_fn = pyjazz.func_lib['argmax']
ce_fn = pyjazz.func_lib['cross_entropy']


for epoch in range(100):
    for batch_id, data in enumerate(tr_loader):
        x, y = data
        one_hot = oh_fn(y, depth=10)
        pred = network(x)
        loss = ce_fn(pred, one_hot, dim=1)
        loss = loss.mean()
        loss.backward(retain_graph=True)
        optimizer.update()
        optimizer.clear_grad()

        # Acc
        pred_lb = armax_fn(pred, dim=1)
        acc = (pred_lb == y).mean()

        if batch_id % 100 == 0:
            print('Epoch {:03d}, Batch {:04d}, Loss {}, Acc {}'.
                  format(epoch, batch_id, loss.squeeze(), acc.squeeze()))
