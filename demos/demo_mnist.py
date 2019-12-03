from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import os

import pyjazz
from pyjazz.core import optim
from pyjazz.utils.manager import AverageMeter
from pyjazz.data.dataset import MNISTDataset


w0 = pyjazz.Parameters(np.random.randn(784, 128) * 0.1, requires_grad=True)
w1 = pyjazz.Parameters(np.random.randn(128, 10) * 0.1, requires_grad=True)
b0 = pyjazz.Parameters(np.zeros((1, 128)), requires_grad=True)
b1 = pyjazz.Parameters(np.zeros((1, 10)), requires_grad=True)


def network(input):
    input = pyjazz.reshape(input, new_shape=(-1, 784))
    y1 = pyjazz.relu(pyjazz.linear(input, w0) + b0)
    logit = pyjazz.linear(y1, w1) + b1
    pred = pyjazz.softmax(logit, dim=1)
    return pred


tr_dataset = MNISTDataset(root_dir=os.path.join('..', 'test_data'))
tr_loader = pyjazz.DataLoader(tr_dataset, batch_size=32)
te_dataset = MNISTDataset(root_dir=os.path.join('..', 'test_data'), split='test')
te_loader = pyjazz.DataLoader(te_dataset, batch_size=32)

optimizer = optim.SGD((w0, w1, b0, b1), lr=1e-2)
acc_meter = AverageMeter()
loss_meter = AverageMeter()
te_acc_meter = AverageMeter()

for epoch in range(100):
    for batch_id, data in enumerate(tr_loader):
        x, y = data
        one_hot = pyjazz.one_hot(y, depth=10)
        pred = network(x)
        loss = pyjazz.cross_entropy(pred, one_hot, dim=1)
        loss = loss.mean().squeeze()
        loss.backward(retain_graph=True)
        optimizer.update()
        optimizer.clear_grad()
        loss_meter.update(loss)

        # Acc
        pred_lb = pyjazz.argmax(pred, dim=1)
        acc = (pred_lb == y).mean().squeeze()
        acc_meter.update(acc)

        if batch_id % 100 == 0:
            print('Epoch {:03d}, Batch {:04d}, Loss {}, Train Acc {}'.
                  format(epoch, batch_id, loss_meter.avg, acc_meter.avg))

    for batch_id, data in enumerate(te_loader):
        x, y = data
        pred = network(x)

        # Acc
        pred_lb = pyjazz.argmax(pred, dim=1)
        acc = (pred_lb == y).mean().squeeze()
        te_acc_meter.update(acc)

        if batch_id % 100 == 0:
            print('Epoch {:03d}, Batch {:04d}, Test Acc {}'.
                  format(epoch, batch_id, te_acc_meter.avg))
