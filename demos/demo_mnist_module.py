from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import os

import pyjazz
import pyjazz.nn as nn
from pyjazz.core import optim
from pyjazz.utils.manager import AverageMeter
from pyjazz.data.dataset import MNISTDataset


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(784, 100, bias=True)
        self.re = nn.ReLU()
        self.fc2 = nn.Linear(100, 10, bias=True)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = pyjazz.reshape(x, new_shape=(-1, 784))
        y = self.fc1(x)
        y = self.re(y)
        y = self.fc2(y)
        y = self.sm(y, dim=-1)
        return y


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, s=2, p=1)
        self.re = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, s=2, p=1)
        self.fc = nn.Linear(49*32, 10)
        self.sm = nn.Softmax()

    def forward(self, x):
        b = x.shape[0]
        x = pyjazz.reshape(x, (b, 1, 28, 28))
        y = self.conv1(x)
        y = self.re(y)
        y = self.conv2(y)
        y = self.re(y)
        y = pyjazz.reshape(y, (b, -1))
        y = self.fc(y)
        pred = self.sm(y)
        return pred


tr_dataset = MNISTDataset(root_dir=os.path.join('..', 'test_data'))
tr_loader = pyjazz.DataLoader(tr_dataset, batch_size=32)
te_dataset = MNISTDataset(root_dir=os.path.join('..', 'test_data'), split='test')
te_loader = pyjazz.DataLoader(te_dataset, batch_size=32)

# net = Net1()
net = Net2()
optimizer = optim.SGD(net.parameters, lr=1e-2)
acc_meter = AverageMeter()
loss_meter = AverageMeter()
te_acc_meter = AverageMeter()

for epoch in range(100):
    for batch_id, data in enumerate(tr_loader):
        x, y = data
        one_hot = pyjazz.one_hot(y, depth=10)
        pred = net(x)
        loss = pyjazz.cross_entropy(pred, one_hot, dim=1)
        loss = loss.mean()
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
        pred = net(x)

        # Acc
        pred_lb = pyjazz.argmax(pred, dim=1)
        acc = (pred_lb == y).mean().squeeze()
        te_acc_meter.update(acc)

        if batch_id % 100 == 0:
            print('Epoch {:03d}, Batch {:04d}, Test Acc {}'.
                  format(epoch, batch_id, te_acc_meter.avg))
