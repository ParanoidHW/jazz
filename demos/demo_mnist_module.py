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


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, s=2, p=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.re = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, s=2, p=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(49*32, 10)
        self.sm = nn.Softmax()

    def forward(self, x):
        b = x.shape[0]
        x = pyjazz.reshape(x, (b, 1, 28, 28))
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.re(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.re(y)
        y = pyjazz.reshape(y, (b, -1))
        y = self.fc(y)
        pred = self.sm(y)
        return pred


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.layers = nn.Stacked(
            nn.Conv2d(1, 16, 3, s=2, p=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, s=2, p=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        self.fc = nn.Linear(49 * 32, 10)
        self.sm = nn.Softmax()

    def forward(self, x):
        b = x.shape[0]
        x = pyjazz.reshape(x, (b, 1, 28, 28))
        y = self.layers(x)
        y = pyjazz.reshape(y, (b, -1))
        y = self.fc(y)
        pred = self.sm(y)
        return pred


tr_dataset = MNISTDataset(root_dir=os.path.join('..', 'test_data'))
tr_loader = pyjazz.DataLoader(tr_dataset, batch_size=32)
te_dataset = MNISTDataset(root_dir=os.path.join('..', 'test_data'), split='test')
te_loader = pyjazz.DataLoader(te_dataset, batch_size=32)

# net = Net1()
# net = Net2()
net = Net4()
optimizer = optim.SGD(net.parameters, lr=1e-2)
acc_meter = AverageMeter()
loss_meter = AverageMeter()
te_acc_meter = AverageMeter()

for epoch in range(100):
    acc_meter.reset()
    loss_meter.reset()
    te_acc_meter.reset()
    for batch_id, data in enumerate(tr_loader):
        x, y = data
        one_hot = pyjazz.one_hot(y, depth=10)
        pred = net(x)
        loss = pyjazz.cross_entropy(pred, one_hot, dim=1)
        loss = loss.mean()
        loss.backward(retain_graph=True)
        optimizer.update()
        optimizer.clear_grad()
        loss_meter.update(loss.item())

        # Acc
        pred_lb = pyjazz.argmax(pred, dim=1)
        acc = (pred_lb == y).mean().item()
        acc_meter.update(acc)

        if batch_id % 100 == 0:
            print(f'Epoch {epoch:03d}, Batch {batch_id:04d}, Loss (avg) {loss_meter.avg:.5f}, '
                  f'Loss (cur) {loss_meter.val:.5f}, '
                  f'Train Acc (avg) {acc_meter.avg:.5f}, Train Acc (cur) {acc_meter.val:.5f}')

    net.train(False)
    for batch_id, data in enumerate(te_loader):
        x, y = data
        pred = net(x)

        # Acc
        pred_lb = pyjazz.argmax(pred, dim=1)
        acc = (pred_lb == y).mean().item()
        te_acc_meter.update(acc)

        if batch_id % 100 == 0:
            print(f'Epoch {epoch:03d}, Batch {batch_id:04d}, '
                  f'Test Acc (avg) {te_acc_meter.avg:.5f}, Test Acc (cur) {te_acc_meter.val:.5f}')
