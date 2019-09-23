from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from python.data.dataset import Dataset, FakeDataset
from python.data.dataloader import DataLoader


def test_dataloader():
    fake_data = FakeDataset(num_cases=40)
    data_loader = DataLoader(fake_data, batch_size=8, shuffle=True, num_workers=0)
    for epoch in range(2):
        for i, data in enumerate(data_loader):
            x, y, z = data
            if i == 0:
                print(x.shape)
                print(y.shape)
                print(z.shape)
        print(i)

    data_iter = data_loader.make_infinite_iter()
    for it in range(10):
        x, y, z = next(data_iter)
        if it == 0:
            print(x.shape)
            print(y.shape)
            print(z.shape)
    print(it)


