from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from python.data.dataset import Dataset
from python.data.dataloader import DataLoader


def test_dataloader():
    class FakeData(Dataset):
        def __init__(self, dim=100, numcase=40):
            super(Dataset).__init__()
            self.x = np.random.rand(numcase, dim)
            self.y = np.random.rand(numcase, 1, dim, dim)
            self.numcase = numcase
            self.z = np.random.rand(numcase)

        def __len__(self):
            return self.numcase

        def __getitem__(self, item):
            return self.x[item], self.y[item], self.z[item]

    fake_data = FakeData()
    data_loader = DataLoader(fake_data, batchsize=8, shuffle=True, numworkers=0)
    for i, data in enumerate(data_loader):
        x, y, z = data
        if i == 0:
            print(x.shape)
            print(y.shape)
            print(z.shape)
    print(i)



