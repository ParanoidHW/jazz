from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import queue


class Dataset(object):
    def __init__(self):
        pass

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ConcatDataset(Dataset):
    def __init__(self, dataset_list):
        super(ConcatDataset, self).__init__()
        for d in dataset_list:
            if not isinstance(d, Dataset):
                raise TypeError('elements in dataset_list should be a `Dataset`, but got {}'.format(type(d)))
        self._num_ds = len(dataset_list)
        self._dataset_list = tuple(dataset_list)
        self._num_cases = tuple(len(d) for d in self._dataset_list)

    def _get_bucket(self, item):
        for i, num in enumerate(self._num_cases):
            if item < num:
                return i, item
            item -= num
        else:
            raise IndexError('item index out of range')

    def __len__(self):
        return sum(self._num_cases)

    def __getitem__(self, item):
        # TODO: how to deal with slicing? PyTorch does not support slicing in Dataset.
        dataset_id, dataset_item = self._get_bucket(item)
        return self._dataset_list[dataset_id].__getitem__(dataset_item)


class FakeDataset(Dataset):
    def __init__(self, dim=28, num_cases=100):
        super(Dataset).__init__()
        self.x = np.random.rand(num_cases, dim)
        self.y = np.random.rand(num_cases, 3, dim, dim)
        self.z = np.random.randint(10, size=(num_cases, ))
        self.num_cases = num_cases

    def __len__(self):
        return self.num_cases

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.z[item]
