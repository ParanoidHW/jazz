from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import multiprocessing
import numpy as np
from ..core.tensor import Zhangliang
from .dataset import Dataset
from .sampler import SequentialSampler, RandomSampler


class DataLoader(object):
    def __init__(self, dataset, batchsize, shuffle=True, drop_last=True, numworkers=0, sampler=None):
        if not isinstance(dataset, Dataset):
            raise TypeError('DataLoader requires a `Dataset`, but got {}'.format(type(dataset)))
        self.dataset = dataset
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.drop_last = drop_last

        if sampler is None:
            if shuffle:
                self.sampler = RandomSampler(len(dataset), batchsize, drop_last)
            else:
                self.sampler = SequentialSampler(len(dataset), batchsize, drop_last)

        if numworkers < 0:
            raise ValueError('numworkers should be greater than 0')
        else:
            self.numworkers = numworkers

        if self.numworkers == 0:
            self.workers = _SingleProcessFetcher(self.dataset)
        else:
            self.workers = _MultiProcessFetcher(self.dataset, self.numworkers)

    def __len__(self):
        return len(self.sampler)

    def __next__(self):
        if self.sampler.one_lap:  # signal to stop the epoch. Sampler.fetch() is an infinite iterable process.
            raise StopIteration
        indices = self.sampler.get_one_batch()
        return self.workers.fetch(indices)

    def __iter__(self):
        return self

    def _shutdown_workers(self):
        if self.numworkers > 0:
            # Do multi-process shutdown
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._shutdown_workers()


def _fetch(dataset, indices):
    batch_data_zl = []
    for i, index in enumerate(indices):
        data = dataset[index]
        if not isinstance(data, (tuple, list)):
            data = (data, )
        if i == 0:
            batch_data_zl = [[] for _ in range(len(data))]

        for j, d in enumerate(data):
            batch_data_zl[j].append(d)

    for j, d in enumerate(batch_data_zl):
        # each value of `d` should be in shape [C, D0, D1, D2, ...]
        d = np.stack(d, axis=0)
        batch_data_zl[j] = Zhangliang(d, dtype=d.dtype, requires_grad=False)
    return tuple(batch_data_zl)


class _SingleProcessFetcher(object):
    def __init__(self, data):
        self.dataset = data

    def fetch(self, indices):
        return _fetch(self.dataset, indices)


def _fetch_processing(dataset, indices_queue, data_queue):
    assert isinstance(indices_queue, multiprocessing.Queue)
    pass


class _MultiProcessFetcher(object):
    def __init__(self, data, numworkers):
        self.dataset = data
        self.workers = []
        self.numworkers = numworkers
        self.data_queue = multiprocessing.Queue()

        self.index_queue = []
        for i in range(self.numworkers):
            w_indices_queue = multiprocessing.Queue()
            w = multiprocessing.Process(target=_fetch_processing,
                                        args=(self.dataset, w_indices_queue, self.data_queue))
            w.daemon = True
            w.start()
            self.index_queue.append(w_indices_queue)

    def fetch(self, indices):
        pass
