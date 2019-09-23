from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import multiprocessing
import numpy as np
from ..core.tensor import Zhangliang
from .dataset import Dataset
from .sampler import SequentialSampler, RandomSampler


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, num_workers=0, sampler=None):
        if not isinstance(dataset, Dataset):
            raise TypeError('DataLoader requires a `Dataset`, but got {}'.format(type(dataset)))
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if sampler is None:
            if shuffle:
                self.sampler = RandomSampler(len(dataset), batch_size, drop_last)
            else:
                self.sampler = SequentialSampler(len(dataset), batch_size, drop_last)

        if num_workers < 0:
            raise ValueError('num_workers should be greater than 0')
        else:
            self.num_workers = num_workers

        if self.num_workers == 0:
            self.workers = _SingleProcessFetcher(self.dataset)
        else:
            self.workers = _MultiProcessFetcher(self.dataset, self.num_workers)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        return self.make_one_shot_iter()

    def _shutdown_workers(self):
        if self.num_workers > 0:
            # Do multi-process shutdown
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._shutdown_workers()

    def make_infinite_iter(self):
        return _InfiniteIterator(self)

    def make_one_shot_iter(self):
        return _OneShotIterator(self)


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
    def __init__(self, data, num_workers):
        self.dataset = data
        self.workers = []
        self.num_workers = num_workers
        self.data_queue = multiprocessing.Queue()

        self.index_queue = []
        for i in range(self.num_workers):
            w_indices_queue = multiprocessing.Queue()
            w = multiprocessing.Process(target=_fetch_processing,
                                        args=(self.dataset, w_indices_queue, self.data_queue))
            w.daemon = True
            w.start()
            self.index_queue.append(w_indices_queue)

    def fetch(self, indices):
        pass


class Iterator(object):
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


class _OneShotIterator(Iterator):
    def __init__(self, data_loader):
        super(_OneShotIterator, self).__init__()
        self.data_loader = data_loader
        self.sampler = data_loader.sampler
        self.workers = data_loader.workers
        self.num_batches = len(data_loader.sampler)
        self.cur_batch = 0

    def __next__(self):
        if self.cur_batch < self.num_batches:
            indices = self.sampler.get_one_batch()
            self.cur_batch += 1
            return self.workers.fetch(indices)
        else:
            raise StopIteration


class _InfiniteIterator(Iterator):
    def __init__(self, data_loader):
        super(_InfiniteIterator, self).__init__()
        self.data_loader = data_loader
        self.sampler = data_loader.sampler
        self.workers = data_loader.workers

    def __next__(self):
        indices = self.sampler.get_one_batch()
        return self.workers.fetch(indices)

