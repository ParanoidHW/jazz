from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import queue

from python.data.sampler import Sampler
from python.core import Zhangliang


class DataLoader(object):
    def __init__(self, batchsize, shuffle=True, drop_last=True, numworkers=1, sampler=Sampler()):
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampler = sampler
        self.numwokers = numworkers

    def get_one_batch(self):
        indices = self.sampler.get_one_batch(batchsize=self.batchsize)
        for i, ind in enumerate(indices):
            if ind > self.numcases:
                raise IndexError('Index {} out of range {}.'.format(ind, self.numcases))
            data = self.get_one_sample(ind)
            if i == 0:
                batch_data = list(data)
            else:
                for data_id, d in enumerate(data):
                    batch_data[data_id].append(d)

        batch_data_zl = []
        for d in batch_data:
            # each value in d should be [b, c, h, w] shape.
            d = np.concatenate(d, axis=0)
            batch_data_zl.append(Zhangliang(d, dtype=d.dtype, requires_grad=False))
        return tuple(batch_data_zl)

    def get_one_sample(self, index):
        raise NotImplementedError

    @property
    def numcases(self):
        raise NotImplementedError

    @property
    def numbatches(self):
        numcases = self.numcases
        numbatches = numcases // self.batchsize
        if numbatches * self.batchsize < numcases and not self.drop_last:
            numbatches += 1
        return numbatches

    def __iter__(self):
        while True:
            yield self.get_one_batch()
