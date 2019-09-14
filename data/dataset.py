from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
from data.sampler import Sampler


class DataLoader(object):
    def __init__(self, batchsize, shuffle=True, drop_last=True, sampler=Sampler):
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampler = sampler()

    def get_one_batch(self, indices):
        raise NotImplementedError

    def __iter__(self):
        indices = np.arange(self.numcase)
        while True:
            if self.curr_index + batchsize >= self.numcase:
                self.curr_index = 0
                if always_shuffle:
                    np.random.shuffle(indices)

            idx = indices[self.curr_index:self.curr_index + batchsize]
            self.curr_index += batchsize

            # Fetch data
            yield self.get_one_batch(idx)
