from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np


class Sampler(object):
    def __init__(self, num_cases, batch_size, drop_last=False, *args, **kwargs):
        self.num_cases = num_cases
        self.drop_last = drop_last
        self.batch_size = batch_size

        num_batches = self.num_cases // self.batch_size
        if num_batches * self.batch_size < self.num_cases and not self.drop_last:
            num_batches += 1
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def get_one_batch(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, num_cases, batch_size, drop_last=False):
        super(SequentialSampler, self).__init__(num_cases, batch_size, drop_last)
        self.indices = np.arange(num_cases)
        self.cur_pos = 0

    def get_one_batch(self):
        if self.cur_pos + self.batch_size >= self.num_cases:
            if self.drop_last:
                fetch = self.indices[0:self.batch_size]
                self.cur_pos = self.batch_size
            else:
                fetch = self.indices[self.cur_pos:self.num_cases]
                self.cur_pos = 0
        else:
            fetch = self.indices[self.cur_pos:self.cur_pos + self.batch_size]
            self.cur_pos += self.batch_size
        return fetch


class RandomSampler(Sampler):
    def __init__(self, num_cases, batch_size, drop_last=False, shuffle_every_epoch=False):
        super(RandomSampler, self).__init__(num_cases, batch_size, drop_last)
        self.indices = np.random.permutation(num_cases)
        self.always_shuffle = shuffle_every_epoch
        self.cur_pos = 0

    def get_one_batch(self):
        if self.cur_pos + self.batch_size >= self.num_cases:
            if self.drop_last:
                fetch = self.indices[0:self.batch_size]
                self.cur_pos = self.batch_size
            else:
                fetch = self.indices[self.cur_pos:self.num_cases]
                self.cur_pos = 0
            if self.always_shuffle:
                np.random.shuffle(self.indices)
        else:
            fetch = self.indices[self.cur_pos:self.cur_pos + self.batch_size]
            self.cur_pos += self.batch_size
        return fetch
