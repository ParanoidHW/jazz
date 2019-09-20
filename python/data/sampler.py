from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np


class Sampler(object):
    def __init__(self, numcase, batchsize, drop_last=False, *args, **kwargs):
        self.numcase = numcase
        self.drop_last = drop_last
        self.batchsize = batchsize

        numbatches = self.numcase // self.batchsize
        if numbatches * self.batchsize < self.numcase and not self.drop_last:
            numbatches += 1
        self.numbatches = numbatches
        self._lap_finished = False

    @property
    def one_lap(self):
        return self._lap_finished

    def __len__(self):
        return self.numbatches

    def get_one_batch(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, numcase, batchsize, drop_last=False):
        super(SequentialSampler, self).__init__(numcase, batchsize, drop_last)
        self.indices = np.arange(numcase)
        self.cur_pos = 0

    def get_one_batch(self):
        self._lap_finished = False
        if self.cur_pos + self.batchsize >= self.numcase:
            if self.drop_last:
                fetch = self.indices[0:self.batchsize]
                self.cur_pos = self.batchsize
            else:
                fetch = self.indices[self.cur_pos:self.numcase]
                self.cur_pos = 0
            self._lap_finished = True
        else:
            fetch = self.indices[self.cur_pos:self.cur_pos + self.batchsize]
            self.cur_pos += self.batchsize
        return fetch


class RandomSampler(Sampler):
    def __init__(self, numcase, batchsize, drop_last=False, shuffle_every_epoch=False):
        super(RandomSampler, self).__init__(numcase, batchsize, drop_last)
        self.indices = np.random.permutation(numcase)
        self.always_shuffle = shuffle_every_epoch
        self.cur_pos = 0

    def get_one_batch(self):
        self._lap_finished = False
        if self.cur_pos + self.batchsize >= self.numcase:
            if self.drop_last:
                fetch = self.indices[0:self.batchsize]
                self.cur_pos = self.batchsize
            else:
                fetch = self.indices[self.cur_pos:self.numcase]
                self.cur_pos = 0
            if self.always_shuffle:
                np.random.shuffle(self.indices)
            self._lap_finished = True
        else:
            fetch = self.indices[self.cur_pos:self.cur_pos + self.batchsize]
            self.cur_pos += self.batchsize
        return fetch
