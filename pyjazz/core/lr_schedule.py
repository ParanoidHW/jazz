from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from pyjazz.utils.misc import strictly_increasing


class AbstractLRSchedule:
    def __init__(self, base_lr=1e-3, *args, **kwargs):
        self.base_lr = base_lr
        self.iter = 0

    def step(self, *args, **kwargs):
        raise NotImplementedError


class FixedLRSchedule(AbstractLRSchedule):
    def __init__(self, base_lr=1e-3):
        super(FixedLRSchedule, self).__init__(base_lr)
        self.lr = base_lr

    def step(self, *args, **kwargs):
        return self.lr


class StairLRSchedule(AbstractLRSchedule):
    def __init__(self, base_lr=1e-3, steps=(), steps_lr=()):
        super(StairLRSchedule, self).__init__(base_lr)
        self.lr = base_lr

        # Check whether the steps are empty
        if len(steps) == 0 or len(steps_lr) == 0:
            print('Step learning rate not specified. The lr schedule will perform as a fixed schedule.')
            self.fix = True
        else:
            self.fix = False

        # Make sure the steps are in an increasing order.
        if not strictly_increasing(steps):
            raise ValueError("The steps should be strictly increasing.")

        self.steps = (0, ) + tuple(steps)
        self.steps_lr = (base_lr, ) + tuple(steps_lr)

        self.cur_step = 0

    def determine_step_index(self, iter):
        if iter >= self.steps[self.cur_step]:
            self.cur_step += 1

    def step(self, iter):
        if not self.fix:
            self.determine_step_index(iter)
        return self.steps_lr[self.cur_step]
