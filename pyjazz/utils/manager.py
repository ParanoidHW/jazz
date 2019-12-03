from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import logging
import os
import re
from .serialization import save, load

_log = logging.getLogger(name='Manager')
_log.setLevel(logging.INFO)


class AverageMeter(object):
    __slots__ = ('avg', 'val', 'sum', 'ncount')

    def __init__(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.ncount = 0

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.ncount = 0

    def update(self, new_value, count=1):
        self.sum += new_value * count
        self.val = new_value
        self.ncount += count
        self.avg = self.sum / self.ncount

    def __str__(self):
        return self.avg


class LatestSaveManager(object):
    def __init__(self, max_to_keep=3):
        self.keep_list = []
        self.max_to_keep = max_to_keep

    def save_model(self, f_dict, file_path):
        assert isinstance(f_dict, dict)
        if len(self.keep_list) == self.max_to_keep:
            oldest = self.keep_list.pop(0)
            os.remove(oldest)
        self.keep_list.append(file_path)
        save(f_dict, file_path)

    def search_models(self, root_dir, pattern=r'(?<=epoch_)\d+(?\.pz)'):
        latest_iter = 0
        weights_file = ''
        resume_weights_file = ''
        self.keep_list = os.listdir(root_dir)

        for f in self.keep_list:
            iter_string = re.findall(pattern, f)
            if len(iter_string) > 0:
                checkpoint_iter = int(iter_string[0])
                if checkpoint_iter >= latest_iter:
                    latest_iter = checkpoint_iter
                    resume_weights_file = f

        if len(self.keep_list) == 0 or latest_iter <= 0:
            _log.warning('Unable to find any existing trained models.')
            return None

        if latest_iter > 0:
            weights_file = os.path.join(root_dir, resume_weights_file)
            return weights_file

    def load_model(self, root_dir, pattern=r'(?<=epoch_)\d+(?\.pz)', key='net'):
        weights_file = self.search_models(root_dir, pattern)
        with open(weights_file, 'rb') as f:
            raw = load(f)
            return raw[key]
