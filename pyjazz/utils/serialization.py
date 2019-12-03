from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import pickle as pkl


def save(obj, f):
    with open(f, 'wb') as fid:
        pkl.dump(obj, fid, protocol=2)


def load(f):
    with open(f, 'rb') as fid:
        obj = pkl.load(fid)
    return obj

