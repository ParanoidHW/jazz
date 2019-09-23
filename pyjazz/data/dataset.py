from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import os
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


# ----------------------------------------
# The following are real or fake datasets.
# ----------------------------------------

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


class MNISTDataset(Dataset):
    def __init__(self, root_dir='.', split='train'):
        super(MNISTDataset, self).__init__()
        if split not in ['train', 'test']:
            raise ValueError('Unsupported dataset split. Expect `train` or `test`, but got `{}`'.format(split))
        self.prefix = 'train' if split == 'train' else 't10k'
        self.images, self.labels = self._download_or_just_load(root_dir)
        self.num_cases = self.images.shape[0]

    def _download_or_just_load(self, target_dir):
        import gzip

        def _load_image_file(filename, training=True):
            f = gzip.open(filename, 'rb')
            content = f.read()
            # Skip the first 16 bytes
            content = content[16:]

            num_cases = 60000 if training else 10000
            images = np.frombuffer(content, dtype=np.uint8)
            images = np.reshape(images, (-1, 28, 28))
            return images

        def _load_label_file(filename, training=True):
            f = gzip.open(filename, 'rb')
            content = f.read()
            # Skip the first 16 bytes
            content = content[8:]

            num_cases = 60000 if training else 10000
            labels = np.frombuffer(content, dtype=np.uint8)
            return np.array(labels)

        image_file = '{}-images-idx3-ubyte.gz'.format(self.prefix)
        label_file = '{}-labels-idx1-ubyte.gz'.format(self.prefix)
        if not os.path.exists(os.path.join(target_dir, image_file)):
            os_cmd = "wget http://yann.lecun.com/exdb/mnist/{}".format(image_file)
            os.system(os_cmd)
        images = _load_image_file(os.path.join(target_dir, image_file))
        images = images.astype(np.float32)
        images = images / 255.
        images_mean = np.mean(images)
        images_std = np.std(images)
        images = (images - images_mean) / (images_std + 1e-8)

        if not os.path.exists(os.path.join(target_dir, label_file)):
            os_cmd = "wget http://yann.lecun.com/exdb/mnist/{}".format(label_file)
            os.system(os_cmd)
        labels = _load_label_file(os.path.join(target_dir, label_file))
        return images, labels

    def __len__(self):
        return self.num_cases

    def __getitem__(self, item):
        return self.images[item], self.labels[item]
