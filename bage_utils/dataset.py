import gzip
import pickle

import numpy as np

from bage_utils.list_util import ListUtil
from bage_utils.one_hot_vector import OneHotVector


class DataSet(object):
    def __init__(self, features: np.ndarray, labels: np.ndarray, features_vector: OneHotVector, labels_vector: OneHotVector, name: str = ''):
        """
        
        :param features: list of data
        :param labels: list of one hot vector 
        """
        self.name = name
        self.features = features if type(features) is np.ndarray else np.array(features)
        self.labels = labels if type(labels) is np.ndarray else np.array(labels)
        self.size = min(len(self.features), len(self.labels))
        self.features_vector = features_vector
        self.labels_vector = labels_vector

        if len(self.labels) > self.size:
            self.labels = self.labels[:self.size]
        if len(self.features) > self.size:
            self.features = self.features[:self.size]

    def next_batch(self, batch_size=50):
        splits = len(self.features) // batch_size
        if len(self.features) % batch_size > 0:
            splits += 1
        for features_batch, labels_batch in zip(np.array_split(self.features, splits),
                                                ListUtil.chunks_with_size(self.labels, chunk_size=batch_size)):
            yield features_batch, labels_batch

    def __repr__(self):
        return '%s "%s" (feature: %s, label:%s, size: %s)' % (self.__class__.__name__, self.name, type(self.features[0]), type(self.labels[0]), self.size)

    def __len__(self):
        return self.size

    @classmethod
    def load(cls, filepath: str, gzip_format=False):
        if gzip_format:
            with gzip.open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)

    def save(self, filepath: str, gzip_format=False):
        if gzip_format:
            with gzip.open(filepath, 'wb') as f_out:
                pickle.dump(self, f_out)
        else:
            with open(filepath, 'wb') as f_out:
                pickle.dump(self, f_out)


if __name__ == '__main__':
    pass
