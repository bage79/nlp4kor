import gzip
import pickle

import math
import numpy as np

from bage_utils.one_hot_vector import OneHotVector
from nlp4kor.config import log


class DataSet(object):
    def __init__(self, features: np.ndarray = None, labels: np.ndarray = None, features_vector: OneHotVector = None,
                 labels_vector: OneHotVector = None, size=0, name: str = ''):
        """
        
        :param features: list of data
        :param labels: list of one hot vector 
        """
        self.name = name
        self.size = size
        self.features_vector = features_vector
        self.labels_vector = labels_vector

        if features is not None:
            self.features = features if type(features) is np.ndarray else np.array(features)
        if labels is not None:
            self.labels = labels if type(labels) is np.ndarray else np.array(labels)
        if features is not None and labels is not None:
            self.size = min(len(self.features), len(self.labels))
            if len(self.labels) > self.size:
                self.labels = self.labels[:self.size]
            if len(self.features) > self.size:
                self.features = self.features[:self.size]

    def next_batch(self, batch_size=50, to_one_hot_vector=True, verbose=False):
        splits = len(self.features) // batch_size
        if len(self.features) % batch_size > 0:
            splits += 1
        for features_batch, labels_batch in zip(np.array_split(self.features, splits), np.array_split(self.labels, splits)):
            if to_one_hot_vector:
                features_batch, labels_batch = self.to_one_hot_vector(features_batch, labels_batch, verbose=verbose)

            yield features_batch, labels_batch

    def convert_to_one_hot_vector(self, verbose=False):
        self.features, self.labels = self.to_one_hot_vector(self.features, self.labels, verbose=verbose)
        return self

    def __repr__(self):
        return '%s "%s" (feature: %s, label:%s, size: %s)' % (self.__class__.__name__, self.name, type(self.features[0]), type(self.labels[0]), self.size)

    def __len__(self):
        return self.size

    def to_one_hot_vector(self, features_batch: np.ndarray, labels_batch: np.ndarray, verbose=False):
        _features, _labels = [], []
        for i, (chars, has_space) in enumerate(zip(features_batch, labels_batch)):
            chars_v = self.features_vector.to_vectors(chars)
            feature = np.concatenate(chars_v)  # concated feature

            label = self.labels_vector.to_vector(has_space)
            _features.append(feature)
            _labels.append(label)
            if verbose and i % 1000 == 0:
                log.info('[%s] to_one_hot_vector %s -> %s, %s (len=%s) %s (len=%s)' % (i, chars, label, feature, len(feature), label, len(label)))
        return np.asarray(_features, dtype=np.int32), np.asarray(_labels, dtype=np.int32)

    @classmethod
    def load(cls, filepath: str, gzip_format=False, verbose=False):
        if gzip_format:
            f = gzip.open(filepath, 'rb')
        else:
            f = open(filepath, 'rb')

        with f:
            d = DataSet()
            d.name, d.size, d.features_vector, d.labels_vector, d.labels = \
                pickle.load(f), pickle.load(f), pickle.load(f), pickle.load(f), pickle.load(f)

            check_interval = 1000  # min(1000, math.ceil(d.size / 100))

            li = []
            for i in range(d.size):
                li.append(pickle.load(f))
                if verbose and i % check_interval == 0:
                    log.info('%.1f%% loaded.' % (i / d.size * 100))
            log.info('100% loaded.')
            d.features = np.asarray(li)
        return d

    def save(self, filepath: str, gzip_format=False, verbose=False):
        if gzip_format:
            f = gzip.open(filepath, 'wb')
        else:
            f = open(filepath, 'wb')

        with f:
            for o in [self.name, self.size, self.features_vector, self.labels_vector, self.labels]:
                pickle.dump(o, f)

            check_interval = min(10000, math.ceil(self.size))
            for i, o in enumerate(self.features):
                pickle.dump(o, f)
                if verbose and i % check_interval == 0:
                    log.info('%.1f%% saved.' % (i / self.size * 100))
            log.info('100% saved.')


if __name__ == '__main__':
    pass
