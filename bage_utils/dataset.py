import gzip
import math
import os
import pickle

import numpy as np

from bage_utils.num_util import NumUtil
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

        if features is None:
            self.features = np.array([])
        else:
            self.features = features if type(features) is np.ndarray else np.array(features)

        if labels is None:
            self.labels = np.array([])
        else:
            self.labels = labels if type(labels) is np.ndarray else np.array(labels)

        if features and labels:
            if len(features) == len(labels):
                self.size = len(features)
            else:  # invalid data size
                self.size = 0
                self.features = np.array([])
                self.labels = np.array([])

    def next_batch(self, batch_size=50, to_one_hot_vector=True, verbose=False):
        if len(self.features) <= batch_size:
            splits = 1
        else:
            splits = len(self.features) // batch_size
            if len(self.features) % batch_size > 0:
                splits += 1

        if splits == 1:
            # log.info('next_batch(batch_size=*splits)= %s * %s = %s' % (NumUtil.comma_str(batch_size), NumUtil.comma_str(splits), NumUtil.comma_str(
            #     len(self.features))))
            if to_one_hot_vector:
                yield self.__to_one_hot_vector(self.features, self.labels, verbose=verbose)
            else:
                yield self.features, self.labels
        else:
            # log.info('next_batch(batch_size=*splits)= %s * %s = %s' % (NumUtil.comma_str(batch_size), NumUtil.comma_str(splits), NumUtil.comma_str(
            #     len(self.features))))

            for features_batch, labels_batch in zip(np.array_split(self.features, splits), np.array_split(self.labels, splits)):
                if to_one_hot_vector:
                    features_batch, labels_batch = self.__to_one_hot_vector(features_batch, labels_batch, verbose=verbose)

                yield features_batch, labels_batch

    def convert_to_one_hot_vector(self, verbose=False):
        self.features, self.labels = self.__to_one_hot_vector(self.features, self.labels, verbose=verbose)
        return self

    def __repr__(self):
        return '%s "%s" (size: %s, feature: %s * %s, label: %s * %s)' % (self.__class__.__name__, self.name, self.size,
                                                                         self.features.dtype, self.features.shape, self.labels.dtype, self.labels.shape)

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.size

    def __to_one_hot_vector(self, features_batch: np.ndarray, labels_batch: np.ndarray, verbose=False):
        _features, _labels = [], []
        check_interval = min(1000, math.ceil(features_batch.shape[0]))
        for i, (feature_string, label_string) in enumerate(zip(features_batch, labels_batch)):
            if isinstance(feature_string, str) or isinstance(feature_string, list):
                feature_v = self.features_vector.to_vectors(feature_string)  # to 2 dim
                feature = np.concatenate(feature_v)  # to 1 dim
            else:
                feature = self.features_vector.to_vector(feature_string)  # to 1 dim

            if isinstance(label_string, str) or isinstance(label_string, list):
                label_v = self.labels_vector.to_vectors(label_string)  # to 2 dim
                label = np.concatenate(label_v)  # to 1 dim
            else:
                label = self.labels_vector.to_vector(label_string)  # to 1 dim

            _features.append(feature)
            _labels.append(label)

            if verbose and i % check_interval == 0:
                log.info('[%s] to_one_hot_vector %s -> %s, %s (len=%s) %s (len=%s)' % (i, feature_string, label, feature, len(feature), label, len(label)))
        return np.asarray(_features, dtype=np.int32), np.asarray(_labels, dtype=np.int32)

    @classmethod
    def load(cls, filepath: str, gzip_format=False, max_len=0, verbose=False):
        filename = os.path.basename(filepath)
        if gzip_format:
            f = gzip.open(filepath, 'rb')
        else:
            f = open(filepath, 'rb')

        with f:
            d = DataSet()
            d.name, d.size, d.features_vector, d.labels_vector = pickle.load(f), pickle.load(f), pickle.load(f), pickle.load(f)

            check_interval = min(100000, math.ceil(d.size))
            features, labels = [], []
            for i in range(d.size):
                if 0 < max_len <= len(features):
                    break
                feature, label = pickle.load(f), pickle.load(f)
                # print('load feature:', feature, 'label:', label)
                features.append(feature)
                labels.append(label)
                if verbose and i % check_interval == 0:
                    log.info('%s %.1f%% loaded.' % (filename, i / d.size * 100))
            log.info('%s 100%% loaded.' % filename)
            d.features = np.asarray(features)
            d.labels = np.asarray(labels)
            log.info('%s features shape: %s' % (filename, d.features.shape))
            log.info('%s labels shape: %s' % (filename, d.features.shape))
        return d

    def save(self, filepath: str, gzip_format=False, verbose=False):
        filename = os.path.basename(filepath)
        if gzip_format:
            f = gzip.open(filepath, 'wb')
        else:
            f = open(filepath, 'wb')

        with f:
            for o in [self.name, self.size, self.features_vector, self.labels_vector]:
                pickle.dump(o, f)

            check_interval = min(100000, math.ceil(self.size))
            for i, (feature, label) in enumerate(zip(self.features, self.labels)):
                # print('save feature:', feature, 'label:', label)
                pickle.dump(feature, f)
                pickle.dump(label, f)
                if verbose and i % check_interval == 0:
                    log.info('%s %.1f%% saved.' % (filename, i / self.size * 100))

            log.info('%s 100%% saved.' % filename)
            log.info('shape: %s' % self.features.shape)


if __name__ == '__main__':
    pass
