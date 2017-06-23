import numpy as np

from bage_utils.dataset import DataSet


class DataSets(object):
    def __init__(self, train: DataSet = None, test: DataSet = None, valid: DataSet = None):
        self.train = train if type(train) is DataSet else None
        self.test = test if type(test) is DataSet else None
        self.valid = valid if type(valid) is DataSet else None

    def __repr__(self):
        return '%s, train: %s, test: %s, valid: %s' % (self.__class__, repr(self.train), repr(self.test), repr(self.valid))

    """
        DataSet을 train, test, valid 로 나누어 반환한다.
        :param d:
        :param test_rate:
        :param valid_rate:
        :param valid_max:
        :param test_max:
        :param shuffle:
        :return: Datasets
        """
    @staticmethod
    def to_datasets(d: DataSet, test_rate: float = 0.2, valid_rate: float = 0.2, test_max: int = -1, valid_max: int = -1, shuffle=False):
        # n = len(d)
        # n_train = n - n_test - n_valid
        if test_max == -1 or valid_max == -1:
            test_max = int(d.size * test_rate)
            valid_max = int(d.size * valid_rate)
        else:
            test_max = min(int(d.size * test_rate), test_max)
            valid_max = min(int(d.size * valid_rate), valid_max)

        if shuffle:
            indices = np.random.permutation(len(d.features))
            test_idx, valid_idx, train_idx = indices[:test_max], indices[test_max:test_max + valid_max], indices[test_max + valid_max:]

            test_features, test_labels = d.features[test_idx,], d.labels[test_idx,]
            valid_features, valid_labels = d.features[valid_idx,], d.labels[valid_idx,]
            train_features, train_labels = d.features[train_idx,], d.labels[train_idx,]
        else:
            test_features, test_labels = d.features[:test_max, ], d.labels[:test_max, ]
            valid_features, valid_labels = d.features[test_max:test_max + valid_max, ], d.labels[test_max:test_max + valid_max, ]
            train_features, train_labels = d.features[test_max + valid_max:, ], d.labels[test_max + valid_max:, ]

        test = DataSet(test_features, test_labels, d.features_vector, d.labels_vector, name='test')
        train = DataSet(train_features, train_labels, d.features_vector, d.labels_vector, name='train')
        valid = DataSet(valid_features, valid_labels, d.features_vector, d.labels_vector, name='valid')
        return DataSets(train=train, test=test, valid=valid)


if __name__ == '__main__':
    pass
