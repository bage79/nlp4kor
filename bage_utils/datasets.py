import numpy as np

from bage_utils.dataset import DataSet


class DataSets(object):
    def __init__(self, train: DataSet, test: DataSet, validation: DataSet = None):
        self.train = train if type(train) is DataSet else DataSet(train)
        self.test = test if type(test) is DataSet else DataSet(test)
        if validation:
            self.validation = validation if type(validation) is DataSet else DataSet(validation, train.features_vector, train.labels_vector)
        else:
            self.validation = None

    def __repr__(self):
        return '%s, train: %s, test: %s, valid: %s' % (self.__class__, repr(self.train), repr(self.test), repr(self.validation))

    @staticmethod
    def to_datasets(d: DataSet, test_rate: float = 0.2, valid_rate: float = 0.2, shuffle=False):
        """
        DataSet을 train, test, validation 로 나누어 반환한다.
        :param d: 
        :param test_rate: 
        :param valid_rate: 
        :param shuffle: 
        :return: test(Dataset), train(DataSet) 
        """
        n = len(d)
        n_test = int(d.size * test_rate)
        n_valid = int(d.size * valid_rate)
        # n_train = n - n_test - n_valid

        if shuffle:  # TODO: need to test
            indices = np.random.permutation(len(d.features))
            train_idx, test_idx, valid_idx = indices[:n_test], indices[n_test:n_test + n_valid], indices[n_test + n_valid:]
            test_features, test_labels = d.features[test_idx,], d.labels[test_idx,]
            valid_features, valid_labels = d.features[valid_idx,], d.labels[valid_idx,]
            train_features, train_labels = d.features[test_idx,], d.labels[test_idx,]
        else:
            test_features, test_labels = d.features[:n_test, ], d.labels[:n_test, ]
            valid_features, valid_labels = d.features[n_test:n_test + n_valid, ], d.labels[n_test:n_test + n_valid, ]
            train_features, train_labels = d.features[n_test + n_valid:, ], d.labels[n_test + n_valid:, ]

        test = DataSet(test_features, test_labels, d.features_vector, d.labels_vector)
        train = DataSet(train_features, train_labels, d.features_vector, d.labels_vector)
        valid = DataSet(valid_features, valid_labels, d.features_vector, d.labels_vector)
        return DataSets(train=train, test=test, validation=valid)


if __name__ == '__main__':
    pass
