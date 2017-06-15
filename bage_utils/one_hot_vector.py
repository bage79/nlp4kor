import numpy as np
from sklearn.preprocessing import LabelBinarizer


class OneHotVector(object):
    def __init__(self, chars: list):
        if not chars or type(chars) is not list or len(chars) == 0:
            raise Exception('values must be list and len(values)>0 %s' % chars)

        self.encoder = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)  # TODO: performance test
        self.encoder.fit(chars)

    @property
    def classes(self):
        return self.encoder.classes_

    def __len__(self):
        return self.encoder.classes_.shape[0]

    def __repr__(self):
        return '%s(len:%s)' % (self.__class__.__name__, self.__len__())

    def to_vector(self, c: str) -> np.ndarray:
        """
        
        :param c: character. len(c)==1
        :return:
        """
        return self.encoder.transform([c])[0]

    def to_vectors(self, chars: list) -> np.ndarray:
        """
        
        :param chars: list of characters. len(chars)>0
        :return:
        """
        if type(chars) is str or np.str_:
            chars = [c for c in chars]
        return self.encoder.transform(chars)

    def to_value(self, v: np.ndarray) -> np.ndarray:
        """
        
        :param v: one hot vector 
        :return: 
        """
        return self.encoder.inverse_transform(np.array([v]))[0]

    def to_values(self, vectors: list) -> np.ndarray:
        """

        :param vectors: list of one hot vector 
        :return: 
        """
        return self.encoder.inverse_transform(vectors)

    def to_index(self, c: str) -> int:
        return np.argmax(self.to_vector(c))


if __name__ == '__main__':
    unary_vector = OneHotVector([0])
    binary_vector = OneHotVector([0, 1])
    ternary_vector = OneHotVector([0, 1, 2])
    print(unary_vector, binary_vector, ternary_vector)
    print('%6s\t%6s\t%6s\t%6s' % ('', 'unary', 'binary', 'ternary'))
    for i in [0, 1, 2]:
        print('%6s\t%6s\t%6s\t%6s' % (i, unary_vector.to_vector(i), binary_vector.to_vector(i), ternary_vector.to_vector(i)))

        # @formatter:off
    # chars = ['0', '1', '2']
    # chars = [1, 0]
    # chars = ['ㅎ', 'ㄱ', 'a', 'b']
    # ohv = OneHotVector(chars)
    # for c in chars:
    #     v = ohv.to_vector(c)
    #     print(c, type(v), v, ohv.to_value(v))
    #
    # vectors = ohv.to_vectors(chars)
    # print(type(vectors), vectors)
    # values = ohv.to_values(vectors)
    # print(type(values), values)
