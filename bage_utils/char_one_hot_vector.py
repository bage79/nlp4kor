import numpy as np
from sklearn.preprocessing import LabelBinarizer


class CharOneHotVector(object):
    def __init__(self, chars: list, added: list = [], unkown_char=' '):
        chars.extend(added)
        if not chars or type(chars) is not list or len(chars) == 0:
            raise Exception('values must be list and len(values)>0 %s' % chars)

        self.unkown_char = unkown_char
        self.chars = chars
        self.encoder = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
        self.encoder.fit(chars)

    @property
    def classes(self):
        return self.encoder.classes_

    def __len__(self):
        return self.encoder.classes_.shape[0]

    @property
    def size(self):
        return self.encoder.classes_.shape[0]

    def __repr__(self):
        return '%s(len:%s)' % (self.__class__.__name__, self.__len__())

    def to_vector(self, char: str) -> np.ndarray:
        """
        
        :param char: character. len(c)==1
        :return:
        """
        return self.encoder.transform([char])[0]

    def to_vectors(self, chars: list) -> np.ndarray:
        """
        
        :param chars: list of characters. len(chars)>0
        :return:
        """
        if type(chars) is str or type(chars) is np.str_:
            chars = [c for c in chars]
        return self.encoder.transform(chars)

    def to_value(self, vector: np.ndarray) -> np.ndarray:
        """
        
        :param vector: one hot vector
        :return: 
        """
        if vector.ndim != 1:
            vector = vector.flatten()

        if not vector.any():
            return self.unkown_char
        else:
            return self.encoder.inverse_transform(np.array([vector]))[0]

    def to_values(self, vectors: np.ndarray) -> np.ndarray:
        """

        :param vectors: list of one hot vector 
        :return: 
        """
        if vectors.ndim != 2:
            vectors = vectors.reshape((len(vectors) // self.size, self.size))

        chars = []
        for vector in vectors:
            chars.append(self.to_value(vector))
        return ''.join(chars)
        # return ''.join(self.encoder.inverse_transform(vectors))

    def to_index(self, c: str) -> int:
        return np.argmax(self.to_vector(c))

    def index2value(self, index):
        if 0 < index < len(self.chars):
            return self.classes[index]
        else:
            return ''


if __name__ == '__main__':
    # chars = ['0', '1', '2']
    # chars = [1, 0]
    chars = ['ㄷ', 'ㄱ', 'ㄴ', 'ㄹ']
    ohv = CharOneHotVector(chars)
    _input = 'ㄱㄴㄷㄹㅎ'
    feature_v = ohv.to_vectors(_input)
    print(_input)
    print(feature_v)
    # print(ohv.to_value(feature_v[-1]))
    print(ohv.to_values(feature_v))
    print(ohv.to_values(np.array([[0.1, 1., 0.5, 0.5]])))

    # unary_vector = OneHotVector([0])
    # binary_vector = OneHotVector([0, 1])
    # ternary_vector = OneHotVector([0, 1, 2])
    # print(unary_vector, binary_vector, ternary_vector)
    # print('%6s\t%6s\t%6s\t%6s' % ('', 'unary', 'binary', 'ternary'))
    # for i in [0, 1, 2]:
    #     print('%6s\t%6s\t%6s\t%6s' % (i, unary_vector.to_vector(i), binary_vector.to_vector(i), ternary_vector.to_vector(i)))

    # @formatter:off
    # for c in chars:
    #     v = ohv.to_vector(c)
    #     print(c, type(v), v, ohv.to_value(v))
    #
    # vectors = ohv.to_vectors(chars)
    # print(type(vectors), vectors)
    # values = ohv.to_values(vectors)
    # print(type(values), values)
