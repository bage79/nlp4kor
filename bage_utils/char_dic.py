import os

import tensorflow as tf

from bage_utils.datafile_util import DataFileUtil


class CharDic(object):
    def __init__(self, chars: list):
        chars = list(set(list(chars) + [' ']))
        chars.sort()
        self.__chars = chars
        self.__char2index = {char: idx for idx, char in enumerate(self.__chars, 0)}
        self.__index2char = {idx: char for idx, char in enumerate(self.__chars, 0)}
        self.dic_size = len(self.__char2index)

    def __repr__(self):
        return '%s(len:%s)' % (self.__class__.__name__, self.__len__())

    def __len__(self):
        return self.dic_size

    def char2index(self, char):
        return self.__char2index.get(char, -1)

    def indices2chars(self, indices):
        return ''.join([self.__index2char.get(index, ' ') for index in indices])

    def chars2indices(self, chars):
        """

        :param chars: "가나다라마바사"
        :return:
        """
        return [self.__char2index.get(char, -1) for char in chars]

    def sentence2indices(self, sentence: str, window_size: int):
        """

        :param sentence: ["가나다라", "마바사자", ...]
        :param window_size:
        :return:
        """
        indices = []
        sentence_indices = self.chars2indices(sentence)
        for start in range(0, len(sentence_indices) - window_size + 1):
            indices.append(sentence_indices[start:start + window_size])
        return indices

    @staticmethod
    def from_chars(sentences: list):
        chars = set()
        for s in sentences:
            for c in s:
                chars.add(c)
        return CharDic(chars)

    @staticmethod
    def from_file(characters_file):
        chars = DataFileUtil.read_list(characters_file)
        return CharDic(chars)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)  # ignore tensorflow info

    # chars = '가나다라'
    # v = CharOneHot(chars)
    # print(v.chars)
    # print(v.char2index)
    # print(v.chars2indices(' 가나다라 마바사.'))
    # exit()

    window_size = 5
    max_sentence_len = 100
    sentence_list = ['아버지가 방에 들어가셨다.', '가는 말이 고와야 오는 말이 곱다.']
    v = CharDic.from_chars(sentence_list)
    print(v.dic_size, v.__char2index)


    def create_graph(window_size):  # TODO: create graph
        x = tf.placeholder(tf.int32, [None, window_size])
        dic_size = tf.placeholder(tf.int32)
        x_one_hot = tf.one_hot(x, depth=dic_size, dtype=tf.int32)  # FIXME: int32 or float32
        return x, dic_size, x_one_hot


    x, dic_size, x_one_hot = create_graph(window_size=window_size)

    with tf.Session() as sess:
        for sentence in sentence_list:
            if len(sentence) > max_sentence_len:
                continue
            print()
            x_indices = v.sentence2indices(sentence, window_size)
            _x_one_hot, = sess.run([x_one_hot], feed_dict={x: x_indices, dic_size: v.dic_size})
            print(sentence, _x_one_hot.shape)
            print(_x_one_hot)
