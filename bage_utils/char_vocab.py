import os

import numpy as np
import tensorflow as tf

from bage_utils.datafile_util import DataFileUtil
from bage_utils.list_util import ListUtil


class CharVocab(object):
    def __init__(self, chars: list):
        chars = list(set(chars))
        chars.sort()
        self.__chars = chars
        self.__char2cid = {char: cid for cid, char in enumerate(self.__chars, 0)}
        self.__cid2char = {cid: char for cid, char in enumerate(self.__chars, 0)}
        self.size = len(self.__char2cid)
        self.random_cids_without_blank = np.arange(1, self.size)

    def __repr__(self):
        return '%s(len:%s)' % (self.__class__.__name__, self.__len__())

    def __len__(self):
        return self.size

    @property
    def chars(self):
        """

        :return: list of characters
        """
        return self.__chars

    def char2cid(self, char):
        """

        :param char:
        :return: integer
        """
        return self.__char2cid.get(char, -1)

    def cids2chars(self, cids, pad_size=0) -> '':
        """

        :param cids:
        :param pad_size:
        :return: string
        """
        if pad_size > 0:
            return ''.join([self.__cid2char.get(cid, ' ') for cid in cids]).rstrip(' ') # TODO: TEST
        else:
            return ''.join([self.__cid2char.get(cid, ' ') for cid in cids])

    def chars2cids(self, chars, pad_size=0, pad_value=-1) -> []:
        """

        :param chars: "가나다라마바사"
        :param pad_size:
        :param pad_value:
        :return: 1d array
        """
        cids = [self.__char2cid.get(char, -1) for char in chars]
        if pad_size > 0:
            return cids + [pad_value] * (pad_size - len(cids))
        else:
            return cids

    def sentence2cids(self, sentence: str, window_size: int) -> [[0, ], ]:
        """

        :param sentence: ["가나다라", "마바사자", ...]
        :param window_size:
        :return: 2d array
        """
        cids = []
        cids_in_sentence = self.chars2cids(sentence)
        for start in range(0, len(cids_in_sentence) - window_size + 1):
            cids.append(cids_in_sentence[start:start + window_size])
        return cids

    @staticmethod
    def from_chars(sentences: list):
        chars = set()
        for s in sentences:
            for c in s:
                chars.add(c)
        return CharVocab(chars)

    @staticmethod
    def from_file(characters_file):
        chars = DataFileUtil.read_list(characters_file)
        return CharVocab(chars)

    def save(self, characters_file):
        with open(characters_file, 'wt') as f:
            for c in self.__chars:
                f.write(c)
                f.write('\n')

    def chars2csv(self, chars):
        return ','.join([str(cid) for cid in self.chars2cids(chars)])

    def random_mask(self, word, ner):
        random_cids = np.random.choice(self.random_cids_without_blank, size=len(word), replace=True)
        return word.replace(ner, self.cids2chars(random_cids))

# noinspection PyUnresolvedReferences
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)  # ignore tensorflow info

    # chars = '가나다라'
    # v = CharOneHot(chars)
    # print(v.chars)
    # print(v.char2index)
    # print(v.chars2indices(' 가나다라 마바사.'))
    # exit()
    from nlp4kor.config import KO_WIKIPEDIA_ORG_CHARACTERS_FILE
    window_size = 6
    characters_file = KO_WIKIPEDIA_ORG_CHARACTERS_FILE
    char_dic = CharVocab.from_file(characters_file)

    # with open('/home/bage/workspace/nlp4kor-ko.wikipedia.org/dataset/spelling_error_correction/ko.wikipedia.org.train.sentences_100.window_size_%s.csv' % window_size, 'rt') as f:
    #     for no, line in enumerate(f, 1):
    #         x_cids = [int(cid) for cid in line.split(',')][:window_size]
    #         y_cids = [int(cid) for cid in line.split(',')][window_size:]
    #         print(no, char_dic.cids2chars(x_cids), '->', char_dic.cids2chars(y_cids))

    batch_size = 2
    max_sentence_len = 100
    sentence_list = ['아버지가 방에 들어가셨다.', '가는 말이 고와야 오는 말이 곱다.']
    v = CharVocab.from_chars(sentence_list)
    print(v.size, v.chars)
    # original = v.chars2cids(sentence_list[0])
    # noised = original.copy()
    # noised[0] = -1
    # print(v.cids2chars(original))
    # print(v.cids2chars(noised))

    embedding_size = 10


    def create_graph_one_hot(dic_size, batch_size, window_size):
        x = tf.placeholder(tf.int32, [batch_size, window_size])
        x_vector = tf.one_hot(tf.cast(x, tf.int32), depth=dic_size, dtype=tf.int32)  # FIXME: int32 or float32
        embeddings = None
        return x, embeddings, x_vector


    def create_graph_embedding(dic_size, batch_size, window_size):
        x = tf.placeholder(tf.int32, [batch_size, window_size])
        embeddings = tf.Variable(tf.random_uniform([dic_size, embedding_size], -1, 1))
        x_vector = tf.nn.embedding_lookup(embeddings, x)
        # x_vector = tf.one_hot(tf.cast(x, tf.int32), depth=dic_size, dtype=tf.int32)  # FIXME: int32 or float32
        return x, embeddings, x_vector


    tf.reset_default_graph()
    tf.set_random_seed(7942)
    x, embeddings, x_vector = create_graph_one_hot(dic_size=v.size, batch_size=batch_size, window_size=window_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        x_indices_all = []
        for sentence in sentence_list:
            if len(sentence) > max_sentence_len:
                continue
            x_indices_all.extend(v.sentence2cids(sentence, window_size))

        x_indices_noised = []
        for x_indices in x_indices_all:
            for loc in range(window_size):
                _x = x_indices.copy()
                _x[loc] = -1
                x_indices_noised.append(_x)

        for x_batch in ListUtil.chunks_with_size(x_indices_noised, chunk_size=batch_size):
            x_vector_, = sess.run([x_vector], feed_dict={x: np.array(x_batch)})
            for a, b in zip(x_batch, x_vector_):
                print()
                print(v.cids2chars(a))
                print(a)
                print(b)
                # print(sentence, _x_one_hot_batch.shape)
                # for _x_indices, _x_one_hot in zip(x_indices_batch, _x_one_hot_batch):
                #     print(v.cids2chars(_x_indices), _x_one_hot)
