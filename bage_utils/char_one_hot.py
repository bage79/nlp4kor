import tensorflow as tf


class CharOneHot(object):
    def __init__(self, chars: list):
        chars = list(set(list(chars) + [' ']))
        chars.sort()
        self.chars = chars
        self.char2index = {char: idx for idx, char in enumerate(self.chars, -1)}
        self.dic_size = len(self.char2index)

    def one_hot(self, text):
        idxs = [self.char2index.get(char, -1) for char in text]
        return tf.one_hot(idxs, depth=self.dic_size)


if __name__ == '__main__':
    text = '가나다라'
    v = CharOneHot(text)
    print(v.chars)
    print(v.char2index)

    sentence_list = [' 가나다라하A']
    sequence_length = 5
    for sentence in sentence_list:
        x = tf.placeholder(tf.int32, [None, sequence_length])
        x_one_hot = v.one_hot(sentence)
        with tf.Session() as sess:
            _x_one_hot, = sess.run([x_one_hot])
            print(_x_one_hot.shape)

            # for char, vector in zip(sentence[:sequence_length], _x_one_hot): # TODO:

