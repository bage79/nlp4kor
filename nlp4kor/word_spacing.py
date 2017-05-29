import gzip
import pickle

import numpy as np

from bage_utils.base_util import is_my_pc
from bage_utils.dataset import DataSet
from bage_utils.file_util import FileUtil
from bage_utils.num_util import NumUtil
from bage_utils.one_hot_vector import OneHotVector
from nlp4kor.config import log


class WordSpacing(object):
    @classmethod
    def chars2vector(cls, features, labels, features_vector, labels_vector):
        _features, _labels = [], []
        for chars, has_space in zip(features, labels):
            chars = list([c for c in chars])  # two characters.
            chars_v = features_vector.to_vectors(chars)
            chars_v = list([v for v in chars_v])
            feature = np.concatenate(chars_v)  # concated feature
            label = labels_vector.to_vector(has_space)

            # log.info('%s, %s = %s (len=%s) -> %s (len=%s)' % (chars[0], chars[1], feature, len(feature), label, len(label)))
            _features.append(feature)
            _labels.append(label)
        return _features, _labels

    # @classmethod
    # def to_features_labels(cls, sentence):
    #     sentence = sentence.strip()
    #     if len(sentence) <= 2:
    #         return [], []
    #
    #     sentence = ' ' + sentence
    #     sentence2 = sentence[1:]
    #     sentence3 = sentence[2:]
    #     sentence = sentence[:-2]
    #     # log.debug(sentence)
    #     # log.debug(sentence2)
    #     # log.debug(sentence3)
    #
    #     features = []  # [a, b]
    #     labels = []  # int
    #     for a, b, c in zip(sentence, sentence2, sentence3):
    #         if c == ' ':
    #             continue
    #
    #         if b == ' ':
    #             features.append((a + c))
    #             labels.append(1)
    #             # print('[%s, %s] -> 1' % (a, c))
    #         else:
    #             features.append((b + c))
    #             labels.append(0)
    #             # print('[%s, %s] -> 0' % (b, c))
    #
    #     return features, labels

    @classmethod
    def to_features_labels(cls, sentence, left_gram=2, right_gram=2):
        if left_gram < 1 or right_gram < 1:
            return [], []

        sentence = sentence.strip()
        if len(sentence) <= 1:
            return [], []

        labels = np.zeros(len(sentence.replace(' ', '')) - 1, dtype=np.int32)
        sentence = '%s%s%s' % (' ' * (left_gram - 1), sentence, ' ' * (right_gram - 1))
        idx = 0
        for mid in range(left_gram, len(sentence) - right_gram + 1):
            # log.debug('"%s" "%s" "%s"' % (sentence[mid - left_gram:mid], sentence[mid], sentence[mid + 1:mid + 1 + right_gram]))
            if sentence[mid] == ' ':
                labels[idx] = 1
            else:
                idx += 1

        no_space_sentence = '%s%s%s' % (' ' * (left_gram - 1), sentence.replace(' ', ''), ' ' * (right_gram - 1))
        features = []
        for i in range(left_gram, len(no_space_sentence) - right_gram + 1):
            a, b = no_space_sentence[i - left_gram: i], no_space_sentence[i: i + right_gram]
            features.append((a, b))
            # log.debug('[%d] "%s" "%s" %s' % (i-2, a, b, labels[i-2]))
        return features, labels.tolist()

    @classmethod
    def create_dataset(cls, sentences_file: str, dataset_file: str, features_vector: OneHotVector, labels_vector: OneHotVector, gzip_format=False,
                       max_len: int = 0) -> DataSet:
        features, labels = [], []

        total = FileUtil.count_lines(sentences_file, gzip_format=True)
        check_interval = 10000 if not is_my_pc() else 1
        log.info('total: %s' % NumUtil.comma_str(total))

        with gzip.open(sentences_file, 'rt') as f:
            for i, line in enumerate(f, 1):
                if max_len and 0 < max_len < i:  # test_mode
                    break

                if i % check_interval == 0:
                    log.info('%.1f%% readed. dataset len: %s' % (i / total * 100, NumUtil.comma_str(len(features))))

                _features, _labels = cls.to_features_labels(line.strip())
                features.extend(_features)
                labels.extend(_labels)

        d = DataSet(features=features, labels=labels, features_vector=features_vector, labels_vector=labels_vector)
        if gzip_format:
            with gzip.open(dataset_file, 'w') as f_out:
                pickle.dump(d, f_out)
        else:
            with open(dataset_file, 'w') as f_out:
                pickle.dump(d, f_out)
        return d

    @classmethod
    def load_dataset(cls, dataset_file: str, gzip_format=False) -> DataSet:
        if gzip_format:
            with gzip.open(dataset_file, 'rb') as f:
                return pickle.load(f)
        else:
            with open(dataset_file, 'rb') as f:
                return pickle.load(f)

    @classmethod
    def spacing(cls, no_space_sentence, features, labels):
        pass

if __name__ == '__main__':
    print(WordSpacing.to_features_labels('나는 밥을 먹었다.', left_gram=1, right_gram=1))
    print(WordSpacing.to_features_labels('나는 밥을 먹었다.', left_gram=2, right_gram=2))
    print(WordSpacing.to_features_labels('나는 밥을 먹었다.', left_gram=3, right_gram=3))
    # print(WordSpacing.to_features_labels('사고'))
    # print(WordSpacing.to_features_labels('사고', left_gram=1, right_gram=1))
    # characters_file = os.path.join(DATA_DIR_KO_WIKIPEDIA_ORG, 'ko.wikipedia.org.characters')
    # log.info('characters_file: %s' % characters_file)
    # log.info('load characters list...')
    # features_vector = OneHotVector(DataFileUtil.read_list(characters_file))
    # labels_vector = OneHotVector([0, 1])  # 붙여쓰기=0, 띄어쓰기=1
    # log.info('load characters list OK. len: %s' % NumUtil.comma_str(len(features_vector)))
    #
    # if is_my_pc():
    #     max_len = 10000  # small
    #     dataset_file = os.path.join(DATA_DIR_KO_WIKIPEDIA_ORG, 'ko.wikipedia.org.dataset.10k.gz')
    # else:
    #     max_len = 0  # full
    #     dataset_file = os.path.join(DATA_DIR_KO_WIKIPEDIA_ORG, 'ko.wikipedia.org.dataset.gz')
    #
    # log.info('dataset_file: %s' % dataset_file)
    # if not os.path.exists(dataset_file):
    #     log.info('create dataset...')
    #     WordSpacing.create_dataset(sentences_file, dataset_file, features_vector, labels_vector, gzip_format=True,
    #                                   max_len=max_len)  # corpus -> dataset(file)
    #     log.debug('create dataset OK.')
    #
    # log.info('dataset_file: %s' % dataset_file)
    # log.info('load dataset...')
    # _dataset = TextPreprocess.load_dataset(dataset_file, gzip_format=True)  # dataset(file) -> dataset(memory)
    # datasets = DataSets.to_datasets(_dataset, test_rate=0.1, valid_rate=0.1)
    # train, test, validation = datasets.train, datasets.test, datasets.validation
    # log.info(train)
    # log.info(test)
    # log.info(validation)
    # log.info('load dataset OK.')
    #
    # # log.info('check samples...')
    # # for i, (features_batch, labels_batch) in enumerate(datasets.test.next_batch(batch_size=10), 1):
    # #     if i > 2:
    # #         break
    # #     for chars, has_space in zip(features_batch, labels_batch):
    # #         chars = list([c for c in chars])  # two characters.
    # #         chars_v = features_vector.to_vectors(chars)  # vectors of two characters
    # #         chars_v = list([v for v in chars_v])  # vectors of two characters
    # #
    # #         feature = np.concatenate(chars_v)  # concated feature
    # #         label = labels_vector.to_vector(has_space)
    # #         log.info('[%s] %s, %s = %s (len=%s) -> %s (len=%s)' % (i, chars[0], chars[1], feature, len(feature), label, len(label)))
    # # log.info('check samples OK.')
    #
    # log.info('characters to vector...')  # TODO: convert to vector on loading time?
    # train_features, train_labels = chars2vector(features=train.features, labels=train.labels)  # character -> vector
    # valid_features, valid_labels = chars2vector(features=validation.features, labels=validation.labels)  # character -> vector
    # test_features, test_labels = chars2vector(features=test.features, labels=test.labels)  # character -> vector
    # log.info('characters to vector OK.')
    #
    # log.info('create tensorflow graph...')
    # n_features = len(train_features[0])  # number of features = 17,380 * 2
    # n_classes = len(train_labels[0])  # number of classes = 2
    # n_hidden1 = 100  # nueron
    # log.info('n_features: %s' % n_features)
    # log.info('n_classes: %s' % n_classes)
    # log.info('n_hidden1: %s' % n_hidden1)
    #
    # tf.set_random_seed(777)  # for reproducibility
    # learning_rate = 0.1
    #
    # X = tf.placeholder(tf.float32, [None, n_features])  # two characters
    # Y = tf.placeholder(tf.float32, [None, n_classes])
    #
    # W1 = tf.Variable(tf.random_normal([n_features, n_hidden1]), name='weight1')
    # b1 = tf.Variable(tf.random_normal([n_hidden1]), name='bias1')
    # layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    #
    # W2 = tf.Variable(tf.random_normal([n_hidden1, n_classes]), name='weight2')
    # b2 = tf.Variable(tf.random_normal([n_classes]), name='bias2')
    # hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    #
    # cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))  # cost/loss function
    # train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    #
    # predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)  # 0 <= hypothesis <= 1
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    # log.info('create tensorflow graph OK.')
    #
    # log.info('learning...')
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     for step, (features_batch, labels_batch) in enumerate(train.next_batch(batch_size=10), 1):
    #         features_batch, labels_batch = chars2vector(features=features_batch, labels=labels_batch)  # character -> vector
    #
    #         sess.run(train, feed_dict={X: features_batch, Y: labels_batch})
    #         # if step % 100 == 0:
    #         log.debug('feature, label: %s -> %s' % (features_batch[0], labels_batch[0]))  # test
    #         log.info('[%s] cost: %s' % (step, sess.run(cost, feed_dict={X: valid_features, Y: valid_labels})))
    #         log.info(sess.run(W1))
    #         log.info(sess.run(W2))
    #
    #     # Accuracy report
    #     h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: test_features, Y: test_labels})
    # log.info('learning OK.')
    # log.info("Hypothesis: %s, Correct: %s, Accuracy: %s", (h, c, a))
