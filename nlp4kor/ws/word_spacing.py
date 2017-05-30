import gzip
import os

import numpy as np
import tensorflow as tf

from bage_utils.base_util import is_my_pc
from bage_utils.datafile_util import DataFileUtil
from bage_utils.dataset import DataSet
from bage_utils.datasets import DataSets
from bage_utils.file_util import FileUtil
from bage_utils.num_util import NumUtil
from bage_utils.one_hot_vector import OneHotVector
from nlp4kor.config import log, DATA_DIR_KO_WIKIPEDIA_ORG, SENTENCES_FILE_KO_WIKIPEDIA_ORG


class WordSpacing(object):
    @classmethod
    def create_dataset(cls, sentences_file: str, features_vector: OneHotVector, labels_vector: OneHotVector, max_len: int = 0) -> DataSet:
        features, labels = [], []

        total = FileUtil.count_lines(sentences_file, gzip_format=True if sentences_file.endswith('.gz') else False)
        check_interval = 10000 if not is_my_pc() else 1
        log.info('total: %s' % NumUtil.comma_str(total))

        with gzip.open(sentences_file, 'rt') as f:
            for i, line in enumerate(f, 1):
                if max_len and 0 < max_len < i:  # test_mode
                    break

                if i % check_interval == 0:
                    log.info('%.1f%% readed. dataset len: %s' % (i / total * 100, NumUtil.comma_str(len(features))))

                _features, _labels = cls.sentence2features_labels(line.strip())
                features.extend(_features)
                labels.extend(_labels)

        d = DataSet(features=features, labels=labels, features_vector=features_vector, labels_vector=labels_vector)
        return d

    @classmethod
    def sentence2features_labels(cls, sentence, left_gram=2, right_gram=2) -> (list, list):
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
            features.append(a + b)
            # log.debug('[%d] "%s" "%s" %s' % (i-2, a, b, labels[i-2]))
        return features, labels.tolist()

    @classmethod
    def spacing(cls, sentence, features, labels, left_gram=2, right_gram=2):
        features2label = dict(zip(features, labels))

        sentence = '%s%s%s' % (' ' * (left_gram - 1), sentence.replace(' ', ''), ' ' * (right_gram - 1))
        answers = []
        for i in range(left_gram, len(sentence) - right_gram + 1):
            a, b = sentence[i - left_gram: i], sentence[i: i + right_gram]
            if features2label.get(a + b, 0) == 1:
                answers.append(1)
                # log.debug('"%s" -> "%s"' % (a + b, a + ' ' + b))
            else:
                answers.append(0)
                # log.debug('"%s" -> "%s"' % (a + b, a + b))

        sentence = sentence.strip()
        left = []
        for idx, right in enumerate(sentence[:-1]):
            left.append(right)
            if answers[idx]:
                left.append(' ')
        left.append(sentence[-1])
        return ''.join(left)


if __name__ == '__main__':
    left_gram, right_gram = 2, 2
    ngram = left_gram + right_gram
    sentences_file = SENTENCES_FILE_KO_WIKIPEDIA_ORG

    log.info('learning...')
    train_set = ['예쁜 운동화']
    features, labels = [], []
    for s in train_set:
        features, labels = WordSpacing.sentence2features_labels(s, left_gram=left_gram, right_gram=right_gram)
        log.info('%s %s' % (features, labels))
    log.info('learning OK.\n')

    log.info('testing...')
    test_set = ['예쁜 운동화', '즐거운 동화', '삼풍동 화재']
    for s in test_set:
        # features, labels = WordSpacing.to_features_labels(s, left_gram=left_gram, right_gram=right_gram)
        # print(features, labels)
        log.info('"%s"' % s)
        log.info('"%s"' % WordSpacing.spacing(s.replace(' ', ''), features, labels, left_gram=left_gram, right_gram=right_gram))
    log.info('testing OK.\n')

    characters_file = os.path.join(DATA_DIR_KO_WIKIPEDIA_ORG, 'ko.wikipedia.org.characters')
    log.info('characters_file: %s' % characters_file)
    log.info('load characters list...')
    features_vector = OneHotVector(DataFileUtil.read_list(characters_file))
    labels_vector = OneHotVector([0, 1])  # 붙여쓰기=0, 띄어쓰기=1
    log.info('load characters list OK. len: %s\n' % NumUtil.comma_str(len(features_vector)))

    if is_my_pc():
        max_len = 100  # small
        dataset_file = os.path.join(DATA_DIR_KO_WIKIPEDIA_ORG, 'ko.wikipedia.org.dataset.%d.left=%d.right=%d.gz' % (max_len, left_gram, right_gram))
    else:
        max_len = 0  # full
        dataset_file = os.path.join(DATA_DIR_KO_WIKIPEDIA_ORG, 'ko.wikipedia.org.dataset.gz')

    if not os.path.exists(dataset_file):
        log.info('create dataset...')
        log.info('dataset_file: %s' % dataset_file)
        features, labels = [], []
        total = FileUtil.count_lines(sentences_file, gzip_format=True)
        check_interval = 10000 if not is_my_pc() else 1
        log.info('total: %s' % NumUtil.comma_str(total))

        with gzip.open(sentences_file, 'rt') as f:
            for i, line in enumerate(f, 1):
                if max_len and 0 < max_len < i:  # test_mode
                    break

                if i % check_interval == 0:
                    log.info('%.1f%% readed. data len: %s' % (i / total * 100, NumUtil.comma_str(len(features))))

                _f, _l = WordSpacing.sentence2features_labels(line.strip(), left_gram=left_gram, right_gram=right_gram)
                features.extend(_f)
                labels.extend(_l)

        dataset = DataSet(features=features, labels=labels, features_vector=features_vector, labels_vector=labels_vector)
        log.info('dataset: %s' % dataset)
        log.info('dataset_file: %s' % dataset_file)
        log.info('dataset save...')
        dataset.save(dataset_file)
        log.info('dataset save OK.\n')
        log.info('create dataset OK.\n')

    log.info('dataset load...')
    dataset = DataSet.load(dataset_file, to_one_hot_vector=True)
    log.info(dataset)
    log.info('dataset load OK.\n')

    log.info('check samples...')
    for i, (features_batch, labels_batch) in enumerate(dataset.next_batch(batch_size=10), 1):
        if i > 3:
            break
        for a, b in zip(features_batch, labels_batch):
            if dataset.is_one_hot_vector:
                feature, label = a, b
                _feature = feature.reshape((ngram, len(features_vector)))
                chars = ''.join(features_vector.to_values(_feature))
                has_space = np.argmax(label)
                log.info('[%s] %s -> %s, %s (len=%s) %s (len=%s)' % (i, chars, label, feature, len(feature), label, len(label)))
            else:
                chars, has_space = a, b
                chars_v = features_vector.to_vectors(chars)  # vectors of two characters
                feature = np.concatenate(chars_v)  # concated feature
                label = labels_vector.to_vector(has_space)
                log.info('[%s] %s -> %s, %s (len=%s) %s (len=%s)' % (i, chars, label, feature, len(feature), label, len(label)))
    log.info('check samples OK.\n')

    log.info('split to train, test, validation...')
    datasets = DataSets.to_datasets(dataset, test_rate=0.1, valid_rate=0.1)
    train, test, validation = datasets.train, datasets.test, datasets.validation
    log.info(train)
    log.info(test)
    log.info(validation)
    log.info('%s %s' % (test.features[0], test.labels[0]))
    log.info('%s %s %s' % (type(test.features), test.features.shape, test.features.dtype))
    log.info('%s %s %s' % (type(test.labels), test.labels.shape, test.labels.dtype))
    log.info('split to train, test, validation OK.\n')

    log.info('create tensorflow graph...')
    n_features = len(train.features[0])  # number of features = 17,380 * 2
    n_classes = len(train.labels[0])  # number of classes = 2
    n_hidden1 = 100  # nueron
    log.info('n_features: %s' % n_features)
    log.info('n_classes: %s' % n_classes)
    log.info('n_hidden1: %s' % n_hidden1)

    tf.set_random_seed(777)  # for reproducibility
    learning_rate = 0.1

    X = tf.placeholder(tf.float32, [None, n_features])  # two characters
    Y = tf.placeholder(tf.float32, [None, n_classes])

    W1 = tf.Variable(tf.random_normal([n_features, n_hidden1]), name='weight1')
    b1 = tf.Variable(tf.random_normal([n_hidden1]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_normal([n_hidden1, n_classes]), name='weight2')
    b2 = tf.Variable(tf.random_normal([n_classes]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))  # cost/loss function
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)  # 0 <= hypothesis <= 1
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    log.info('create tensorflow graph OK.\n')

    log.info('learning...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step, (features_batch, labels_batch) in enumerate(train.next_batch(batch_size=10), 1):
            sess.run(train_step, feed_dict={X: features_batch, Y: labels_batch})
            if step % 100 == 0:
                log.info('[%s] cost: %s' % (step, sess.run(cost, feed_dict={X: validation.features, Y: validation.labels})))
                # log.info(sess.run(W1))
                # log.info(sess.run(W2))

        # Accuracy report
        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: test.features, Y: test.labels})
    log.info('learning OK.\n')
    log.info("Hypothesis: %s, Correct: %s, Accuracy: %s", (h, c, a))
