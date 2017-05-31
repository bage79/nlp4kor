import gzip
import os

import numpy as np
import tensorflow as tf
import math

from bage_utils.base_util import is_my_pc
from bage_utils.datafile_util import DataFileUtil
from bage_utils.dataset import DataSet
from bage_utils.datasets import DataSets
from bage_utils.file_util import FileUtil
from bage_utils.num_util import NumUtil
from bage_utils.one_hot_vector import OneHotVector
from nlp4kor.config import log, KO_WIKIPEDIA_ORG_DATA_DIR, KO_WIKIPEDIA_ORG_SENTENCES_FILE


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


def learning(sentences_file, batch_size, left_gram, right_gram, model_file, n_hidden1=100, max_sentences=0, learning_rate=0.1):
    ngram = left_gram + right_gram
    characters_file = os.path.join(KO_WIKIPEDIA_ORG_DATA_DIR, 'ko.wikipedia.org.characters')
    log.info('characters_file: %s' % characters_file)
    log.info('load characters list...')
    features_vector = OneHotVector(DataFileUtil.read_list(characters_file))
    labels_vector = OneHotVector([0, 1])  # 붙여쓰기=0, 띄어쓰기=1
    log.info('load characters list OK. len: %s\n' % NumUtil.comma_str(len(features_vector)))

    # dataset_file = os.path.join(KO_WIKIPEDIA_ORG_DATA_DIR, 'ko.wikipedia.org.dataset.%d.left=%d.right=%d.gz' % (max_sentences, left_gram, right_gram))
    # log.info('dataset_file: %s' % dataset_file)

    train_file = os.path.join(KO_WIKIPEDIA_ORG_DATA_DIR, 'ko.wikipedia.org.dataset.%d.train.left=%d.right=%d.gz' % (max_sentences, left_gram, right_gram))
    validation_file = os.path.join(KO_WIKIPEDIA_ORG_DATA_DIR,
                                   'ko.wikipedia.org.dataset.%d.validation.left=%d.right=%d.gz' % (max_sentences, left_gram, right_gram))
    test_file = os.path.join(KO_WIKIPEDIA_ORG_DATA_DIR, 'ko.wikipedia.org.dataset.%d.test.left=%d.right=%d.gz' % (max_sentences, left_gram, right_gram))
    if not os.path.exists(train_file) or not os.path.exists(validation_file) or not os.path.exists(test_file):
        log.info('create dataset...')
        features, labels = [], []
        total = FileUtil.count_lines(sentences_file, gzip_format=True)
        check_interval = min(10000, math.ceil(total))
        log.info('total: %s' % NumUtil.comma_str(total))

        with gzip.open(sentences_file, 'rt') as f:
            for i, line in enumerate(f, 1):
                if max_sentences and 0 < max_sentences < i:  # test_mode
                    break

                if i % check_interval == 0:
                    log.info('%.1f%% readed. data len: %s' % (i / total * 100, NumUtil.comma_str(len(features))))

                _f, _l = WordSpacing.sentence2features_labels(line.strip(), left_gram=left_gram, right_gram=right_gram)
                features.extend(_f)
                labels.extend(_l)

        dataset = DataSet(features=features, labels=labels, features_vector=features_vector, labels_vector=labels_vector)
        log.info('dataset: %s' % dataset)

        log.info('split to train, test, validation...')
        datasets = DataSets.to_datasets(dataset, test_rate=0.1, valid_rate=0.1, shuffle=True)
        train, test, validation = datasets.train, datasets.test, datasets.validation
        log.info(train)
        log.info(test)
        log.info(validation)
        # log.info('%s %s' % (test.features[0], test.labels[0]))
        log.info('split to train, test, validation OK.\n')

        log.info('dataset save...%s' % train_file)
        train.save(train_file, verbose=True)  # counter 형식으로 저장할 까?
        log.info('dataset save OK.\n')
        log.info('create dataset OK.\n')

        log.info('dataset save...%s' % validation_file)
        validation = validation.convert_to_one_hot_vector(verbose=True)
        validation.save(validation_file, verbose=True)
        log.info('dataset save OK.\n')

        log.info('dataset save...%s' % test_file)
        test = test.convert_to_one_hot_vector(verbose=True)
        test.save(test_file, verbose=True)
        log.info('dataset save OK.\n')

    log.info('dataset load...')
    train = DataSet.load(train_file, verbose=True)
    validation = DataSet.load(validation_file, verbose=True)
    test = DataSet.load(test_file, verbose=True)
    log.info(train)
    log.info(validation)
    log.info(test)
    log.info('dataset load OK.\n')

    log.info('check samples...')
    for i, (features_batch, labels_batch) in enumerate(train.next_batch(batch_size=5, to_one_hot_vector=True), 1):
        if i > 2:
            break
        for a, b in zip(features_batch, labels_batch):
            feature, label = a, b
            _feature = feature.reshape((ngram, len(features_vector)))
            chars = ''.join(features_vector.to_values(_feature))
            has_space = np.argmax(label)
            log.info('[%s] %s -> %s, %s (len=%s) %s (len=%s)' % (i, chars, has_space, feature, len(feature), label, len(label)))
    log.info('check samples OK.\n')

    log.info('create tensorflow graph...')
    n_features = len(features_vector.classes) * ngram  # number of features = 17,380 * 4
    n_classes = 1  # number of classes = 2 but len=1
    log.info('n_features: %s' % n_features)
    log.info('n_classes: %s' % n_classes)
    log.info('n_hidden1: %s' % n_hidden1)

    tf.set_random_seed(777)  # for reproducibility

    X = tf.placeholder(tf.float32, [None, n_features])  # two characters
    Y = tf.placeholder(tf.float32, [None, n_classes])

    W1 = tf.Variable(tf.random_normal([n_features, n_hidden1]), name='weight1')
    b1 = tf.Variable(tf.random_normal([n_hidden1]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_normal([n_hidden1, n_classes]), name='weight2')
    b2 = tf.Variable(tf.random_normal([n_classes]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))  # cost/loss function
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)  # 20분, Accuracy: 0.689373, cost: 0.8719
    # train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  #

    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)  # 0 <= hypothesis <= 1
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    log.info('create tensorflow graph OK.\n')

    log.info('learning...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        n_input = 0
        log.info('total: %s' % NumUtil.comma_str(train.size))
        for step, (features_batch, labels_batch) in enumerate(train.next_batch(batch_size=batch_size), 1):
            n_input += batch_size
            sess.run(train_step, feed_dict={X: features_batch, Y: labels_batch})
            log.info('[%s][%.1f%%] cost: %.4f' % (NumUtil.comma_str(n_input), n_input / train.size * 100,
                                                  sess.run(cost, feed_dict={X: validation.features, Y: validation.labels})))
            # log.info(sess.run(W1))
            # log.info(sess.run(W2))

        _hypothesis, _correct, _accuracy = sess.run([hypothesis, predicted, accuracy], feed_dict={X: test.features, Y: test.labels})  # Accuracy report
        log.info('model save... %s' % model_file)
        model_dir = os.path.dirname(model_file)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        saver = tf.train.Saver()
        saver.save(sess, model_file)
        log.info('model save OK. %s' % model_file)

    log.info('learning OK.\n')
    log.info('')
    # log.info('hypothesis: %s %s' % (_hypothesis.shape, _hypothesis))
    # log.info('correct: %s %s' % (_correct.shape, _correct))
    log.info('accuracy: %s %s' % (_accuracy.shape, _accuracy))
    log.info('')


if __name__ == '__main__':
    log.info('is_my_pc(): %s' % is_my_pc())
    sentences_file = KO_WIKIPEDIA_ORG_SENTENCES_FILE
    batch_size = 1000  # mini batch size
    left_gram, right_gram = 2, 2
    n_hidden1 = 100
    max_sentences = 100 if is_my_pc() else 10000
    # max_sentences = 100 if is_my_pc() else 0  # FIXME: TEST

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

    model_file = os.path.join(KO_WIKIPEDIA_ORG_DATA_DIR, 'models/word_spacing_model')
    learning(sentences_file, batch_size, left_gram, right_gram, model_file, n_hidden1=n_hidden1, max_sentences=max_sentences, learning_rate=0.1)

    # @formatter:off
    # log.info('chek result...') # TODO:
    # with tf.Session() as sess:
    #     saver = tf.train.Saver()
    #     saver.restore(sess, model_file)
    # log.info('chek result OK.')
