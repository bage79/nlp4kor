import gzip
import os
import sys
import time
import traceback

import numpy as np
import tensorflow as tf

from bage_utils.datafile_util import DataFileUtil
from bage_utils.dataset import DataSet
from bage_utils.file_util import FileUtil
from bage_utils.num_util import NumUtil
from bage_utils.one_hot_vector import OneHotVector
from bage_utils.slack_util import SlackUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor.config import log, KO_WIKIPEDIA_ORG_DIR, KO_WIKIPEDIA_ORG_CHARACTERS_FILE, \
    KO_WIKIPEDIA_ORG_WORD_SPACING_MODEL_DIR, KO_WIKIPEDIA_ORG_TRAIN_SENTENCES_FILE, KO_WIKIPEDIA_ORG_TEST_SENTENCES_FILE, KO_WIKIPEDIA_ORG_VALID_SENTENCES_FILE, \
    KO_WIKIPEDIA_ORG_SENTENCES_FILE


class WordSpacing(object):
    graph_nodes = {}

    @classmethod
    def learning(cls, total_epoch, n_train, n_valid, n_test, batch_size, left_gram, right_gram, model_file, features_vector, labels_vector, n_hidden1=100,
                 learning_rate=0.01, early_stop_cost=0.001):
        ngram = left_gram + right_gram
        n_features = len(features_vector) * ngram  # number of features = 17,380 * 4
        n_classes = len(labels_vector) if len(labels_vector) >= 3 else 1  # number of classes = 2 but len=1

        log.info('load characters list...')
        log.info('load characters list OK. len: %s\n' % NumUtil.comma_str(len(features_vector)))
        watch = WatchUtil()

        train_file = os.path.join(KO_WIKIPEDIA_ORG_DIR, 'datasets', 'word_spacing',
                                  'ko.wikipedia.org.dataset.sentences=%s.left=%d.right=%d.train.gz' % (n_train, left_gram, right_gram))
        valid_file = os.path.join(KO_WIKIPEDIA_ORG_DIR, 'datasets', 'word_spacing',
                                  'ko.wikipedia.org.dataset.sentences=%s.left=%d.right=%d.test.gz' % (n_valid, left_gram, right_gram))
        test_file = os.path.join(KO_WIKIPEDIA_ORG_DIR, 'datasets', 'word_spacing',
                                 'ko.wikipedia.org.dataset.sentences=%s.left=%d.right=%d.valid.gz' % (n_test, left_gram, right_gram))

        log.info('train_file: %s' % train_file)
        log.info('valid_file: %s' % valid_file)
        log.info('test_file: %s' % test_file)
        if not os.path.exists(train_file) or not os.path.exists(valid_file) or not os.path.exists(test_file):
            dataset_dir = os.path.dirname(train_file)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)

            watch.start('create dataset')
            log.info('create dataset...')

            data_files = (('train', KO_WIKIPEDIA_ORG_TRAIN_SENTENCES_FILE, n_train, train_file, False),
                          ('valid', KO_WIKIPEDIA_ORG_VALID_SENTENCES_FILE, n_valid, valid_file, False),
                          ('test', KO_WIKIPEDIA_ORG_TEST_SENTENCES_FILE, n_test, test_file, False))

            for name, data_file, total, dataset_file, to_one_hot_vector in data_files:
                check_interval = 10000
                log.info('check_interval: %s' % check_interval)
                log.info('%s %s total: %s' % (name, os.path.basename(data_file), NumUtil.comma_str(total)))

                features, labels = [], []
                with gzip.open(data_file, 'rt') as f:
                    for i, line in enumerate(f, 1):
                        if total < i:
                            break

                        if i % check_interval == 0:
                            time.sleep(0.01)  # prevent cpu overload
                            percent = i / total * 100
                            log.info('create dataset... %.1f%% readed. data len: %s. %s' % (percent, NumUtil.comma_str(len(features)), data_file))

                        _f, _l = WordSpacing.sentence2features_labels(line.strip(), left_gram=left_gram, right_gram=right_gram)
                        features.extend(_f)
                        labels.extend(_l)

                dataset = DataSet(features=features, labels=labels, features_vector=features_vector, labels_vector=labels_vector, name=name)
                log.info('dataset save... %s' % dataset_file)
                dataset.save(dataset_file, gzip_format=True, verbose=True)
                log.info('dataset save OK. %s' % dataset_file)
                log.info('dataset: %s' % dataset)

            log.info('create dataset OK.')
            log.info('')
            watch.stop('create dataset')

        watch.start('dataset load')
        log.info('dataset load...')
        train = DataSet.load(train_file, gzip_format=True, verbose=True)

        if n_train >= int('100,000'.replace(',', '')):
            valid = DataSet.load(valid_file, gzip_format=True, verbose=True)
        else:
            valid = DataSet.load(train_file, gzip_format=True, verbose=True)
        log.info('valid.convert_to_one_hot_vector()...')
        valid = valid.convert_to_one_hot_vector(verbose=True)
        log.info('valid.convert_to_one_hot_vector() OK.')

        log.info('train dataset: %s' % train)
        log.info('valid dataset: %s' % valid)
        log.info('dataset load OK.')
        log.info('')
        watch.stop('dataset load')

        graph = WordSpacing.build_FFNN(n_features, n_classes, n_hidden1, learning_rate, watch)

        train_step, X, Y, cost, predicted, accuracy = graph['train_step'], graph['X'], graph['Y'], graph['cost'], graph['predicted'], graph['accuracy']

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            check_interval = 10  # max(1, min(1000, n_train // 10))
            nth_train, nth_input, total_input = 0, 0, total_epoch * train.size

            log.info('learn...')
            log.info('total: %s' % NumUtil.comma_str(train.size))
            watch.start('learn')
            valid_cost = sys.float_info.max
            for epoch in range(1, total_epoch + 1):
                if valid_cost < early_stop_cost:
                    break
                for step, (features_batch, labels_batch) in enumerate(train.next_batch(batch_size=batch_size), 1):
                    if valid_cost < early_stop_cost:
                        log.info('valid_cost: %s, early_stop_cost: %s, early stopped.' % (valid_cost, early_stop_cost))
                        break
                    nth_train += 1
                    nth_input += features_batch.shape[0]
                    sess.run(train_step, feed_dict={X: features_batch, Y: labels_batch})

                    # if step % check_interval == 1:
                    percent = nth_input / total_input * 100
                    valid_cost = sess.run(cost, feed_dict={X: valid.features, Y: valid.labels})
                    log.info('[epoch=%s][%.1f%%] %s cost: %.4f' % (epoch, percent, valid.name, valid_cost))
            watch.stop('learn')
            log.info('learn OK.\n')

            log.info('model save... %s' % model_file)
            watch.start('model save...')
            model_dir = os.path.dirname(model_file)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            saver = tf.train.Saver()
            saver.save(sess, model_file)
            watch.stop('model save...')
            log.info('model save OK. %s' % model_file)

        log.info('\n')
        log.info('batch_size: %s' % batch_size)
        log.info(watch.summary())
        log.info('\n')

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
    def spacing(cls, sentence, labels):
        sentence = sentence.replace(' ', '')
        left = []
        for idx, right in enumerate(sentence[:-1]):
            left.append(right)
            if labels[idx]:
                left.append(' ')
        left.append(sentence[-1])
        return ''.join(left)

    @classmethod
    def build_FFNN(cls, n_features, n_classes, n_hidden1, learning_rate, watch=WatchUtil()):  # TODO: 2 layers
        log.info('\nbuild_FFNN')
        if len(cls.graph_nodes) == 0:
            n_hidden3 = n_hidden2 = n_hidden1
            log.info('create tensorflow graph...')
            watch.start('create tensorflow graph')
            log.info('n_features: %s' % n_features)
            log.info('n_classes: %s' % n_classes)
            log.info('n_hidden1: %s' % n_hidden1)
            log.info('n_hidden2: %s' % n_hidden2)
            log.info('n_hidden3: %s' % n_hidden3)

            tf.set_random_seed(777)  # for reproducibility

            X = tf.placeholder(tf.float32, [None, n_features], name='X')  # two characters
            Y = tf.placeholder(tf.float32, [None, n_classes], name='Y')

            # W1 = tf.Variable(tf.truncated_normal([n_features, n_hidden1], mean=0.0, stddev=0.1), name='W1')
            # b1 = tf.Variable(tf.constant(0.1, shape=[n_hidden1]), name='b1')
            W1 = tf.Variable(tf.random_normal([n_features, n_hidden1]), name='W1')
            b1 = tf.Variable(tf.random_normal([n_hidden1]), name='b1')
            layer1 = tf.nn.relu(tf.matmul(X, W1) + b1, name='layer1')

            W2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]), name='W2')
            b2 = tf.Variable(tf.random_normal([n_hidden2]), name='b2')
            layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2, name='layer2')

            W3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]), name='W3')
            b3 = tf.Variable(tf.random_normal([n_hidden3]), name='b3')
            layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3, name='layer3')

            W4 = tf.Variable(tf.random_normal([n_hidden3, n_classes]), name='W4')
            b4 = tf.Variable(tf.random_normal([n_classes]), name='b4')
            y_hat = tf.add(tf.matmul(layer3, W4), b4, name='y_hat')

            # cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis), name='cost')  # cost/loss function
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=Y), name='cost')

            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                cost)  # Very Very good!! sentences=10000 + layer=4, 10분, accuracy 0.9294, cost: 0.1839

            predicted = tf.cast(y_hat > 0.5, dtype=tf.float32, name='predicted')  # 0 <= hypothesis <= 1

            accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32), name='accuracy')

            watch.stop('create tensorflow graph')
            log.info('create tensorflow graph OK.\n')
            cls.graph_nodes = {'predicted': predicted, 'accuracy': accuracy, 'X': X, 'Y': Y, 'train_step': train_step, 'cost': cost}
        return cls.graph_nodes

    @classmethod
    def sim_two_sentence(cls, original, generated, left_gram=2, right_gram=2):
        _, labels1 = WordSpacing.sentence2features_labels(original, left_gram=left_gram, right_gram=right_gram)
        _, labels2 = WordSpacing.sentence2features_labels(generated, left_gram=left_gram, right_gram=right_gram)
        incorrect = 0
        for idx, l in enumerate(labels1):
            if l == 1 and labels2[idx] != 1:
                incorrect += 1

        total_spaces = labels1.count(1)  # 정답에 있는 공백 개수
        correct = total_spaces - incorrect  # 정답에 있는 공백과 같은 곳에 공백이 있는지

        if total_spaces == 0:
            sim = 1
        else:
            sim = correct / total_spaces
        return sim, correct, total_spaces


if __name__ == '__main__':
    train_sentences_file = KO_WIKIPEDIA_ORG_TRAIN_SENTENCES_FILE
    valid_sentences_file = KO_WIKIPEDIA_ORG_VALID_SENTENCES_FILE
    test_sentences_file = KO_WIKIPEDIA_ORG_TEST_SENTENCES_FILE
    log.info('train_sentences_file: %s' % train_sentences_file)
    log.info('valid_sentences_file: %s' % valid_sentences_file)
    log.info('test_sentences_file: %s' % test_sentences_file)

    characters_file = KO_WIKIPEDIA_ORG_CHARACTERS_FILE
    log.info('characters_file: %s' % characters_file)
    try:
        if len(sys.argv) == 4:
            n_train = int(sys.argv[1])
            left_gram = int(sys.argv[2])
            right_gram = int(sys.argv[3])
        else:
            n_train, left_gram, right_gram = 1000000, 3, 3
            # n_train, left_gram, right_gram = int('1,000,000'.replace(',', '')), 2, 2

        if left_gram is None:
            left_gram = 2
        if right_gram is None:
            right_gram = 2

        ngram = left_gram + right_gram
        n_valid, n_test = 100, 100
        log.info('n_train: %s' % NumUtil.comma_str(n_train))
        log.info('n_valid: %s' % NumUtil.comma_str(n_valid))
        log.info('n_test: %s' % NumUtil.comma_str(n_test))
        log.info('left_gram: %s, right_gram: %s' % (left_gram, right_gram))
        log.info('ngram: %s' % ngram)

        total_sentences = FileUtil.count_lines(KO_WIKIPEDIA_ORG_SENTENCES_FILE)
        model_file = os.path.join(KO_WIKIPEDIA_ORG_WORD_SPACING_MODEL_DIR,
                                  'word_spacing_model.sentences=%s.left_gram=%s.right_gram=%s/model' % (
                                      n_train, left_gram, right_gram))  # .%s' % max_sentences
        log.info('model_file: %s' % model_file)

        batch_size = 1000  # mini batch size
        log.info('batch_size: %s' % batch_size)

        total_epoch = min(100, 1000000 // n_train)  # 1 ~ 100
        features_vector = OneHotVector(DataFileUtil.read_list(characters_file))
        labels_vector = OneHotVector([0, 1])  # 붙여쓰기=0, 띄어쓰기=1
        n_features = len(features_vector) * ngram  # number of features = 17,380 * 4
        n_classes = len(labels_vector) if len(labels_vector) >= 3 else 1  # number of classes = 2 but len=1
        n_hidden1 = 100
        learning_rate = min(0.1, 0.001 * total_epoch)  # 0.1 ~ 0.001
        early_stop_cost = 0.0001
        log.info('features_vector: %s' % features_vector)
        log.info('labels_vector: %s' % labels_vector)
        log.info('n_features: %s' % n_features)
        log.info('n_classes: %s' % n_classes)
        log.info('n_hidden1: %s' % n_hidden1)
        log.info('learning_rate: %s' % learning_rate)
        log.info('early_stop_cost: %s' % early_stop_cost)
        log.info('total_epoch: %s' % total_epoch)

        # log.info('sample testing...')
        # test_set = ['예쁜 운동화', '즐거운 동화', '삼풍동 화재']
        # for s in test_set:
        #     features, labels = WordSpacing.sentence2features_labels(s, left_gram=left_gram, right_gram=right_gram)
        #     log.info('%s -> %s' % (features, labels))
        #     log.info('in : "%s"' % s)
        #     log.info('out: "%s"' % WordSpacing.spacing(s.replace(' ', ''), labels))
        # log.info('sample testing OK.\n')

        if not os.path.exists(model_file + '.index') or not os.path.exists(model_file + '.meta'):
            if n_train >= int('100,000'.replace(',', '')):
                SlackUtil.send_message('%s start (max_sentences=%s, left_gram=%s, right_gram=%.1f)' % (sys.argv[0], n_train, left_gram, right_gram))
            WordSpacing.learning(total_epoch, n_train, n_valid, n_test, batch_size, left_gram, right_gram, model_file, features_vector, labels_vector,
                                 n_hidden1=n_hidden1,
                                 learning_rate=learning_rate, early_stop_cost=early_stop_cost)
            if n_train >= int('100,000'.replace(',', '')):
                SlackUtil.send_message('%s end (max_sentences=%s, left_gram=%s, right_gram=%.1f)' % (sys.argv[0], n_train, left_gram, right_gram))

        log.info('chek result...')
        watch = WatchUtil()
        watch.start('read sentences')

        sentences = []  # '아버지가 방에 들어 가신다.', '가는 말이 고와야 오는 말이 곱다.']
        max_test_sentences = 100

        if n_train >= int('100,000'.replace(',', '')):
            sentences_file = test_sentences_file
        else:
            sentences_file = train_sentences_file

        with gzip.open(sentences_file, 'rt') as f:
            for i, line in enumerate(f, 1):
                if len(sentences) >= max_test_sentences:
                    break

                s = line.strip()
                if s.count(' ') > 0:  # sentence must have one or more space.
                    sentences.append(s)
        log.info('len(sentences): %s' % NumUtil.comma_str(len(sentences)))
        watch.stop('read sentences')

        watch.start('run tensorflow')
        accuracies, sims = [], []
        with tf.Session() as sess:
            graph = WordSpacing.build_FFNN(n_features, n_classes, n_hidden1, learning_rate)
            X, Y, predicted, accuracy = graph['X'], graph['Y'], graph['predicted'], graph['accuracy']

            saver = tf.train.Saver()
            try:
                restored = saver.restore(sess, model_file)
            except:
                log.error('restore failed. model_file: %s' % model_file)
            try:
                for i, s in enumerate(sentences):
                    log.info('')
                    log.info('[%s] in : "%s"' % (i, s))
                    _features, _labels = WordSpacing.sentence2features_labels(s, left_gram, right_gram)
                    dataset = DataSet(features=_features, labels=_labels, features_vector=features_vector, labels_vector=labels_vector)
                    dataset.convert_to_one_hot_vector()
                    if len(dataset) > 0:
                        _predicted, _accuracy = sess.run([predicted, accuracy], feed_dict={X: dataset.features, Y: dataset.labels})  # Accuracy report

                        sentence_hat = WordSpacing.spacing(s.replace(' ', ''), _predicted)
                        sim, correct, total = WordSpacing.sim_two_sentence(s, sentence_hat, left_gram=left_gram, right_gram=right_gram)

                        accuracies.append(_accuracy)
                        sims.append(sim)

                        log.info('[%s] out: "%s" (accuracy: %.1f%%, sim: %.1f%%=%s/%s)' % (i, sentence_hat, _accuracy * 100, sim * 100, correct, total))
            except:
                log.error(traceback.format_exc())

        log.info('chek result OK.')
        # noinspection PyStringFormat
        log.info('mean(accuracy): %.2f%%, mean(sim): %.2f%%' % (np.mean(accuracies) * 100, np.mean(sims) * 100))
        log.info('secs/sentence: %.4f' % (watch.elapsed('run tensorflow') / len(sentences)))
        log.info(watch.summary())
    except:
        log.error(traceback.format_exc())
