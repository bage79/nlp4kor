import gzip
import math
import os
import re
import sys
import time
import traceback

import numpy as np
import tensorflow as tf

from bage_utils.base_util import is_server, is_my_pc, is_my_gpu_pc
from bage_utils.datafile_util import DataFileUtil
from bage_utils.dataset import DataSet
from bage_utils.hangul_util import HangulUtil
from bage_utils.num_util import NumUtil
from bage_utils.one_hot_vector import OneHotVector
from bage_utils.slack_util import SlackUtil
from bage_utils.string_util import StringUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor.config import log, KO_WIKIPEDIA_ORG_SPELLING_ERROR_CORRECTION_MODEL_DIR, \
    KO_WIKIPEDIA_ORG_TRAIN_SENTENCES_FILE, KO_WIKIPEDIA_ORG_VALID_SENTENCES_FILE, KO_WIKIPEDIA_ORG_TEST_SENTENCES_FILE, KO_WIKIPEDIA_ORG_CHARACTERS_FILE, \
    KO_WIKIPEDIA_ORG_DIR


class SpellingErrorCorrection(object):
    graph = {}

    @classmethod
    def learning(cls, total_epoch, n_train, n_valid, n_test, batch_size, window_size, noise_rate, noise_sampling, model_file, features_vector, labels_vector,
                 n_hidden1,
                 learning_rate,
                 dropout_keep_rate):
        n_features = len(features_vector) * window_size  # number of features = 17,382 * 10

        log.info('load characters list...')
        log.info('load characters list OK. len: %s' % NumUtil.comma_str(len(features_vector)))
        watch = WatchUtil()

        train_file = os.path.join(KO_WIKIPEDIA_ORG_DIR, 'datasets', 'spelling_error_correction',
                                  'ko.wikipedia.org.dataset.sentences=%s.window_size=%d.train.gz' % (n_train, window_size))
        valid_file = os.path.join(KO_WIKIPEDIA_ORG_DIR, 'datasets', 'spelling_error_correction',
                                  'ko.wikipedia.org.dataset.sentences=%s.window_size=%d.train.gz' % (n_valid, window_size))
        test_file = os.path.join(KO_WIKIPEDIA_ORG_DIR, 'datasets', 'spelling_error_correction',
                                 'ko.wikipedia.org.dataset.sentences=%s.window_size=%d.train.gz' % (n_test, window_size))
        if is_my_pc() or is_my_gpu_pc() or not os.path.exists(train_file) or not os.path.exists(valid_file) or not os.path.exists(test_file):
            dataset_dir = os.path.dirname(train_file)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)

            watch.start('create dataset')
            log.info('create dataset...')

            if is_server():
                data_files = (('train', KO_WIKIPEDIA_ORG_TRAIN_SENTENCES_FILE, n_train, train_file, False),
                              ('valid', KO_WIKIPEDIA_ORG_VALID_SENTENCES_FILE, n_valid, valid_file, False),
                              ('test', KO_WIKIPEDIA_ORG_TEST_SENTENCES_FILE, n_test, test_file, False))
            else:
                data_files = (('train', KO_WIKIPEDIA_ORG_TRAIN_SENTENCES_FILE, n_train, train_file, False),)

            for (name, data_file, total, dataset_file, to_one_hot_vector) in (data_files):
                check_interval = max(1, min(10000, batch_size // 10))
                log.info('check_interval: %s' % check_interval)
                log.info('%s %s total: %s' % (name, os.path.basename(data_file), NumUtil.comma_str(total)))
                log.info('noise_rate: %s' % noise_rate)

                features, labels = [], []
                with gzip.open(data_file, 'rt') as f:
                    for i, line in enumerate(f, 1):
                        if total < i:
                            break

                        if i % check_interval == 0:
                            time.sleep(0.01)  # prevent cpu overload
                            percent = i / total * 100
                            log.info('create dataset... %.1f%% readed. data len: %s. %s' % (percent, NumUtil.comma_str(len(features)), data_file))

                        sentence = line.strip()
                        for start in range(0, len(sentence) - window_size + 1):  # 문자 단위로 노이즈(공백) 생성
                            chars = sentence[start: start + window_size]
                            for idx in range(len(chars)):
                                noised_chars = StringUtil.replace_with_index(chars, ' ', idx)
                                features.append(noised_chars)
                                labels.append(chars)
                                log.debug('create dataset... %s "%s" -> "%s"' % (name, noised_chars, chars))

                # log.info('noise_sampling: %s' % noise_sampling)
                #         for nth_sample in range(noise_sampling): # 초성, 중성, 종성 단위로 노이즈 생성
                #             for start in range(0, len(sentence) - window_size + 1):
                #                 chars = sentence[start: start + window_size]
                #                 noised_chars = SpellingErrorCorrection.encode_noise(chars, noise_rate=noise_rate, noise_with_blank=True)
                #                 if chars == noised_chars:
                #                     continue
                #                 if i % check_interval == 0 and nth_sample == 0:
                #                     log.info('create dataset... %s "%s" -> "%s"' % (name, noised_chars, chars))
                #                 features.append(noised_chars)
                #                 labels.append(chars)

                # print('dataset features:', features)
                # print('dataset labels:', labels)
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

        if is_server():
            valid = DataSet.load(valid_file, gzip_format=True, verbose=True)
        else:
            valid = DataSet.load(train_file, gzip_format=True, verbose=True)  # valid with train set
        log.info('valid.convert_to_one_hot_vector()...')
        valid = valid.convert_to_one_hot_vector(verbose=True)
        log.info('valid.convert_to_one_hot_vector() OK.')

        log.info('train dataset: %s' % train)
        log.info('valid dataset: %s' % valid)
        log.info('dataset load OK.')
        log.info('')
        watch.stop('dataset load')

        X, Y, dropout_keep_prob, train_step, cost, y_hat, accuracy = SpellingErrorCorrection.build_DAE(n_features, window_size, noise_rate, n_hidden1,
                                                                                                       learning_rate, watch)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            check_interval = max(1, min(1000, n_train // 10))

            nth_train, nth_input, total_input = 0, 0, total_epoch * train.size
            log.info('')
            log.info('learn...')
            log.info('total_epoch: %s' % total_epoch)
            log.info('train.size (total features): %s' % NumUtil.comma_str(train.size))
            log.info('check_interval: %s' % check_interval)
            log.info('total_epoch: %s' % total_epoch)
            log.info('batch_size: %s' % batch_size)
            log.info('total_input: %s (total_epoch * train.size)' % total_input)
            log.info('')
            watch.start('learn')
            for epoch in range(1, total_epoch + 1):
                for step, (features_batch, labels_batch) in enumerate(train.next_batch(batch_size=batch_size, to_one_hot_vector=True), 1):
                    nth_train += 1
                    nth_input += features_batch.shape[0]
                    sess.run(train_step, feed_dict={X: features_batch, Y: labels_batch, dropout_keep_prob: dropout_keep_rate})

                    percent = nth_input / total_input * 100

                    # if nth_train % check_interval == 1:
                    valid_cost = sess.run(cost, feed_dict={X: valid.features, Y: valid.labels, dropout_keep_prob: 1.0})
                    log.info('[epoch=%s][%.1f%%] %s cost: %.8f' % (epoch, percent, valid.name, valid_cost))

            watch.stop('learn')
            log.info('learn OK.')
            log.info('')

            log.info('model save... %s' % model_file)
            watch.start('model save...')
            model_dir = os.path.dirname(model_file)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            saver = tf.train.Saver()
            saver.save(sess, model_file)
            watch.stop('model save...')
            log.info('model save OK. %s' % model_file)

        log.info('')
        log.info('total_epoch: %s' % total_epoch)
        log.info('batch_size: %s' % batch_size)
        log.info('total_input: %s (total_epoch * train.size)' % total_input)
        log.info('')
        log.info(watch.summary())
        log.info('')

    @classmethod
    def build_DAE(cls, n_features, window_size, noise_rate, n_hidden1, learning_rate, watch=WatchUtil()):
        if len(cls.graph) == 0:
            log.info('')
            log.info('create tensorflow graph...')
            watch.start('create tensorflow graph')

            features_vector_size = n_features // window_size
            log.info('n_features: %s' % n_features)
            log.info('window_size: %s' % window_size)
            log.info('features_vector_size: %s' % features_vector_size)

            log.info('noise_rate: %.1f' % noise_rate)
            log.info('n_hidden1: %s' % n_hidden1)

            tf.set_random_seed(777)  # for reproducibility

            X = tf.placeholder(tf.float32, [None, n_features], name='X')  # shape=(batch_size, window_size * feature_vector.size)
            Y = tf.placeholder(tf.float32, [None, n_features], name='Y')  # shape=(batch_size, window_size * feature_vector.size)
            dropout_keep_prob = tf.placeholder(tf.float32)

            # layers = 3
            # n_hidden2 = n_hidden1
            # W1 = tf.Variable(tf.random_normal([n_features, n_hidden1]), name='W1')
            # b1 = tf.Variable(tf.random_normal([n_hidden1]), name='b1')
            # layer1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1, name='layer1')
            # layer1_dropout = tf.nn.dropout(layer1, dropout_keep_prob, name='layer1_dropout')
            #
            # W2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]), name='W2')
            # b2 = tf.Variable(tf.random_normal([n_hidden2]), name='b2')
            # layer2 = tf.nn.sigmoid(tf.matmul(layer1_dropout, W2) + b2, name='layer2')
            # layer2_dropout = tf.nn.dropout(layer2, dropout_keep_prob, name='layer2_dropout')
            #
            # W3 = tf.Variable(tf.random_normal([n_hidden1, n_features]), name='W3')
            # b3 = tf.Variable(tf.random_normal([n_features]), name='b3')
            # y_hat = tf.add(tf.matmul(layer2_dropout, W3), b3, name='y_hat')

            # layers = 2
            W1 = tf.Variable(tf.random_normal([n_features, n_hidden1]), name='W1')
            b1 = tf.Variable(tf.random_normal([n_hidden1]), name='b1')
            layer1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1, name='layer1')
            layer1_dropout = tf.nn.dropout(layer1, dropout_keep_prob, name='layer1_dropout')

            W2 = tf.Variable(tf.random_normal([n_hidden1, n_features]), name='W2')
            b2 = tf.Variable(tf.random_normal([n_features]), name='b2')
            y_hat = tf.add(tf.matmul(layer1_dropout, W2), b2, name='y_hat')  # shape=(batch_size, window_size * feature_vector.size)

            labels_hat = tf.reshape(y_hat, shape=(-1, window_size, features_vector_size))  # shape=(batch_size, window_size, feature_vector.size)
            labels = tf.reshape(Y, shape=(-1, window_size, features_vector_size))  # shape=(batch_size, window_size, feature_vector.size)

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=labels_hat, labels=labels), name='cost')
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            accuracy = tf.reduce_mean(tf.cast(tf.abs(tf.nn.softmax(y_hat) - Y) < 0.1, dtype=tf.float32), name='accuracy')
            # log.debug('X:', X)
            # log.debug('Y:', Y)
            # log.debug('y_hat:', y_hat)
            # log.debug('labels_hat:', labels_hat)
            # log.debug('labels:', labels)
            # log.debug('cost:', cost)
            # log.debug('accuracy:', accuracy)

            watch.stop('create tensorflow graph')
            log.info('create tensorflow graph OK.')
            log.info('')
            cls.graph = {'X': X, 'Y': Y, 'dropout_keep_prob': dropout_keep_prob,
                         'train_step': train_step, 'cost': cost, 'y_hat': y_hat, 'accuracy': accuracy, }
        return cls.graph['X'], cls.graph['Y'], cls.graph['dropout_keep_prob'], \
               cls.graph['train_step'], cls.graph['cost'], cls.graph['y_hat'], cls.graph['accuracy']

    @classmethod
    def sentence2features_labels(cls, sentence, noised_sentence, window_size=10, skip_same=False) -> (list, list):
        try:
            sentence, noised_sentence = sentence.strip(), noised_sentence.strip()
            if window_size < 1 or len(sentence) != len(noised_sentence):  # or len(sentence) <= 1 :
                return [], []

            features, labels = [], []
            for start in range(0, len(sentence) - window_size + 1):
                feature, label = noised_sentence[start:start + window_size], sentence[start:start + window_size]
                if skip_same and feature != label:
                    continue

                features.append(feature)
                labels.append(label)
            return features, labels
        except:
            return [], []

    @classmethod
    def encode_noise(cls, s, noise_rate=0.1, noise_with_blank=False, verbose=False):
        try:
            hangul_indexs = [idx for idx, c in enumerate(s) if HangulUtil.is_hangul_char(c)]
            if len(hangul_indexs) == 0:
                return s
            target_indexs = np.random.choice(hangul_indexs, math.ceil(len(hangul_indexs) * noise_rate), replace=False)
            _s = list(s)
            for idx in target_indexs:
                if noise_with_blank:
                    _s[idx] = ' '
                else:
                    c = s[idx]
                    _c = HangulUtil.encode_noise(c)
                    if verbose:
                        log.info('encode: %s -> %s' % (c, _c))
                    _s[idx] = _c
            return ''.join(_s)
        except:
            return s

    @classmethod
    def decode_noise(cls, noised_sentence, features_list, labels_list, verbose=False):
        try:
            if len(features_list) != len(labels_list) or len(features_list[0]) != len(labels_list[0]):
                return noised_sentence

            idx2chars = dict()
            for feature, label in zip(features_list, labels_list):
                for off in [i for i in range(len(feature)) if feature[i] != label[i]]:
                    for start in [m.start() for m in re.finditer(feature, noised_sentence)]:
                        idx = start + off
                        if idx not in idx2chars:
                            idx2chars[idx] = label[off]
            sentence = list(noised_sentence)
            for idx, char in idx2chars.items():
                if verbose:
                    log.info('denoise: "%s" -> "%s"' % (noised_sentence[idx], char))
                sentence[idx] = char
            return ''.join(sentence)
        except:
            return noised_sentence

    @classmethod
    def sim_two_sentence(cls, sentence, sentence_hat):
        sim, correct, total = 0, 0, len(sentence)
        if len(sentence) != len(sentence_hat):
            return sim, correct, total

        for a, b in zip(sentence, sentence_hat):
            if a == b:
                correct += 1

        sim = correct / total
        return sim, correct, total


if __name__ == '__main__':
    train_sentences_file = KO_WIKIPEDIA_ORG_TRAIN_SENTENCES_FILE
    valid_sentences_file = KO_WIKIPEDIA_ORG_VALID_SENTENCES_FILE
    test_sentences_file = KO_WIKIPEDIA_ORG_TEST_SENTENCES_FILE
    log.info('train_sentences_file: %s' % train_sentences_file)
    log.info('valid_sentences_file: %s' % valid_sentences_file)
    log.info('test_sentences_file: %s' % test_sentences_file)
    log.info('')

    characters_file = KO_WIKIPEDIA_ORG_CHARACTERS_FILE
    log.info('characters_file: %s' % characters_file)
    try:
        if len(sys.argv) == 4:
            n_train = int(sys.argv[1])
            window_size = int(sys.argv[2])
            noise_rate = float(sys.argv[3])
        else:
            n_train, noise_rate, window_size = None, None, None

        if n_train is None or n_train == 0:
            n_train = int('1,000,000'.replace(',', ''))

        if is_server():  # batch server
            n_train = 10000
            n_valid = min(10, n_train)
            n_test = min(10, n_train)
        else:  # for demo
            n_train = n_valid = n_test = 3

        if noise_rate is None or window_size is None:
            window_size = 10  # 2 ~ 10 # feature로 추출할 문자 수 (label과 동일)
            noise_rate = max(0.1, 1 / window_size)  # 0.0 ~ 1.0 # noise_rate = 노이즈 문자 수 / 전체 문자 수 (windos 안에서 최소 한 글자는 노이즈가 생기도록 함.)

        dropout_keep_rate = 1.0  # 0.0 ~ 1.0 # one hot vector에 경우에 dropout 사용시, 학습이 안 됨.
        noise_sampling = 100  # 한 입력에 대하여 몇 개의 노이즈 샘플을 생성할지. blank 방식(문자 단위)으로 noise 생성할 때는 사용 안함.

        total_epoch = max(10, 100 // window_size)  # 10 ~ 100 # window_size 가 클 수록 total_epoch는 작아도 됨.
        batch_size = min(100, 10 * window_size)  # 1 ~ 100 # one hot vector 입력이면, batch_size 작게 잡아야 학습이 잘 된다. batch_size가 너무 크면, 전혀 학습이 안 됨.

        n_hidden1 = min(1000, 10 * window_size)  # 10 ~ 1000
        learning_rate = 1 / total_epoch  # 0.01  # 0.1 ~ 0.001  # total_epoch 가 클 수록 learning_rate는 작아도 됨.

        log.info('')
        log.info('n_train (sentences): %s' % NumUtil.comma_str(n_train))
        log.info('n_valid (sentences): %s' % NumUtil.comma_str(n_valid))
        log.info('n_test (sentences): %s' % NumUtil.comma_str(n_test))
        log.info('')
        log.info('window_size: %s' % window_size)
        log.info('noise_rate: %s' % noise_rate)
        log.info('dropout_keep_rate: %s' % dropout_keep_rate)
        log.info('noise_sampling: %s' % noise_sampling)
        log.info('')
        log.info('n_hidden1: %s' % n_hidden1)
        log.info('learning_rate: %s' % learning_rate)
        log.info('')
        log.info('total_epoch: %s' % total_epoch)
        log.info('batch_size: %s' % batch_size)

        model_file = os.path.join(KO_WIKIPEDIA_ORG_SPELLING_ERROR_CORRECTION_MODEL_DIR,
                                  'spelling_error_correction_model.sentences=%s.window_size=%s.noise_rate=%.1f.n_hidden=%s/model' % (
                                      n_train, window_size, noise_rate, n_hidden1))  # .%s' % max_sentences
        log.info('model_file: %s' % model_file)

        features_vector = OneHotVector(DataFileUtil.read_list(characters_file), added=[' '])
        labels_vector = OneHotVector(DataFileUtil.read_list(characters_file), added=[' '])
        n_features = len(features_vector) * window_size  # number of features = 17,450 * 10
        n_classes = len(labels_vector) * window_size  # number of features = 17,450 * 10
        log.info('')
        log.info('features_vector: %s' % features_vector)
        log.info('labels_vector: %s' % labels_vector)
        log.info('n_features (input vector size): %s' % n_features)
        log.info('n_classes (input vector size): %s' % n_classes)

        log.info('')
        log.info('sample testing...')
        DEMO_SENTENCES = ['아버지가 방에 들어 가신다.', '오빠가 방에 들어 간다.', '이모가 방에 들어 가신다.']
        for i, sentence in enumerate(DEMO_SENTENCES):
            noised_sentence = SpellingErrorCorrection.encode_noise(sentence, noise_rate=0.1, noise_with_blank=True)
            log.info('[%s] ori: "%s"' % (i, sentence))
            _features, _labels = SpellingErrorCorrection.sentence2features_labels(sentence, noised_sentence, window_size)
            for _f, _l in zip(_features, _labels):
                log.info('learn "%s" -> "%s"' % (_f, _l))

            sentence_hat = SpellingErrorCorrection.decode_noise(noised_sentence, _features, _labels)
            log.info('[%s] in : "%s"' % (i, noised_sentence))

            sim, correct, total = SpellingErrorCorrection.sim_two_sentence(sentence, sentence_hat)
            log.info('[%s] out: "%s" (sim: %.1f%%=%s/%s)' % (i, sentence_hat, sim * 100, correct, total))
        log.info('sample testing OK.')
        log.info('')

        # if is_my_pc() or is_my_gpu_pc() or not os.path.exists(model_file + '.index') or not os.path.exists(model_file + '.meta'):
        if not os.path.exists(model_file + '.index') or not os.path.exists(model_file + '.meta'):
            if n_train > int('100,000'.replace(',', '')):
                SlackUtil.send_message('%s start (max_sentences=%s, window_size=%s, noise_rate=%.1f)' % (sys.argv[0], n_train, window_size, noise_rate))

            SpellingErrorCorrection.learning(total_epoch, n_train, n_valid, n_test, batch_size, window_size, noise_rate, noise_sampling,
                                             model_file, features_vector, labels_vector, n_hidden1=n_hidden1,
                                             learning_rate=learning_rate, dropout_keep_rate=dropout_keep_rate)
            if n_train > int('100,000'.replace(',', '')):
                SlackUtil.send_message('%s end (max_sentences=%s, window_size=%s, noise_rate=%.1f)' % (sys.argv[0], n_train, window_size, noise_rate))

        log.info('chek result...')
        watch = WatchUtil()
        watch.start('read sentences')

        max_test_sentences = 1
        sentences = []
        if is_server():
            sentences_file = test_sentences_file
        else:
            sentences_file = train_sentences_file

        with gzip.open(sentences_file, 'rt') as f:
            for i, line in enumerate(f, 1):
                if len(sentences) >= n_test:
                    break

                s = line.strip()
                sentences.append(s)
        log.info('len(sentences): %s' % NumUtil.comma_str(len(sentences)))
        watch.stop('read sentences')

        watch.start('run tensorflow')

        with tf.Session() as sess:
            X, Y, dropout_keep_prob, train_step, cost, y_hat, accuracy = SpellingErrorCorrection.build_DAE(n_features, window_size, noise_rate, n_hidden1,
                                                                                                           learning_rate, watch)

            saver = tf.train.Saver()
            try:
                restored = saver.restore(sess, model_file)
            except Exception as e:
                log.error('restore failed. model_file: %s' % model_file)
                raise e

            train_file = os.path.join(KO_WIKIPEDIA_ORG_DIR, 'datasets', 'spelling_error_correction',
                                      'ko.wikipedia.org.dataset.sentences=%s.window_size=%d.train.gz' % (n_train, window_size))
            train = DataSet.load(train_file, gzip_format=True, verbose=True)
            train_vector = DataSet.load(train_file, gzip_format=True, verbose=True)
            train_vector.convert_to_one_hot_vector()
            try:
                accuracies, costs, sims = [], [], []
                total_test_sampling = 1
                for i, sentence in enumerate(sentences):
                    for nth in range(total_test_sampling):
                        # log.info('[%s] noise(%.1f) "%s" -> "%s"' % (nth, noise_rate, sentence, noised_sentence))
                        noised_sentence = SpellingErrorCorrection.encode_noise(sentence, noise_rate=0.1)
                        log.info('')
                        log.info('[%s] ori: "%s"' % (nth, sentence))
                        log.info('[%s] in : "%s"' % (nth, noised_sentence))

                        denoised_sentence = noised_sentence[:]  # will be changed with predict
                        for start in range(0, len(noised_sentence) - window_size + 1):
                            chars = denoised_sentence[start: start + window_size]
                            original_chars = sentence[start: start + window_size]
                            _features = [chars]
                            _labels = [original_chars]

                            dataset = DataSet(features=_features, labels=_labels, features_vector=features_vector, labels_vector=features_vector)
                            dataset.convert_to_one_hot_vector()
                            try:
                                _y_hat, _cost, _accuracy = sess.run([y_hat, cost, accuracy],
                                                         feed_dict={X: dataset.features, Y: dataset.labels, dropout_keep_prob: dropout_keep_rate})
                                costs.append(_cost)
                                accuracies.append(_accuracy)
                            except:
                                log.error('"%s"%s "%s"%s' % (chars, dataset.features.shape, original_chars, dataset.labels.shape))

                            y_hats = [features_vector.to_values(_l) for _l in _y_hat]
                            if _features[0] == y_hats[0]:
                                log.info('same   : "%s"' % (_features[0]))
                            else:
                                log.info('denoise: "%s" -> "%s"' % (_features[0], y_hats[0]))
                            denoised_sentence = denoised_sentence.replace(_features[0], y_hats[0])
                            # print(denoised_sentence)

                        sim, correct, total = SpellingErrorCorrection.sim_two_sentence(sentence, denoised_sentence)
                        sims.append(sim)

                        if correct == total:
                            log.info('[%s] out: "%s" (cost: %.4f, accuracy: %.4f, sim: %.1f%%=%s/%s)' % (
                                nth, denoised_sentence, _cost, _accuracy, sim * 100, correct, total))
                        else:
                            log.error('[%s] out: "%s" (cost: %.4f, accuracy: %.4f, sim: %.1f%%=%s/%s) incorrect!!' % (
                                nth, denoised_sentence, _cost, _accuracy, sim * 100, correct, total))
            except Exception as e:
                raise e

        log.info('chek result OK.')
        # noinspection PyStringFormat
        log.info('mean(cost): %.4f, mean(sim): %.2f%%' % (np.mean(costs), np.mean(sims) * 100))
        log.info('secs/sentence: %.4f' % (watch.elapsed('run tensorflow') / len(sentences)))
        log.info('total_epoch: %s, batch_size: %s' % (NumUtil.comma_str(total_epoch), NumUtil.comma_str(batch_size)))
        log.info(watch.summary())
    except:
        log.error(traceback.format_exc())
