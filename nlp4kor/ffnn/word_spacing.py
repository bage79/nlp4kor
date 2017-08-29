import gzip
import math
import os
import sys
import time
import traceback

import numpy as np
import tensorflow as tf

from bage_utils.char_one_hot_vector import CharOneHotVector
from bage_utils.datafile_util import DataFileUtil
from bage_utils.dataset import DataSet
from bage_utils.file_util import FileUtil
from bage_utils.num_util import NumUtil
from bage_utils.slack_util import SlackUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor.config import log, WIKIPEDIA_CHARACTERS_FILE, \
    WORD_SPACING_MODEL_DIR, WIKIPEDIA_TRAIN_SENTENCES_FILE, WIKIPEDIA_VALID_SENTENCES_FILE, \
    WIKIPEDIA_SENTENCES_FILE, WORD_SPACING_DATASET_DIR, WIKIPEDIA_TEST_SENTENCES_FILE


class WordSpacing(object):
    graph_nodes = {}

    @classmethod
    def load_datasets(cls, train_sentences_file, valid_sentences_file, n_train, n_valid, left_gram, right_gram, features_vector, labels_vector, same_train_valid_data=False):
        log.info('load characters list...')
        log.info('load characters list OK. len: %s\n' % NumUtil.comma_str(len(features_vector)))
        watch = WatchUtil()

        train_dataset_file = os.path.join(WORD_SPACING_DATASET_DIR, 'ko.wikipedia.org.dataset.sentences=%s.left=%d.right=%d.train.gz' % (n_train, left_gram, right_gram))
        valid_dataset_file = os.path.join(WORD_SPACING_DATASET_DIR, 'ko.wikipedia.org.dataset.sentences=%s.left=%d.right=%d.valid.gz' % (n_valid, left_gram, right_gram))

        log.info('train_sentences_file: %s -> %s' % (train_sentences_file, train_dataset_file))
        log.info('valid_sentences_file: %s -> %s' % (valid_sentences_file, valid_dataset_file))
        # log.info('test_file: %s' % test_file)
        if not os.path.exists(train_dataset_file) or not os.path.exists(valid_dataset_file):
            dataset_dir = os.path.dirname(train_dataset_file)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)

            watch.start('create dataset')
            log.info('create dataset...')

            data_files = (('train', train_sentences_file, n_train, train_dataset_file, False),
                          ('valid', valid_sentences_file, n_valid, valid_dataset_file, False),
                          )

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
        train = DataSet.load(train_dataset_file, gzip_format=True, verbose=True)

        if same_train_valid_data:
            valid = DataSet.load(train_dataset_file, gzip_format=True, max_len=n_valid * 10, verbose=True)  # train=valid
        else:
            valid = DataSet.load(valid_dataset_file, gzip_format=True, verbose=True)

        log.info('valid.convert_to_one_hot_vector()...')
        valid = valid.convert_to_one_hot_vector(verbose=True)
        log.info('valid.convert_to_one_hot_vector() OK.')

        log.info('train dataset: %s' % train)
        log.info('valid dataset: %s' % valid)
        log.info('dataset load OK.')
        log.info('')
        watch.stop('dataset load')
        return train, valid

    @classmethod
    def learning(cls, total_epoch, train, valid, batch_size, left_gram, right_gram, model_file, features_vector, labels_vector, activation, n_hidden1, n_layers, learning_rate, decay_steps, decay_rate,
                 early_stop_cost, valid_interval):
        ngram = left_gram + right_gram
        n_features = len(features_vector) * ngram  # number of features = 17,380 * 4
        n_classes = len(labels_vector) if len(labels_vector) >= 3 else 1  # number of classes = 2 but len=1

        graph = WordSpacing.build_FFNN(n_features, n_classes, activation, n_hidden1, n_layers, learning_rate, decay_steps, decay_rate)

        train_step, X, Y, cost, predicted, accuracy = graph['train_step'], graph['X'], graph['Y'], graph['cost'], graph['predicted'], graph['accuracy']

        saver = tf.train.Saver()
        model_dir = os.path.dirname(model_file)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        watch = WatchUtil()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            nth_train, nth_input, total_input = 0, 0, total_epoch * train.size

            log.info('learn...')
            log.info('n_hidden1 * n_layres: %s * %s' % (n_hidden1, n_layers))

            watch.start('learn')
            best_valid_cost = sys.float_info.max
            for epoch in range(1, total_epoch + 1):
                if best_valid_cost < early_stop_cost:
                    log.info('early stopped.')
                    break
                for step, (features_batch, labels_batch) in enumerate(train.next_batch(batch_size=batch_size), 1):
                    nth_train += 1
                    nth_input += features_batch.shape[0]
                    sess.run(train_step, feed_dict={X: features_batch, Y: labels_batch})

                    if step % valid_interval == 0:
                        percent = nth_input / total_input * 100
                        valid_cost = sess.run(cost, feed_dict={X: valid.features, Y: valid.labels})

                        if valid_cost < best_valid_cost:
                            best_valid_cost = valid_cost
                            saver.save(sess, model_file)
                            log.info('[epoch=%s][%.1f%%] valid_cost: %.8f model saved' % (epoch, percent, valid_cost))
                            if best_valid_cost < early_stop_cost:
                                log.info('valid_cost: %s, early_stop_cost: %s, early stopped.' % (valid_cost, early_stop_cost))
                                break
                        else:
                            log.info('[epoch=%s][%.1f%%] valid_cost: %.8f' % (epoch, percent, valid_cost))

            watch.stop('learn')
            log.info('learn OK.\n')

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
    def build_FFNN(cls, n_features, n_classes, activation, n_hiddens1, layers, learning_rate, decay_steps, decay_rate):
        log.info('\nbuild_FFNN')

        if len(cls.graph_nodes) == 0:
            log.info('create tensorflow graph...')
            log.info('n_features: %s' % n_features)
            log.info('n_classes: %s' % n_classes)
            log.info('n_hidden1 * n_layres: %s * %s' % (n_hidden1, n_layers))

            tf.set_random_seed(7942)  # for reproducibility

            x = tf.placeholder(tf.float32, [None, n_features], name='x')  # two characters
            y = tf.placeholder(tf.float32, [None, n_classes], name='Y')

            if layers == 4:
                log.info('layers: 4')
                n_hiddens2 = n_hiddens3 = n_hidden1
                w1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[n_features, n_hiddens1], name='w1')
                b1 = tf.get_variable(initializer=tf.constant(0., shape=[n_hiddens1]), name='b1')
                layer1 = activation(tf.matmul(x, w1) + b1, name='layer1')

                w2 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[n_hidden1, n_hiddens2], name='w2')
                b2 = tf.get_variable(initializer=tf.constant(0., shape=[n_hiddens2]), name='b2')
                layer2 = activation(tf.matmul(layer1, w2) + b2, name='layer2')

                w3 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[n_hiddens2, n_hiddens3], name='w3')
                b3 = tf.get_variable(initializer=tf.constant(0., shape=[n_hiddens3]), name='b3')
                layer3 = activation(tf.matmul(layer2, w3) + b3, name='layer3')

                w4 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[n_hiddens3, n_classes], name='w4')
                b4 = tf.get_variable(initializer=tf.constant(0., shape=[n_classes]), name='b4')
                y_hat = tf.add(tf.matmul(layer3, w4), b4, name='y_hat')
            else:
                log.info('layers: 2')
                w1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[n_features, n_hiddens1], name='w1')
                b1 = tf.get_variable(initializer=tf.constant(0., shape=[n_hiddens1]), name='b1')
                layer1 = activation(tf.matmul(x, w1) + b1, name='layer1')

                w2 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[n_hiddens1, n_classes], name='w2')
                b2 = tf.get_variable(initializer=tf.constant(0., shape=[n_classes]), name='b2')
                y_hat = tf.add(tf.matmul(layer1, w2), b2, name='y_hat')

            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y), name='cost')

            global_step = tf.get_variable(initializer=tf.constant(0), dtype=tf.int32, trainable=False, name='global_step')
            current_learning_rate = tf.train.exponential_decay(learning_rate, global_step=global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True, name='current_learning_rate')
            train_step = tf.train.AdamOptimizer(learning_rate=current_learning_rate).minimize(cost, name='train_step')

            predicted = tf.cast(y_hat > 0, dtype=tf.float32, name='predicted')  # -inf <= y_hat <= inf
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32), name='accuracy')

            log.info('create tensorflow graph OK.\n')
            cls.graph_nodes = {'predicted': predicted, 'accuracy': accuracy, 'X': x, 'Y': y, 'train_step': train_step, 'cost': cost}
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

    @classmethod
    def test(cls, test_sentences_file, n_features, n_classes, activation, n_hidden1, n_layers, learning_rate, decay_steps, decay_rate, watch):
        log.info('chek result...')
        watch.start('read sentences')

        sentences = ['아버지가 방에 들어 가신다.', '가는 말이 고와야 오는 말이 곱다.']
        if same_train_valid_data:
            test_sentences_file = train_sentences_file
        # else: # TODO:
        #     test_sentences_file = valid_sentences_file

        log.info('test_sentences_file: %s' % test_sentences_file)
        with gzip.open(test_sentences_file, 'rt') as f:
            for i, line in enumerate(f, 1):
                if len(sentences) >= n_test_sentences:
                    break

                s = line.strip()
                if s.count(' ') > 0:  # sentence must have one or more space.
                    sentences.append(s)
        log.info('len(sentences): %s' % NumUtil.comma_str(len(sentences)))
        watch.stop('read sentences')

        graph = WordSpacing.build_FFNN(n_features, n_classes, activation, n_hidden1, n_layers, learning_rate, decay_steps, decay_rate)
        X, Y, predicted, accuracy = graph['X'], graph['Y'], graph['predicted'], graph['accuracy']

        accuracies, sims = [], []
        saver = tf.train.Saver()

        watch.start('run tensorflow')
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
        with tf.Session(config=config) as sess:
            try:
                try:
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess, model_file)
                except:
                    log.error('restore failed. model_file: %s' % model_file)

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
        log.info('secs/sentence: %.4f' % (watch.elapsed('run tensorflow') / len(sentences)))
        log.info('')
        # noinspection PyStringFormat
        log.info('same_train_valid_data: %s, test_sim: %d%%, n_train_sentences: %s, elapsed: %s, total_epochs: %d, batch_size: %s, '
                 'activation: %s, n_hidden1 * layers: %d * %d: , learning_rate: %.6f, batches_in_epoch: %s, decay_steps: %s, decay_rate: %.2f, early_stop_cost: %.8f' % (
                     same_train_valid_data, np.mean(sims) * 100, NumUtil.comma_str(n_train_sentences), watch.elapsed_string(), total_epochs, NumUtil.comma_str(batch_size),
                     activation.__name__, n_hidden1, n_layers, learning_rate, NumUtil.comma_str(batches_in_epoch), NumUtil.comma_str(decay_steps), decay_rate, early_stop_cost))
        log.info(watch.summary())
        pass


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)  # ignore tensorflow info

    watch = WatchUtil()
    watch.start()

    train_sentences_file = WIKIPEDIA_TRAIN_SENTENCES_FILE
    valid_sentences_file = WIKIPEDIA_VALID_SENTENCES_FILE
    test_sentences_file = WIKIPEDIA_TEST_SENTENCES_FILE

    log.info('train_sentences_file: %s' % train_sentences_file)
    log.info('valid_sentences_file: %s' % valid_sentences_file)
    log.info('test_sentences_file: %s' % test_sentences_file)

    characters_file = WIKIPEDIA_CHARACTERS_FILE
    log.info('characters_file: %s' % characters_file)
    try:
        if len(sys.argv) == 4:
            n_train_sentences = int(sys.argv[1])
            ngram = int(sys.argv[2])
        else:
            # n_train_sentences, ngram = 1000000, 4
            n_train_sentences, ngram = 1000, 4  # FIXME: TEST

        left_gram = right_gram = math.ceil(ngram / 2)

        n_valid_sentences, n_test_sentences = 100, 100
        log.info('n_train_sentences: %s' % NumUtil.comma_str(n_train_sentences))
        log.info('n_valid_sentences: %s' % NumUtil.comma_str(n_valid_sentences))
        log.info('n_test_sentences: %s' % NumUtil.comma_str(n_test_sentences))
        log.info('ngram: %s = left_gram: %s + right_gram: %s' % (ngram, left_gram, right_gram))

        total_sentences = FileUtil.count_lines(WIKIPEDIA_SENTENCES_FILE)

        features_vector = CharOneHotVector(DataFileUtil.read_list(characters_file))
        labels_vector = CharOneHotVector([0, 1])  # 붙여쓰기=0, 띄어쓰기=1
        log.info('features_vector: %s' % features_vector)
        log.info('labels_vector: %s' % labels_vector)

        n_features = len(features_vector) * ngram  # number of features
        n_classes = 1  # labes= 0 or 1 (붙여쓰기=0, 띄어쓰기=1)
        log.info('n_features: %s' % n_features)
        log.info('n_classes: %s' % n_classes)

        same_train_valid_data = True if n_train_sentences <= 100 else False
        do_train = True
        do_test = True
        batch_size = 8000  # mini batch size
        total_epochs = max(3, math.ceil(20 / math.log10(n_train_sentences)))

        n_layers, n_hidden1, learning_rate = 2, 200, 1e-2
        # n_layers, n_hidden1, learning_rate = 4, 200, 1e-3

        activation = tf.nn.elu
        # activation = tf.tanh
        decay_epochs = 1
        decay_rate = 0.99
        early_stop_cost = 1e-2
        # same_train_valid_data: True, test_sim: 97%, n_train_sentences: 100, elapsed: 00:02:54, total_epochs: 10, batch_size: 8,000, activation: tanh, n_hidden:200 * n_hidden1: , learning_rate: 0.010000, batches_in_epoch: 1, decay_steps: 1, decay_rate: 0.96, early_stop_cost: 0.01000000
        # same_train_valid_data: True, test_sim: 97%, n_train_sentences: 100, elapsed: 00:02:53, total_epochs: 10, batch_size: 8,000, activation: elu, n_hidden:200 * n_hidden1: , learning_rate: 0.010000, batches_in_epoch: 1, decay_steps: 1, decay_rate: 0.96, early_stop_cost: 0.01000000
        # same_train_valid_data: True, test_sim: 81%, n_train_sentences: 10,000, total_epochs: 3, activation: tf.tanh, n_hidden1: 200 * 4 layers, learning_rate: 1e-3, decay_epochs: 1, decay_rate: 0.99, early_stop_cost: 1e-2

        # same_train_valid_data: False, test_sim: 62%, n_train_sentences: 1,000, elapsed: 00:13:36, total_epochs: 7, batch_size: 8,000, activation: elu, n_hidden1 * layers: 200 * 2: , learning_rate: 0.010000, batches_in_epoch: 7, decay_steps: 7, decay_rate: 0.99, early_stop_cost: 0.01000000
        # same_train_valid_data: False, test_sim: 67%, n_train_sentences: 10,000, elapsed: 01:34:59, total_epochs: 5, batch_size: 8,000, activation: elu, n_hidden1 * layers: 200 * 2: , learning_rate: 0.010000, batches_in_epoch: 64, decay_steps: 64, decay_rate: 0.99, early_stop_cost: 0.01000000

        valid_interval = 1
        log.info('')
        log.info('n_hidden1 * n_layres: %s * %s' % (n_hidden1, n_layers))
        log.info('total_epoch: %s' % total_epochs)
        log.info('batch_size: %s' % batch_size)
        log.info('')
        log.info('activation: %s' % activation.__name__)
        log.info('n_hidden1: %s' % n_hidden1)
        log.info('learning_rate: %s' % learning_rate)
        log.info('decay_rate: %.2f' % decay_rate)
        log.info('early_stop_cost: %s' % early_stop_cost)

        model_file = os.path.join(WORD_SPACING_MODEL_DIR, 'word_spacing_model.sentences_%s.ngram_%s.total_epoch_%s.activation_%s.n_hiddens_%d.n_layers_%d.learning_rate_%.4f/model' % (
            n_train_sentences, ngram, total_epochs, activation.__name__, n_hidden1, n_layers, learning_rate))  # .%s' % max_sentences
        log.info('model_file: %s' % model_file)

        if n_test_sentences > 100:
            log.info('sample testing...')
            test_set = ['예쁜 운동화', '즐거운 동화', '삼풍동 화재']
            for s in test_set:
                features, labels = WordSpacing.sentence2features_labels(s, left_gram=left_gram, right_gram=right_gram)
                log.info('%s -> %s' % (features, labels))
                log.info('in : "%s"' % s)
                log.info('out: "%s"' % WordSpacing.spacing(s.replace(' ', ''), labels))
            log.info('sample testing OK.\n')

        train_dataset, valid_dataset = WordSpacing.load_datasets(train_sentences_file, valid_sentences_file, n_train_sentences, n_valid_sentences, left_gram, right_gram, features_vector,
                                                                 labels_vector, same_train_valid_data=same_train_valid_data)
        batches_in_epoch = math.ceil(len(train_dataset) / batch_size)
        decay_steps = decay_epochs * batches_in_epoch
        log.info('decay_steps: %s' % decay_steps)

        if do_train or not os.path.exists(model_file + '.index'):
            WordSpacing.learning(total_epochs, train_dataset, valid_dataset, batch_size, left_gram, right_gram, model_file, features_vector, labels_vector, activation, n_hidden1, n_layers,
                                 learning_rate,
                                 decay_steps, decay_rate, early_stop_cost, valid_interval)
            if n_train_sentences > 100:
                SlackUtil.send_message('%s end (max_sentences=%s, left_gram=%s, right_gram=%.1f)' % (sys.argv[0], n_train_sentences, left_gram, right_gram))

        if do_test:
            WordSpacing.test(test_sentences_file, n_features, n_classes, activation, n_hidden1, n_layers, learning_rate, decay_steps, decay_rate, watch)
    except:
        log.error(traceback.format_exc())
