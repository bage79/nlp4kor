import math
import os
import traceback

import numpy as np
import tensorflow as tf
import time

from bage_utils.date_util import DateUtil
from bage_utils.timer_util import TimerUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor.config import DATA_DIR, TENSORBOARD_LOG_DIR, log, MODELS_DIR


def next_batch(filenames, data_size, batch_size=1, delim='\t', splits=2, shuffle=False):
    """
    read big data can't be loaded in memory
    :param filenames: list of input file names
    :param data_size: max data size
    :param batch_size: batch size >= 1
    :param delim: delimiter of line
    :param splits: splits of line
    :return: batch data (x_batch, y_batch)
    """
    _data_size = 0
    for filename in filenames:
        with open(filename) as f:
            _features, _labels = [], []
            for line in f.readlines():
                _data_size += 1
                line = line.strip()
                tokens = line.split(delim)
                if len(tokens) != splits:  # invalid line
                    continue
                _features.append([float(t) for t in tokens[:-1]])
                _labels.append(float(tokens[-1]))

                if len(_features) >= batch_size:
                    x_batch = np.array(_features, dtype=np.float32)
                    y_batch = np.array(_labels, dtype=np.float32)
                    y_batch = y_batch.reshape(len(_labels), -1)
                    _features, _labels = [], []
                    yield x_batch, y_batch
                    if _data_size >= data_size:  # restart this function
                        return


def next_batch_in_memory(filenames, data_size, batch_size=1, delim='\t', splits=2, shuffle=False):
    """
    read small data can be loaded in memory
    :param filenames: list of input file names
    :param data_size: max data size
    :param batch_size: batch size >= 1
    :param delim: delimiter of line
    :param splits: splits of line
    :param shuffle: shuffle data
    :return: batch data (x_batch, y_batch)
    """
    if not hasattr(next_batch_in_memory, 'batches') or not hasattr(next_batch_in_memory, 'batch_size') or next_batch_in_memory.batch_size != batch_size:
        next_batch_in_memory.batches = []
        next_batch_in_memory.batch_size = batch_size

        _features, _labels = [], []  # read all data
        watch = WatchUtil()
        watch.start()
        for filename in filenames:
            if len(_features) > data_size:
                return
            with open(filename) as f:
                for line in f.readlines():
                    line = line.strip()
                    tokens = line.split(delim)
                    if len(tokens) != splits:  # invalid line
                        continue
                    _features.append([float(t) for t in tokens[:-1]])
                    _labels.append(float(tokens[-1]))

        features = np.array(_features, dtype=np.float32)
        labels = np.array(_labels, dtype=np.float32)
        if shuffle:
            random_idx = np.random.permutation(len(_features))
            features, labels = features[random_idx], labels[random_idx]

        labels = labels.reshape(len(labels), -1)

        splits = len(features) // batch_size
        if len(features) % batch_size > 0:
            splits += 1
        next_batch_in_memory.batches = list(zip(np.array_split(features, splits), np.array_split(labels, splits)))

    for x_batch, y_batch in next_batch_in_memory.batches:
        yield x_batch, y_batch


def create_data4add(data_file, n_data, digit_max=99):
    """
    create data of x1 + x2 = y
    :param data_file: output file path
    :param n_data: total data size
    :param digit_max: 0 < x1, x2 < digit_max
    :return: None
    """
    input_len = 2  # x1, x2
    train_x = np.random.randint(digit_max + 1, size=input_len * n_data).reshape(-1, input_len)
    train_y = np.array([a + b for a, b in train_x])
    # log.info(train_x.shape)
    # log.info(train_y.shape)

    with open(data_file, 'wt') as f:
        for (x1, x2), y in zip(train_x, train_y):
            # log.info('%s + %s = %s' % (x1, x2, y))
            f.write('%s\t%s\t%s\n' % (x1, x2, y))


# noinspection PyUnusedLocal
def create_graph(scope_name, input_len=2, output_len=1, verbose=False):
    """
    create or reuse graph
    :param output_len: x1, x2
    :param input_len: y
    :param scope_name:
    :param verbose: print graph nodes
    :return: tensorflow graph nodes
    """
    with tf.variable_scope('common') as variable_scope:  # for reusing graph
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        x = tf.placeholder(dtype=tf.float32, shape=[None, input_len], name='x')
        y = tf.placeholder(dtype=tf.float32, shape=[None, output_len], name='y')

        W1 = tf.get_variable(dtype=tf.float32, shape=[input_len, output_len], initializer=tf.random_normal_initializer(), name='W1')
        b1 = tf.get_variable(dtype=tf.float32, initializer=tf.constant(0.0, shape=[output_len]), name='b1')

        y_hat = tf.add(tf.matmul(x, W1), b1, name='y_hat')
        cost = tf.reduce_mean(tf.square(y_hat - y), name='cost')
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(cost, name='train_step')

    with tf.variable_scope(scope_name, reuse=None) as scope:
        _W1 = tf.summary.histogram(values=W1, name='_W1')
        _b1 = tf.summary.histogram(values=b1, name='_b1')
        _cost = tf.summary.scalar(tensor=cost, name='_cost')
        summary = tf.summary.merge([_W1, _b1, _cost])

    if verbose:
        log.info('')
        log.info(x)
        log.info(W1)
        log.info(b1)
        log.info('')
        log.info(y)
        log.info(y_hat)
        log.info(cost)
    return x, y, learning_rate, W1, b1, y_hat, cost, train_step, summary


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)  # ignore tensorflow info

    train_file = os.path.join(DATA_DIR, 'add.train.tsv')
    valid_file = os.path.join(DATA_DIR, 'add.valid.tsv')
    test_file = os.path.join(DATA_DIR, 'add.test.tsv')

    total_train_time = 5
    valid_check_interval = 0.5
    save_model_each_epochs = False  # defualt False

    input_len = 2  # x1, x2
    output_len = 1  # y
    _learning_rate = 0.01

    is_big_data = True  # next_batch or next_batch_in_memory

    n_train, n_valid, n_test = 1000, 100, 10
    if not os.path.exists(train_file):
        create_data4add(train_file, n_train, digit_max=99)
    if not os.path.exists(valid_file):
        create_data4add(valid_file, n_valid, digit_max=99)
    if not os.path.exists(test_file):
        create_data4add(test_file, n_test, digit_max=99)

    for training_mode in [True, False]:  # training & testing
        for batch_size in [1, 10, 100]:
            tf.reset_default_graph()  # Clears the default graph stack and resets the global default graph.
            log.info('')
            log.info('training_mode: %s, batch_size: %s, total_train_time: %s secs' % (training_mode, batch_size, total_train_time))

            model_name = os.path.basename(__file__).replace('.py', '')
            model_file = os.path.join(MODELS_DIR, '%s.n_train_%s.batch_size_%s.total_train_time_%s/model' % (model_name, n_train, batch_size, total_train_time))
            model_dir = os.path.dirname(model_file)
            log.info('model_name: %s' % model_name)
            log.info('model_file: %s' % model_file)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            scope_name = '%s.%s.batch_size_%s.total_train_time_%s' % (model_name, DateUtil.current_yyyymmdd_hhmm(), batch_size, total_train_time)
            log.info('scope_name: %s' % scope_name)

            with tf.device('/gpu:0'):
                with tf.Graph().as_default():  # for reusing graph
                    checkpoint = tf.train.get_checkpoint_state(model_dir)
                    is_training = True if training_mode or not checkpoint else False  # learning or testing

                    x, y, learning_rate, W1, b1, y_hat, cost, train_step, summary = create_graph(scope_name, input_len=input_len, output_len=output_len, verbose=False)

                    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
                    with tf.Session(config=config) as sess:
                        sess.run(tf.global_variables_initializer())
                        saver = tf.train.Saver(max_to_keep=None)

                        if is_training:  # training
                            train_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/train', sess.graph)
                            valid_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/valid', sess.graph)

                            batch_count = math.ceil(n_train / batch_size)  # batch count for one epoch
                            try:
                                watch = WatchUtil()
                                stop_timer = TimerUtil(interval_secs=total_train_time)
                                valid_timer = TimerUtil(interval_secs=valid_check_interval)
                                watch.start()
                                stop_timer.start()
                                valid_timer.start()

                                nth_batch, min_valid_epoch, min_valid_cost = 0, 0, 1e10
                                epoch, running = 0, True
                                batch_func = next_batch if is_big_data else next_batch_in_memory
                                while running:
                                    epoch += 1
                                    for _x_batch, _y_batch in batch_func([train_file], data_size=n_train, batch_size=batch_size, delim='\t',
                                                                         splits=3, shuffle=False):
                                        if stop_timer.is_over():
                                            running = False
                                            break

                                        if len(_x_batch) != batch_size:
                                            log.error('len(_x_batch): %s' % _x_batch.shape)

                                        # print(_x_batch[:2], _x_batch.shape)
                                        # print()
                                        # time.sleep(1)

                                        nth_batch += 1
                                        _, _train_cost, _summary = sess.run([train_step, cost, summary],
                                                                            feed_dict={x: _x_batch, y: _y_batch, learning_rate: _learning_rate})
                                        train_writer.add_summary(_summary, global_step=nth_batch)
                                        # print(_x_batch.shape, _y_batch.shape)

                                        if valid_timer.is_over():
                                            # noinspection PyAssignmentToLoopOrWithParameter
                                            for _x_batch, _y_batch in next_batch([valid_file], data_size=n_valid, batch_size=n_valid, delim='\t',
                                                                                 splits=3):
                                                _, _valid_cost, _summary = sess.run([train_step, cost, summary],
                                                                                    feed_dict={x: _x_batch, y: _y_batch,
                                                                                               learning_rate: _learning_rate})
                                                valid_writer.add_summary(_summary, global_step=nth_batch)
                                                if _valid_cost < min_valid_cost:
                                                    min_valid_cost = _valid_cost
                                                    min_valid_epoch = epoch
                                                # noinspection PyUnboundLocalVariable
                                                log.info('[epoch: %s, nth_batch: %s] train cost: %.8f valid cost: %.8f' % (
                                                    epoch, nth_batch, _train_cost, _valid_cost))
                                                if min_valid_epoch == epoch:  # save the lastest best model
                                                    saver.save(sess, model_file)

                                    if save_model_each_epochs:
                                        saver.save(sess, model_file, global_step=epoch)
                                log.info('')
                                log.info(
                                    '"%s" train: min_valid_cost: %.8f, min_valid_epoch: %s,  %.2f secs (batch_size: %s, total_epochs: %s, total_train_time: %s secs)' % (
                                        model_name, min_valid_cost, min_valid_epoch, watch.elapsed(),
                                        batch_size, epoch, total_train_time))
                                log.info('')
                            except:
                                log.info(traceback.format_exc())
                        else:  # testing
                            log.info('')
                            log.info('model loaded... %s' % model_file)
                            saver.restore(sess, model_file)
                            log.info('model loaded OK. %s' % model_file)

                            try:
                                watch = WatchUtil()
                                watch.start()
                                for _x_batch, _y_batch in next_batch([test_file], data_size=n_test, batch_size=n_test, delim='\t', splits=3):
                                    _, _test_cost, _y_hat_batch, _W1, _b1 = sess.run([train_step, cost, y_hat, W1, b1],
                                                                                     feed_dict={x: _x_batch, y: _y_batch,
                                                                                                learning_rate: _learning_rate})

                                    log.info('')
                                    log.info('W1: %s' % ['%.8f' % i for i in _W1])
                                    log.info('b1: %.8f' % _b1)
                                    for (x1, x2), _y, _y_hat in zip(_x_batch, _y_batch, _y_hat_batch):
                                        log.debug('%3d + %3d = %4d (y_hat: %4.1f)' % (x1, x2, _y, _y_hat))
                                    log.info('')
                                    log.info(
                                        '"%s" test: test_cost: %.8f, %.2f secs (batch_size: %s)' % (model_name, _test_cost, watch.elapsed(), batch_size))
                                    log.info('')
                            except:
                                log.info(traceback.format_exc())
