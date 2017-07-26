import math
import os
import traceback

import numpy as np
import tensorflow as tf

from bage_utils.date_util import DateUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor.config import DATA_DIR, TENSORBOARD_LOG_DIR, log, MODELS_DIR


def next_batch(filenames, data_size, batch_size=1, delim='\t', splits=2):
    """
    read big data can't be loaded in memory
    :param filenames: list of input file names
    :param data_size: max data size
    :param batch_size: batch size >= 1
    :param delim: delimiter of line
    :param splits: splits of line
    :return: batch data (features_batch, labels_batch)
    """
    _data_size = 0
    for filename in filenames:
        if _data_size > data_size:
            return
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
                    features_batch = np.array(_features, dtype=np.float32)
                    labels_batch = np.array(_labels, dtype=np.float32)
                    labels_batch = labels_batch.reshape(len(_labels), -1)
                    _features, _labels = [], []
                    yield features_batch, labels_batch
                if _data_size > data_size:
                    return


def next_batch_in_memory(filenames, data_size, batch_size=100, delim='\t', splits=2, shuffle=True):
    """
    read small data can be loaded in memory
    :param filenames: list of input file names
    :param data_size: max data size
    :param batch_size: batch size >= 1
    :param delim: delimiter of line
    :param splits: splits of line
    :param shuffle: shuffle data
    :return: batch data (features_batch, labels_batch)
    """
    _features, _labels = [], []  # read all data
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
                if len(_features) >= data_size:
                    return

    features = np.array(_features, dtype=np.float32)
    labels = np.array(_labels, dtype=np.float32)
    if shuffle:
        random_idx = np.random.permutation(len(_features))
        features, labels = features[random_idx], labels[random_idx]

    labels = labels.reshape(len(labels), -1)

    splits = len(features) // batch_size
    if len(features) % batch_size > 0:
        splits += 1
    batches = zip(np.array_split(features, splits), np.array_split(labels, splits))

    for features_batch, labels_batch in batches:
        yield features_batch, labels_batch


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
def create_graph(variable_scope_name, is_training=False, verbose=False):
    """
    create or reuse graph
    :param variable_scope_name: variable scope name
    :param is_training: is learning mode
    :param verbose: print graph nodes
    :return: tensorflow graph nodes
    """
    with tf.variable_scope(variable_scope_name) as variable_scope:  # for reusing graph
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        x = tf.placeholder(dtype=tf.float32, shape=[None, input_len], name='x')
        y = tf.placeholder(dtype=tf.float32, shape=[None, output_len], name='y')

        # W1 = tf.get_variable(dtype=tf.float32, shape=[input_len, output_len], initializer=tf.contrib.layers.xavier_initializer(), name='W1')
        W1 = tf.get_variable(dtype=tf.float32, shape=[input_len, output_len], initializer=tf.random_normal_initializer(), name='W1')
        b1 = tf.get_variable(dtype=tf.float32, initializer=tf.constant(0.0, shape=[output_len]), name='b1')

        y_hat = tf.add(tf.matmul(x, W1), b1, name='y_hat')
        cost = tf.reduce_mean(tf.square(y_hat - y), name='cost')
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(cost, name='train_step')

        tf.summary.histogram(values=W1, name='summary_W1')
        tf.summary.histogram(values=b1, name='summary_b1')
        tf.summary.scalar(tensor=cost, name='summary_cost')
        summary_all = tf.summary.merge_all()

        if verbose:  # print graph if the first of learning mode
            log.info('')
            log.info(x)
            log.info(W1)
            log.info(b1)
            log.info('')
            log.info(y)
            log.info(y_hat)
            log.info(cost)  # cost operation is valid? check y_hat's shape and y's shape

        variable_scope.reuse_variables()
    return learning_rate, x, y, W1, b1, y_hat, cost, train_step, summary_all


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)  # ignore tensorflow info

    train_file = os.path.join(DATA_DIR, 'add.train.tsv')
    valid_file = os.path.join(DATA_DIR, 'add.valid.tsv')
    test_file = os.path.join(DATA_DIR, 'add.test.tsv')

    valid_interval = 10
    save_model_each_epochs = False  # defualt False

    input_len = 2  # x1, x2
    output_len = 1  # y

    n_train, n_valid, n_test = 1000, 100, 20
    if not os.path.exists(train_file):
        create_data4add(train_file, n_train, digit_max=99)
    if not os.path.exists(valid_file):
        create_data4add(valid_file, n_valid, digit_max=99)
    if not os.path.exists(test_file):
        create_data4add(test_file, n_test, digit_max=99)

    for training_mode in [True, False]:  # training & testing
        for batch_size, total_epochs in zip([1, 10, 100], [6, 50, 120]):
            tf.reset_default_graph()  # Clears the default graph stack and resets the global default graph.
            log.info('')
            log.info('training_mode: %s, batch_size: %s, total_epochs: %s' % (training_mode, batch_size, total_epochs))

            model_name = os.path.basename(__file__).replace('.py', '')
            model_file = os.path.join(MODELS_DIR, '%s.n_train_%s.batch_size_%s.total_epochs_%s/model' % (model_name, n_train, batch_size, total_epochs))
            model_dir = os.path.dirname(model_file)
            log.info('model_name: %s' % model_name)
            log.info('model_file: %s' % model_file)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            variable_scope = '%s.%s.batch_size_%s.total_epochs_%s' % (model_name, DateUtil.current_yyyymmdd_hhmm(), batch_size, total_epochs)
            log.info('variable_scope: %s' % variable_scope)

            with tf.device('/gpu:0'):
                with tf.Graph().as_default():  # for reusing graph
                    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
                    with tf.Session(config=config) as sess:
                        checkpoint = tf.train.get_checkpoint_state(model_dir)
                        # if checkpoint:
                        #     log.debug('')
                        #     log.debug('checkpoint:')
                        #     log.debug(checkpoint)
                        #     log.debug('checkpoint.model_checkpoint_path: %s' % checkpoint.model_checkpoint_path)
                        is_training = True if training_mode or not checkpoint else False  # learning or testing

                        learning_rate, x, y, W1, b1, y_hat, cost, train_step, summary_all = create_graph(variable_scope, is_training=is_training)

                        sess.run(tf.global_variables_initializer())
                        saver = tf.train.Saver(max_to_keep=None)

                        if is_training:  # training
                            train_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/train', sess.graph)
                            valid_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/valid', sess.graph)

                            batch_count = math.ceil(n_train / batch_size)  # batch count for one epoch
                            try:
                                watch = WatchUtil()
                                watch.start()
                                for epoch in range(1, total_epochs + 1):
                                    nth_batch, min_valid_epoch, min_valid_cost = 0, 0, 1e10
                                    for _features_batch, _labels_batch in next_batch([train_file], data_size=n_train, batch_size=batch_size, delim='\t',
                                                                                     splits=3):
                                        nth_batch += 1
                                        _, _train_cost, _summary_all = sess.run([train_step, cost, summary_all],
                                                                                feed_dict={x: _features_batch, y: _labels_batch, learning_rate: 0.01})
                                        train_writer.add_summary(_summary_all, global_step=nth_batch)
                                        # print(_features_batch.shape, _labels_batch.shape)

                                        if nth_batch % valid_interval == 0:
                                            # noinspection PyAssignmentToLoopOrWithParameter
                                            for _features_batch, _labels_batch in next_batch([train_file], data_size=n_valid, batch_size=n_valid, delim='\t',
                                                                                             splits=3):
                                                _, _valid_cost, _summary_all = sess.run([train_step, cost, summary_all],
                                                                                        feed_dict={x: _features_batch, y: _labels_batch, learning_rate: 0.01})
                                                valid_writer.add_summary(_summary_all, global_step=nth_batch)
                                                if _valid_cost < min_valid_cost:
                                                    min_valid_cost = _valid_cost
                                                    min_valid_epoch = epoch

                                                # noinspection PyUnboundLocalVariable
                                                log.info('[epoch: %s, nth_batch: %s] train cost: %.4f valid cost: %.4f' % (epoch, nth_batch, _train_cost, _valid_cost))

                                    if save_model_each_epochs:
                                        saver.save(sess, model_file, global_step=epoch)
                                    if min_valid_epoch == epoch:  # save the lastest best model
                                        saver.save(sess, model_file)
                                log.info('')
                                # noinspection PyUnboundLocalVariable
                                log.info('"%s" train: min_valid_cost: %.4f, min_valid_epoch: %s,  %.2f secs (batch_size: %s, total_epochs: %s)' % (
                                    model_name, min_valid_cost, min_valid_epoch, watch.elapsed(),
                                    batch_size, total_epochs))
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
                                for _features_batch, _labels_batch in next_batch([test_file], data_size=n_test, batch_size=n_test, delim='\t', splits=3):
                                    _, _test_cost, _y_hat_batch, _W1, _b1 = sess.run([train_step, cost, y_hat, W1, b1],
                                                                                     feed_dict={x: _features_batch, y: _labels_batch, learning_rate: 0.01})

                                    log.info('')
                                    log.info('W1: %s' % ['%.4f' % i for i in _W1])
                                    log.info('b1: %.4f' % _b1)
                                    for (x1, x2), _y, _y_hat in zip(_features_batch, _labels_batch, _y_hat_batch):
                                        log.debug('%3d + %3d = %4d (y_hat: %4.1f)' % (x1, x2, _y, _y_hat))
                                    log.info('')
                                    log.info(
                                        '"%s" test: test_cost: %.4f, %.2f secs (batch_size: %s, total_epochs: %s)' % (
                                            model_name, _test_cost, watch.elapsed(),
                                            batch_size, total_epochs))
                                    log.info('')
                            except:
                                log.info(traceback.format_exc())
