import math
import os
import traceback

import numpy as np
import tensorflow as tf

from bage_utils.date_util import DateUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor.config import DATA_DIR, TENSORBOARD_LOG_DIR, log, MODELS_DIR


def input_pipeline(filenames, batch_size=1, delim='\t', splits=2, shuffle=False):
    """
    create graph nodes for streaming data input
    :param filenames: list of input file names
    :param batch_size: batch size >= 1
    :param delim: delimiter of line
    :param splits: splits of line
    :param shuffle: shuffle fileanames and datas
    :return: graph nodes (features_batch, labels_batch)
    """
    min_after_dequeue = max(100, batch_size * 2)  # batch_size
    capacity = min_after_dequeue + 3 * batch_size

    filename_queue = tf.train.string_input_producer(filenames, name='filename_queue', shuffle=shuffle)
    reader = tf.TextLineReader(skip_header_lines=None, name='reader')
    _key, value = reader.read(filename_queue)
    tokens = tf.decode_csv(value, field_delim=delim, record_defaults=[[0.] for _ in range(splits)], name='decode_csv')
    # log.debug('%s %s %s' % (x1, x2, y))
    feature = tf.reshape([tokens[:-1]], shape=[splits - 1])
    label = tf.reshape([tokens[-1]], shape=[1])
    # log.debug(feature)

    if shuffle:
        features_batch, labels_batch = tf.train.shuffle_batch([feature, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        # features_valid, labels_valid = tf.train.shuffle_batch([feature, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)  # FIXME: valid input
    else:
        features_batch, labels_batch = tf.train.batch([feature, label], batch_size=batch_size, capacity=capacity)

    # return tf.identity(features_batch, name='x'), tf.identity(labels_batch, name='y') # FIXME: naming
    return features_batch, labels_batch


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


def create_graph(variable_scope_name, is_learning=False, verbose=False):
    """
    create or reuse graph
    :param variable_scope_name: variable scope name
    :param is_learning: is learning mode
    :param verbose: print graph nodes
    :return: tensorflow graph nodes
    """
    with tf.variable_scope(variable_scope_name) as variable_scope:  # for reusing graph
        if is_learning:
            x, y = input_pipeline([train_file], batch_size=batch_size, shuffle=True, delim='\t', splits=3)
        else:
            # x, y = input_pipeline([test_file], batch_size=n_test, shuffle=True, delim='\t', splits=3)
            x = tf.placeholder(dtype=tf.float32, shape=[None, input_len], name='x')
            y = tf.placeholder(dtype=tf.float32, shape=[None, output_len], name='y')

        # W1 = tf.get_variable(dtype=tf.float32, shape=[input_len, output_len], initializer=tf.contrib.layers.xavier_initializer(), name='W1')
        W1 = tf.get_variable(dtype=tf.float32, shape=[input_len, output_len], initializer=tf.random_normal_initializer(), name='W1')
        b1 = tf.get_variable(dtype=tf.float32, initializer=tf.constant(0.0, shape=[output_len]), name='b1')

        y_hat = tf.add(tf.matmul(x, W1), b1, name='y_hat')
        cost = tf.reduce_mean(tf.square(y_hat - y), name='cost')
        train_step = tf.train.AdamOptimizer(learning_rate=0.01, name='optimizer').minimize(cost, name='train_step')

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
    return x, y, W1, b1, y_hat, cost, train_step, summary_all


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

    for learning_mode in [True, False]:  # learning & training
        for batch_size, total_epochs in zip([1, 10, 100], [6, 18, 20]):
            tf.reset_default_graph()  # Clears the default graph stack and resets the global default graph.
            log.info('')
            log.info('learning_mode: %s, batch_size: %s, total_epochs: %s' % (learning_mode, batch_size, total_epochs))

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
                    checkpoint = tf.train.get_checkpoint_state(model_dir)
                    # if checkpoint:
                    #     log.debug('')
                    #     log.debug('checkpoint:')
                    #     log.debug(checkpoint)
                    #     log.debug('checkpoint.model_checkpoint_path: %s' % checkpoint.model_checkpoint_path)
                    is_learning = True if learning_mode or not checkpoint else False  # learning or testing

                    x, y, W1, b1, y_hat, cost, train_step, summary_all = create_graph(variable_scope, is_learning=is_learning)

                    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
                    with tf.Session(config=config) as sess:
                        sess.run(tf.global_variables_initializer())
                        saver = tf.train.Saver(max_to_keep=None)

                        if is_learning:  # learning
                            train_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/train', sess.graph)
                            valid_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/valid', sess.graph)

                            valid_features_batch, valid_labels_batch = input_pipeline([valid_file], batch_size=n_valid, splits=3)

                            coordinator = tf.train.Coordinator()  # coordinator for enqueue threads
                            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)  # start filename queue
                            batch_count = math.ceil(n_train / batch_size)  # batch count for one epoch
                            try:
                                watch = WatchUtil()
                                watch.start()
                                nth_batch, min_valid_epoch, min_valid_cost = 0, 0, 1e10
                                for epoch in range(1, total_epochs + 1):
                                    for i in range(1, batch_count + 1):
                                        if coordinator.should_stop():
                                            break

                                        nth_batch += 1
                                        _, _train_cost, _summary_all = sess.run([train_step, cost, summary_all])  # no feed_dict
                                        train_writer.add_summary(_summary_all, global_step=nth_batch)

                                        # if nth_batch % valid_interval == 0:  # FIXME:
                                        #     # noinspection PyAssignmentToLoopOrWithParameter
                                        #     _features_batch, _labels_batch = sess.run([valid_features_batch, valid_labels_batch])
                                        #     _test_cost, _y_hat_batch, _W1, _b1 = sess.run([cost, y_hat, W1, b1],
                                        #                                                   feed_dict={x: _features_batch, y: _labels_batch})
                                        #     valid_writer.add_summary(_summary_all, global_step=nth_batch)
                                    if _train_cost < min_valid_cost:
                                        min_valid_cost = _train_cost
                                        min_valid_epoch = epoch
                                    log.info('[epoch: %s, nth_batch: %s] train cost: %.8f' % (epoch, nth_batch, _train_cost))
                                    if save_model_each_epochs:
                                        saver.save(sess, model_file, global_step=epoch)
                                    if min_valid_epoch == epoch:  # save the lastest best model
                                        saver.save(sess, model_file)

                                log.info('')
                                log.info('"%s" train: min_valid_cost: %.4f, min_valid_epoch: %s,  %.2f secs (batch_size: %s, total_epochs: %s)' % (
                                    model_name, min_valid_cost, min_valid_epoch, watch.elapsed(),
                                    batch_size, total_epochs))
                                log.info('')
                            except:
                                log.info(traceback.format_exc())
                            finally:
                                coordinator.request_stop()
                                coordinator.join(threads)  # Wait for threads to finish.
                        else:  # testing
                            log.info('')
                            log.info('model loaded... %s' % model_file)
                            saver.restore(sess, model_file)
                            log.info('model loaded OK. %s' % model_file)

                            test_features_batch, test_labels_batch = input_pipeline([test_file], batch_size=n_test, splits=3)

                            coordinator = tf.train.Coordinator()
                            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
                            try:
                                watch = WatchUtil()
                                watch.start()
                                _features_batch, _labels_batch = sess.run([test_features_batch, test_labels_batch])
                                _test_cost, _y_hat_batch, _W1, _b1 = sess.run([cost, y_hat, W1, b1], feed_dict={x: _features_batch, y: _labels_batch})

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
                            finally:
                                coordinator.request_stop()
                                coordinator.join(threads)  # Wait for threads to finish.
