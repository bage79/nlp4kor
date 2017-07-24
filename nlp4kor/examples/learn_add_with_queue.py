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
    else:
        features_batch, labels_batch = tf.train.batch([feature, label], batch_size=batch_size, capacity=capacity)

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


def create_graph(variable_scope, reuse=False, is_learning=False, verbose=False):
    """
    create or reuse graph
    :param variable_scope:
    :param reuse: reuse graph whenever learning or testing
    :param is_learning: is learning mode
    :param verbose: print graph nodes
    :return: tensorflow graph nodes
    """
    with tf.variable_scope(variable_scope, reuse=reuse):  # for reusing graph
        if is_learning:
            x, y = input_pipeline([train_file], batch_size=batch_size, shuffle=True, delim='\t', splits=3)
        else:
            x, y = input_pipeline([test_file], batch_size=batch_size, shuffle=True, delim='\t', splits=3)

        # W1 = tf.get_variable(dtype=tf.float32, shape=[input_len, output_len], initializer=tf.contrib.layers.xavier_initializer(), name='W1')
        W1 = tf.get_variable(dtype=tf.float32, shape=[input_len, output_len], initializer=tf.random_normal_initializer(), name='W1')
        b1 = tf.get_variable(dtype=tf.float32, initializer=tf.constant(0.0, shape=[output_len]), name='b1')

        y_hat = tf.add(tf.matmul(x, W1), b1, name='y_hat')
        cost = tf.reduce_mean(tf.square(y_hat - y), name='cost')
        train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        scope_postfix = '%s.batch_size_%s.total_epochs_%s' % (DateUtil.current_yyyymmdd_hhmm(), batch_size, total_epochs)
        log.debug('scope_postfix: %s' % scope_postfix)
        tf.summary.histogram(values=W1, name='W1/' + scope_postfix)
        tf.summary.histogram(values=b1, name='b1/' + scope_postfix)
        tf.summary.scalar(tensor=cost, name='cost/' + scope_postfix)
        summary_merge = tf.summary.merge_all()

        if verbose:  # print graph if the first of learning mode
            log.info('')
            log.info(x)
            log.info(W1)
            log.info(b1)
            log.info('')
            log.info(y)
            log.info(y_hat)
            log.info(cost)  # cost operation is valid? check y_hat's shape and y's shape
    return x, y, W1, b1, y_hat, cost, train_step, summary_merge


if __name__ == '__main__':
    learning_mode = True  # TODO: input from arguments
    train_file = os.path.join(DATA_DIR, 'add.train.tsv')
    test_file = os.path.join(DATA_DIR, 'add.test.tsv')

    input_len = 2  # x1, x2
    output_len = 1  # y

    n_train, n_test = 1000, 10
    total_epochs = 10

    if not os.path.exists(train_file):
        create_data4add(train_file, n_train, digit_max=99)
    if not os.path.exists(test_file):
        create_data4add(test_file, n_test, digit_max=99)

    for batch_size in [1, 10, 100]:
        log.info('batch_size: %s' % batch_size)

        model_name = os.path.basename(__file__).replace('.py', '')
        log.info('model_name: %s' % model_name)
        model_file = os.path.join(MODELS_DIR, '%s.n_train=%s.batch_size=%s/model' % (model_name, n_train, batch_size))
        model_dir = os.path.dirname(model_file)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        log.info('model_file: %s' % model_file)

        variable_scope = os.path.basename(__file__)
        reuse = False
        with tf.device('/gpu:0'):
            with tf.Graph().as_default():  # for reusing graph
                checkpoint = tf.train.get_checkpoint_state(model_dir)
                if checkpoint:
                    log.debug('')
                    log.debug('checkpoint:')
                    log.debug(checkpoint)
                    log.debug('checkpoint.model_checkpoint_path: %s' % checkpoint.model_checkpoint_path)

                is_learning = True if learning_mode or not checkpoint else False  # learning or testing

                x, y, W1, b1, y_hat, cost, train_step, summary_merge = create_graph(variable_scope, reuse=reuse, is_learning=is_learning)
                reuse = True

                min_epoch, min_cost = 0, 1e10
                nth_batch = 0

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True  # do not use entire memory for this session
                with tf.Session(config=config) as sess:
                    sess.run(tf.global_variables_initializer())
                    saver = tf.train.Saver(max_to_keep=100)

                    if is_learning:  # learning
                        writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR, sess.graph)

                        coordinator = tf.train.Coordinator()  # coordinator for enqueue threads
                        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)  # start filename queue

                        batch_count = math.ceil(n_train / batch_size)  # batch count for one epoch
                        try:
                            watch = WatchUtil()
                            watch.start()
                            for epoch in range(1, total_epochs + 1):
                                for i in range(1, batch_count + 1):
                                    if coordinator.should_stop():
                                        break
                                    nth_batch += 1
                                    _, _train_cost, _summary_merge = sess.run([train_step, cost, summary_merge])  # no feed_dict
                                    writer.add_summary(_summary_merge, global_step=nth_batch)

                                if _train_cost < min_cost:
                                    min_cost = _train_cost
                                    min_epoch = epoch
                                log.info('[epoch: %s, nth_batch: %s] train cost: %.8f' % (epoch, nth_batch, _train_cost))
                                # saver.save(sess, model_file, global_step=epoch)  # no need, redundant models
                                if min_epoch == epoch:  # save lastest best model
                                    saver.save(sess, model_file)
                            log.info('[min_epoch: %s] min_cost: %.8f' % (min_epoch, min_cost))
                            log.info('')
                            log.info('train with %s: %.2f secs (batch_size: %s)' % (model_name, watch.elapsed(), batch_size))
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
                            _, _test_cost, _y_hat_batch, _W1, _b1 = sess.run([train_step, cost, y_hat, W1, b1],
                                                                             feed_dict={x: _features_batch, y: _labels_batch})

                            log.info('')
                            log.info('test cost: %.4f' % _test_cost)
                            log.info('W1: %s' % ['%.4f' % i for i in _W1])
                            log.info('b1: %.4f' % _b1)
                            for (x1, x2), _y, _y_hat in zip(_features_batch, _labels_batch, _y_hat_batch):
                                log.debug('%3d + %3d = %4d (y_hat: %4.1f)' % (x1, x2, _y, _y_hat))
                            log.info('')
                            log.info('test with %s: %.2f secs (batch_size: %s)' % (model_name, watch.elapsed(), batch_size))
                            log.info('')
                        except:
                            log.info(traceback.format_exc())
                        finally:
                            coordinator.request_stop()
                            coordinator.join(threads)  # Wait for threads to finish.
