import math
import os
import traceback

import numpy as np
import tensorflow as tf

from bage_utils.date_util import DateUtil
from bage_utils.num_util import NumUtil
from bage_utils.timer_util import TimerUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor.config import SAMPLE_DATA_DIR, TENSORBOARD_LOG_DIR, log, SAMPLE_MODELS_DIR


def input_pipeline(filenames, batch_size=1, delim='\t', splits=2, shuffle=False):
    """
    create graph nodes for streaming data input
    :param filenames: list of input file names
    :param batch_size: batch size >= 1
    :param delim: delimiter of line
    :param splits: splits of line
    :param shuffle: shuffle fileanames and datas
    :return: graph nodes (x_batch, y_batch)
    """
    min_after_dequeue = max(1000, batch_size * 10)  # batch_size
    capacity = min_after_dequeue + 3 * batch_size

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, name='filename_queue')
    reader = tf.TextLineReader(skip_header_lines=None, name='reader')
    _key, value = reader.read(filename_queue)
    tokens = tf.decode_csv(value, field_delim=delim, record_defaults=[[0.] for _ in range(splits)], name='decode_csv')
    feature = tf.reshape([tokens[:-1]], shape=[splits - 1], name='x_reshape')
    label = tf.reshape([tokens[-1]], shape=[1], name='y_reshape')

    if shuffle:
        x_batch, y_batch = tf.train.shuffle_batch([feature, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    else:
        x_batch, y_batch = tf.train.batch([feature, label], batch_size=batch_size, capacity=capacity)

    return tf.identity(x_batch, name='x'), tf.identity(y_batch, name='y')


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

    with open(data_file, 'wt') as f:
        for (x1, x2), y in zip(train_x, train_y):
            f.write('%s\t%s\t%s\n' % (x1, x2, y))


# noinspection PyUnusedLocal
def create_graph(model_name, scope_name, first_pipeline, second_pipeline, verbose=False):
    """
    create or reuse graph
    :param model_name:
    :param first_pipeline:
    :param second_pipeline:
    :param scope_name:
    :param verbose: print graph nodes
    :return: tensorflow graph nodes
    """
    with tf.variable_scope('common'):  # for reusing graph
        use_first_pipeline = tf.placeholder(dtype=bool)
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        W1 = tf.get_variable(dtype=tf.float32, shape=[input_len, output_len], initializer=tf.random_normal_initializer(), name='W1')
        b1 = tf.get_variable(dtype=tf.float32, initializer=tf.constant(0.0, shape=[output_len]), name='b1')

        x, y = tf.cond(use_first_pipeline, lambda: first_pipeline, lambda: second_pipeline)

        y_hat = tf.add(tf.matmul(x, W1), b1, name='y_hat')
        cost = tf.reduce_mean(tf.square(y_hat - y), name='cost')
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, name='train_step')

    with tf.variable_scope(scope_name, reuse=None):
        _W1 = tf.summary.histogram(values=W1, name='_W1')
        _b1 = tf.summary.histogram(values=b1, name='_b1')
        _cost = tf.summary.scalar(tensor=cost, name='_cost')
        summary = tf.summary.merge([_W1, _b1, _cost], name='summary')  # merge_all()
        if verbose:
            log.info('')
            log.info(x)
            log.info(W1)
            log.info(b1)
            log.info('')
            log.info(y)
            log.info(y_hat)
            log.info(cost)
    return x, y, learning_rate, use_first_pipeline, W1, b1, y_hat, cost, train_step, summary


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)  # ignore tensorflow info

    train_file = os.path.join(SAMPLE_DATA_DIR, 'add.train.tsv')
    valid_file = os.path.join(SAMPLE_DATA_DIR, 'add.valid.tsv')
    test_file = os.path.join(SAMPLE_DATA_DIR, 'add.test.tsv')

    total_train_time = 5
    valid_check_interval = 0.5
    save_model_each_epochs = False  # default False

    input_len = 2  # x1, x2
    output_len = 1  # y
    _learning_rate = 0.01

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
            model_file = os.path.join(SAMPLE_MODELS_DIR, '%s.n_train_%s.batch_size_%s.total_train_time_%s/model' % (model_name, n_train, batch_size, total_train_time))
            model_dir = os.path.dirname(model_file)
            log.info('model_name: %s' % model_name)
            log.info('model_file: %s' % model_file)

            scope_name = '%s.%s.batch_size_%s.total_train_time_%s' % (model_name, DateUtil.current_yyyymmdd_hhmm(), batch_size, total_train_time)
            log.info('scope_name: %s' % scope_name)

            with tf.device('/gpu:0'):
                with tf.Graph().as_default():  # for reusing graph
                    checkpoint = tf.train.get_checkpoint_state(model_dir)
                    is_training = True if training_mode or not checkpoint else False  # learning or testing

                    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
                    with tf.Session(config=config) as sess:
                        train_pipeline = input_pipeline([train_file], batch_size=batch_size, shuffle=True, delim='\t', splits=3)
                        valid_pipeline = input_pipeline([valid_file], batch_size=n_valid, shuffle=True, delim='\t', splits=3)
                        test_pipeline = input_pipeline([test_file], batch_size=n_test, shuffle=True, delim='\t', splits=3)

                        if is_training:  # training
                            x, y, learning_rate, use_first_pipeline, W1, b1, y_hat, cost, train_step, summary = create_graph(model_name,
                                                                                                                             scope_name,
                                                                                                                             first_pipeline=train_pipeline,
                                                                                                                             second_pipeline=valid_pipeline,
                                                                                                                             verbose=False)
                            sess.run(tf.global_variables_initializer())
                            saver = tf.train.Saver(max_to_keep=None)

                            train_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/train', sess.graph)
                            valid_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/valid', sess.graph)

                            coordinator = tf.train.Coordinator()  # coordinator for enqueue threads
                            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)  # start filename queue
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
                                while running:
                                    epoch += 1
                                    for i in range(1, batch_count + 1):
                                        if stop_timer.is_over():
                                            running = False
                                            break

                                        if coordinator.should_stop():
                                            break

                                        nth_batch += 1
                                        _, _train_cost, _summary = sess.run([train_step, cost, summary],
                                                                            feed_dict={learning_rate: _learning_rate, use_first_pipeline: True})
                                        train_writer.add_summary(_summary, global_step=nth_batch)

                                        if valid_timer.is_over():
                                            _valid_cost, = sess.run([cost],
                                                                    feed_dict={learning_rate: _learning_rate, use_first_pipeline: False})
                                            valid_writer.add_summary(_summary, global_step=nth_batch)
                                            if _valid_cost < min_valid_cost:
                                                min_valid_cost = _valid_cost
                                                min_valid_epoch = epoch
                                            log.info('[epoch: %s, nth_batch: %s] train cost: %.8f, valid cost: %.8f' % (epoch, nth_batch, _train_cost,
                                                                                                                        _valid_cost))
                                            if min_valid_epoch == epoch:  # save the lastest best model
                                                saver.save(sess, model_file)

                                    if save_model_each_epochs:
                                        saver.save(sess, model_file, global_step=epoch)

                                log.info('')
                                log.info(
                                    '"%s" train: min_valid_cost: %.8f, min_valid_epoch: %s,  %.2f secs (batch_size: %s,  total_input_data: %s, total_epochs: %s, total_train_time: %s secs)' % (
                                        model_name, min_valid_cost, min_valid_epoch, watch.elapsed(),
                                        batch_size, NumUtil.comma_str(batch_size * nth_batch), epoch, total_train_time))
                                log.info('')
                            except:
                                log.info(traceback.format_exc())
                            finally:
                                coordinator.request_stop()
                                coordinator.join(threads)  # Wait for threads to finish.
                        else:  # testing
                            x, y, learning_rate, use_first_pipeline, W1, b1, y_hat, cost, train_step, summary = create_graph(model_name,
                                                                                                                             scope_name,
                                                                                                                             first_pipeline=test_pipeline,
                                                                                                                             second_pipeline=test_pipeline,
                                                                                                                             verbose=False)

                            log.info('')
                            log.info('model loaded... %s' % model_file)
                            saver = tf.train.Saver(max_to_keep=None)
                            saver.restore(sess, model_file)
                            log.info('model loaded OK. %s' % model_file)

                            coordinator = tf.train.Coordinator()
                            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
                            try:
                                watch = WatchUtil()
                                watch.start()

                                _test_cost, _W1, _b1, _x_batch, _y_batch, _y_hat_batch = sess.run([cost, W1, b1, x, y, y_hat],
                                                                                                  feed_dict={learning_rate: _learning_rate,
                                                                                                             use_first_pipeline: True})

                                log.info('')
                                log.info('W1: %s' % ['%.4f' % i for i in _W1])
                                log.info('b1: %.4f' % _b1)
                                for (x1, x2), _y, _y_hat in zip(_x_batch, _y_batch, _y_hat_batch):
                                    log.debug('%3d + %3d = %4d (y_hat: %4.1f)' % (x1, x2, _y, _y_hat))
                                log.info('')
                                log.info(
                                    '"%s" test: test_cost: %.8f, %.2f secs (batch_size: %s)' % (model_name, _test_cost, watch.elapsed(), batch_size))
                                log.info('')
                            except:
                                log.info(traceback.format_exc())
                            finally:
                                coordinator.request_stop()
                                coordinator.join(threads)  # Wait for threads to finish.
