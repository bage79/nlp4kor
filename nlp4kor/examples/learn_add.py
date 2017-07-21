import os
import traceback

import numpy as np
import tensorflow as tf

from bage_utils.date_util import DateUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor.config import DATA_DIR, TENSORBOARD_LOG_DIR, log, MODELS_DIR


def input_pipeline(filenames, batch_size=100, shuffle=True, delim='\t', tokens=2):
    min_after_dequeue = batch_size
    capacity = min_after_dequeue + 3 * batch_size

    filename_queue = tf.train.string_input_producer(filenames, name='filename_queue')
    reader = tf.TextLineReader(skip_header_lines=None, name='reader')
    _key, value = reader.read(filename_queue)
    x1, x2, y = tf.decode_csv(value, field_delim=delim, record_defaults=[[0.] for _ in range(tokens)], name='decode_csv')
    # log.debug('%s %s %s' % (x1, x2, y))
    feature = tf.reshape([x1, x2], shape=[tokens - 1])
    label = tf.reshape([y], shape=[1])
    # log.debug(feature)

    if shuffle:
        features_batch, labels_batch = tf.train.shuffle_batch([feature, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    else:
        features_batch, labels_batch = tf.train.batch([feature, label], batch_size=batch_size, capacity=capacity)

    return features_batch, labels_batch


def batch_from_cpu(filenames, batch_size=100, shuffle=True, delim='\t'):
    features, labels = [], []
    for filename in filenames:
        with open(filename) as f:
            for line in f:
                f, l = line.split(delim)
                features.append(f)
                labels.append(l)

    if shuffle:
        random_idx = np.random.permutation(len(features))
        features, labels = features[random_idx], labels[random_idx]
    else:
        features, labels = np.array(features), np.array(labels)

    features_batch, labels_batch = [], []
    for f, l in zip(features, labels):
        features_batch.append(f)
        labels_batch.append(l)
        if len(features) >= batch_size:
            yield features_batch, labels_batch
            features_batch, labels_batch = [], []


def create_data4add(data_file, n_data, digit_max=99):
    input_len = 2  # x1, x2
    train_x = np.random.randint(digit_max + 1, size=input_len * n_data).reshape(-1, input_len)
    train_y = np.array([a + b for a, b in train_x])
    # log.info(train_x.shape)
    # log.info(train_y.shape)

    with open(data_file, 'wt') as f:
        for (x1, x2), y in zip(train_x, train_y):
            # log.info('%s + %s = %s' % (x1, x2, y))
            f.write('%s\t%s\t%s\n' % (x1, x2, y))


if __name__ == '__main__':
    train_file = os.path.join(DATA_DIR, 'add.train.tsv')
    test_file = os.path.join(DATA_DIR, 'add.test.tsv')

    input_len = 2  # x1, x2
    output_len = 1  # y

    n_train, n_test = 1000, 10
    batch_size = 1
    total_epochs = 10
    model_save_interval_epoch = 1

    scope_postfix = DateUtil.current_yyyymmddhhmm()

    variable_scope = os.path.basename(__file__)
    if not os.path.exists(train_file):
        create_data4add(train_file, n_train, digit_max=99)
    if not os.path.exists(test_file):
        create_data4add(test_file, n_test, digit_max=99)

    model_name = os.path.basename(__file__).replace('.py', '')
    log.info('model_name: %s' % model_name)
    model_file = os.path.join(MODELS_DIR, '%s.n_train=%s.batch_size=%s/model' % (model_name, n_train, batch_size))  # .%s' % max_sentences
    model_dir = os.path.dirname(model_file)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log.info('model_file: %s' % model_file)

    with tf.variable_scope(variable_scope):
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_len], name='x')
        y = tf.placeholder(dtype=tf.float32, shape=[None, output_len], name='y')

        W1 = tf.get_variable(dtype=tf.float32, shape=[input_len, output_len], initializer=tf.random_normal_initializer(), name='W1')
        b1 = tf.get_variable(dtype=tf.float32, initializer=tf.constant(0.0, shape=[output_len]), name='b1')

        y_hat = tf.add(tf.matmul(x, W1), b1, name='y_hat')
        cost = tf.reduce_mean(tf.square(y_hat - y), name='cost')
        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

        tf.summary.histogram(values=W1, name='W1/' + scope_postfix)
        tf.summary.histogram(values=b1, name='b1/' + scope_postfix)
        tf.summary.scalar(tensor=cost, name='cost/' + scope_postfix)
        summary_merge = tf.summary.merge_all()

        # cost operation is valid? check y_hat's shape and y's shape
        log.info('')
        log.info(x)
        log.info(W1)
        log.info(b1)
        log.info('')
        log.info(y)
        log.info(y_hat)
        log.info(cost)

        watch = WatchUtil()
        watch.start()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # do not use entire memory for this session
        # with queue runner
        saver = tf.train.Saver(max_to_keep=100)
        checkpoint = tf.train.get_checkpoint_state(model_dir)  # TODO:
        log.debug('')
        log.debug('checkpoint:')
        log.debug(checkpoint)

        min_epoch, min_cost = 0, 1e10
        nth_batch = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            if checkpoint and checkpoint.model_checkpoint_path:  # TODO:
                log.debug('')
                log.debug('model loaded... %s' % checkpoint.model_checkpoint_path)
                saver.restore(sess, checkpoint.model_checkpoint_path)
                log.debug('model loaded OK. %s' % checkpoint.model_checkpoint_path)

                test_features_batch, test_labels_batch = input_pipeline([test_file], batch_size=n_test, tokens=3)

                coordinator = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
                try:
                    _features_batch, _labels_batch = sess.run([test_features_batch, test_labels_batch])

                    _, _test_cost, _y_hat_batch, _W1, _b1 = sess.run([train_op, cost, y_hat, W1, b1], feed_dict={x: _features_batch, y: _labels_batch})
                    log.info('')
                    log.info('test cost: %.4f' % _test_cost)
                    log.info('W1: %s' % ['%.4f' % i for i in _W1])
                    log.info('b1: %.4f' % _b1)
                    for (x1, x2), _y, _y_hat in zip(_features_batch, _labels_batch, _y_hat_batch):
                        log.debug('%3d + %3d = %4d (y_hat: %4.1f)' % (x1, x2, _y, _y_hat))
                except:
                    log.info(traceback.format_exc())
                finally:
                    coordinator.request_stop()
                    coordinator.join(threads)  # Wait for threads to finish.
            else:
                writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR, sess.graph)

                train_features_batch, train_labels_batch = input_pipeline([train_file], batch_size=batch_size, tokens=3)

                coordinator = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
                try:
                    for epoch in range(total_epochs):
                        for i in range(n_train // batch_size):
                            nth_batch += 1
                            if coordinator.should_stop():
                                break

                            _features_batch, _labels_batch = sess.run([train_features_batch, train_labels_batch])
                            _, _train_cost, _summary_merge = sess.run([train_op, cost, summary_merge], feed_dict={x: _features_batch, y: _labels_batch})

                            writer.add_summary(_summary_merge, global_step=nth_batch)

                        if _train_cost < min_cost:
                            min_cost = _train_cost
                            min_epoch = epoch

                        log.info('[epoch: %s, nth_batch: %s] train cost: %.4f' % (epoch, nth_batch, _train_cost))
                        if epoch == 0 or epoch % model_save_interval_epoch == 0:
                            saver.save(sess, model_file + '.epoch=%s' % epoch)
                    log.info('[min_epoch: %s] min_cost: %.4f' % (min_epoch, min_cost))
                except:
                    log.info(traceback.format_exc())
                finally:
                    coordinator.request_stop()
                    coordinator.join(threads)  # Wait for threads to finish.

    log.info('with queue runner: %.2f secs' % watch.elapsed())

    # without queue runner
    watch.start()
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     try:
    #         for i in range(total_input // batch_size):
    #             _features_batch, _labels_batch = sess.run([features_batch, labels_batch])
    #     except:
    #         log.info(traceback.format_exc())
    # log.info('without queue runner: %.2f' % watch.elapsed())
