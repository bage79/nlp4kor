import math
import os
import traceback

import numpy as np
import tensorflow as tf

from bage_utils.date_util import DateUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor.config import DATA_DIR, TENSORBOARD_LOG_DIR, log, MODELS_DIR


def next_batch(filenames, data_size, batch_size=100, delim='\t'):
    """
    read from big data files.
    :param filenames:
    :param data_size:
    :param batch_size:
    :param delim:
    :return:
    """
    _data_size = 0
    for filename in filenames:
        if _data_size > data_size:
            return
        with open(filename) as f:
            _features, _labels = [], []
            for line in f.readlines():
                if len(_features) >= batch_size:
                    features_batch = np.array(_features, dtype=np.float32)
                    labels_batch = np.array(_labels, dtype=np.float32)
                    labels_batch = labels_batch.reshape(len(_labels), -1)
                    _features, _labels = [], []
                    yield features_batch, labels_batch
                if _data_size > data_size:
                    return

                line = line.strip()
                x1, x2, y = line.split(delim)
                _features.append((float(x1), float(x2)))
                _labels.append(float(y))
                _data_size += 1


def next_batch_in_memory(filenames, data_size, batch_size=100, shuffle=True, delim='\t'):
    """
    read from big data files.
    :param filenames:
    :param data_size:
    :param batch_size:
    :param shuffle:
    :param delim:
    :return:
    """
    _features, _labels = [], []  # read all data
    for filename in filenames:
        if len(_features) > data_size:
            return
        with open(filename) as f:
            for line in f.readlines():
                if len(_features) >= data_size:
                    return
                line = line.strip()
                x1, x2, y = line.split(delim)
                _features.append((int(x1), int(x2)))
                _labels.append(int(y))

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
    reuse_model = False  # FIXME: if model file exists, reuse it.
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

    for batch_size in [100]:  # FIXME: [1, 10, 100]:
        model_name = os.path.basename(__file__).replace('.py', '')
        log.info('model_name: %s' % model_name)
        model_file = os.path.join(MODELS_DIR, '%s.n_train=%s.batch_size=%s/model' % (model_name, n_train, batch_size))  # .%s' % max_sentences
        model_dir = os.path.dirname(model_file)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        log.info('model_file: %s' % model_file)

        variable_scope = os.path.basename(__file__)
        graph_exists = False
        with tf.device('/gpu:0'):
            with tf.Graph().as_default():  # for reusing graph
                with tf.variable_scope(variable_scope, reuse=True if graph_exists else None):  # for reusing graph
                    x = tf.placeholder(dtype=tf.float32, shape=[None, input_len], name='x')
                    y = tf.placeholder(dtype=tf.float32, shape=[None, output_len], name='y')

                    W1 = tf.get_variable(dtype=tf.float32, shape=[input_len, output_len], initializer=tf.truncated_normal_initializer(), name='W1')
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

                    if not graph_exists:
                        log.info('')
                        log.info(x)
                        log.info(W1)
                        log.info(b1)
                        log.info('')
                        log.info(y)
                        log.info(y_hat)
                        log.info(cost)  # cost operation is valid? check y_hat's shape and y's shape
                    graph_exists = True

                saver = tf.train.Saver(max_to_keep=100)
                if reuse_model:
                    checkpoint = tf.train.get_checkpoint_state(model_dir)
                    log.debug('')
                    log.debug('checkpoint:')
                    log.debug(checkpoint)

                min_epoch, min_cost = 0, 1e10
                nth_batch = 0

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True  # do not use entire memory for this session
                with tf.Session(config=config) as sess:
                    sess.run(tf.global_variables_initializer())

                    writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR, sess.graph)

                    batch_count = math.ceil(n_train / batch_size)  # batch count for one epoch
                    try:
                        watch = WatchUtil()
                        watch.start()
                        for epoch in range(1, total_epochs + 1):
                            for _features_batch, _labels_batch in next_batch([train_file], data_size=n_train, batch_size=batch_size, delim='\t'):
                                nth_batch += 1
                                _, _train_cost, _summary_merge = sess.run([train_step, cost, summary_merge],
                                                                          feed_dict={x: _features_batch, y: _labels_batch})
                                writer.add_summary(_summary_merge, global_step=nth_batch)

                            if _train_cost < min_cost:
                                min_cost = _train_cost
                                min_epoch = epoch
                            log.info('[epoch: %s, nth_batch: %s] train cost: %.4f' % (epoch, nth_batch, _train_cost))
                            saver.save(sess, model_file, global_step=epoch, latest_filename='checkpoint')  # end of each epoch
                        log.info('[min_epoch: %s] min_cost: %.4f' % (min_epoch, min_cost))
                        log.info('')
                        log.info('without queue runner: %.2f secs (batch_size: %s)' % (watch.elapsed(), batch_size))
                        log.info('')
                    except:
                        log.info(traceback.format_exc())
