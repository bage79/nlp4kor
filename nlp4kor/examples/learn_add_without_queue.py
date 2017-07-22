import os
import traceback

import numpy as np
import tensorflow as tf

from bage_utils.date_util import DateUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor.config import DATA_DIR, TENSORBOARD_LOG_DIR, log, MODELS_DIR


def features_labels_from_file(filenames, data_size, batch_size=100, shuffle=True, delim='\t'):
    _features, _labels = [], []
    for filename in filenames:
        if len(_features) > data_size:
            break
        with open(filename) as f:
            for line in f.readlines():
                if len(_features) >= data_size:
                    break
                line = line.strip()
                x1, x2, y = line.split(delim)
                _features.append((int(x1), int(x2)))
                _labels.append(int(y))

    features = np.array(_features)
    labels = np.array(_labels)
    if shuffle:
        random_idx = np.random.permutation(len(_features))
        features, labels = features[random_idx], labels[random_idx]

    labels = labels.reshape(len(labels), -1)

    splits = len(features) // batch_size
    if len(features) % batch_size > 0:
        splits += 1
    return zip(np.array_split(features, splits), np.array_split(labels, splits))


def next_batch(batches):
    for features_batch, labels_batch in batches:
        # print(features_batch, labels_batch)
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
    train_file = os.path.join(DATA_DIR, 'add.train.tsv')
    test_file = os.path.join(DATA_DIR, 'add.test.tsv')

    input_len = 2  # x1, x2
    output_len = 1  # y

    n_train, n_test = 11, 10
    batch_size = 2
    total_epochs = 1

    scope_postfix = DateUtil.current_yyyymmdd_hhmm()
    log.debug('scope_postfix: %s' % scope_postfix)

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
        train_batches = features_labels_from_file([train_file], data_size=n_train, batch_size=batch_size, shuffle=True, delim='\t')
        test_batches = features_labels_from_file([test_file], data_size=n_train, batch_size=batch_size, shuffle=True, delim='\t')
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR, sess.graph)

            batch_count = n_train // batch_size  # batch count for one epoch
            try:
                for epoch in range(1, total_epochs + 1):
                    for _features_batch, _labels_batch in next_batch(train_batches):
                        nth_batch += 1
                        _, _train_cost, _summary_merge = sess.run([train_op, cost, summary_merge], feed_dict={x: _features_batch, y: _labels_batch})  # FIXME:

                        writer.add_summary(_summary_merge, global_step=epoch)

                        if _train_cost < min_cost:
                            min_cost = _train_cost
                            min_epoch = epoch
                    log.info('[epoch: %s, nth_batch: %s] train cost: %.4f' % (epoch, nth_batch, _train_cost))
                    saver.save(sess, model_file, global_step=epoch, latest_filename='checkpoint')  # end of each epoch
                log.info('[min_epoch: %s] min_cost: %.4f' % (min_epoch, min_cost))
            except:
                log.info(traceback.format_exc())

    log.info('without queue runner: %.2f secs' % watch.elapsed())
