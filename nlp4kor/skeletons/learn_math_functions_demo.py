import math
import os
import time
import traceback

import numpy as np
import tensorflow as tf

from bage_utils.date_util import DateUtil
from bage_utils.num_util import NumUtil
from bage_utils.timer_util import TimerUtil
from bage_utils.watch_util import WatchUtil
from nlp4kor.config import log, TENSORBOARD_LOG_DIR, MODELS_DIR


def add(x_data):
    y_data = np.sum(x_data, axis=1)  # sum = add all
    return np.expand_dims(y_data, axis=1)


def average(x_data):
    y_data = np.average(x_data, axis=1)
    return np.expand_dims(y_data, axis=1)


def multiply(x_data):
    y_data = np.prod(x_data, axis=1)  # product = multiply all
    return np.expand_dims(y_data, axis=1)


def build_graph(scope_name, n_features, n_hiddens, n_classes, learning_rate, optimizer=tf.train.AdamOptimizer, activation=tf.tanh, weights_initializer=tf.truncated_normal_initializer, bias_value=0.0):
    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_classes])

    w1 = tf.get_variable(initializer=weights_initializer(), shape=[n_features, n_hiddens], name='w1')
    b1 = tf.get_variable(initializer=tf.constant(bias_value, shape=[n_hiddens]), name='b1')
    h1 = tf.nn.xw_plus_b(x, w1, b1)
    h1_out = activation(h1)

    w2 = tf.get_variable(initializer=weights_initializer(), shape=[n_hiddens, n_classes], name='w2')
    b2 = tf.get_variable(initializer=tf.constant(bias_value, shape=[n_classes]), name='b2')
    h2 = tf.nn.xw_plus_b(h1_out, w2, b2)
    y_hat = h2

    cost = tf.reduce_mean(tf.square(y - y_hat), name='cost')
    train_step = optimizer(learning_rate=learning_rate, name='optimizer').minimize(cost, name='train_step')

    rmse = tf.sqrt(cost, name='rmse')
    with tf.name_scope(scope_name):
        cost_ = tf.summary.scalar(tensor=cost, name='cost')
        summary = tf.summary.merge([cost_])
    return x, y, y_hat, cost, rmse, train_step, summary


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)  # ignore tensorflow info

    func = multiply  # TODO: 다른 데이터 생성 함수로 교체해 볼것 add, average
    n_features = 2  # x1, x2
    n_classes = 1  # y
    digits = list(range(-99, 100, 1))
    n_train, n_test = 4000, 100  # 10% of 200 * 200

    x_data = np.random.choice(digits, (n_train + n_test, n_features), replace=True)
    y_data = func(x_data)
    x_train, x_test = x_data[:n_train], x_data[n_train:]
    y_train, y_test = y_data[:n_train], y_data[n_train:]

    log.info('')
    log.info('func: %s' % func.__name__)
    log.info('digits: %s ~ %s ' % (min(digits), max(digits)))
    log.info('x_train: %s' % str(x_train.shape))
    log.info(x_data[:5])
    log.info('y_train: %s' % str(y_train.shape))
    log.info(y_data[:5])
    log.info('x_test: %s' % str(x_test.shape))
    log.info('y_test %s' % str(y_test.shape))

    valid_check_interval = 0.5
    bias_value = 0.0
    early_stop_cost = 0.1  # stop learning

    # default values
    optimizer = tf.train.AdamOptimizer
    activation = tf.sigmoid
    weights_initializer = tf.random_normal_initializer
    n_hiddens = 10  # in one layer
    learning_rate = 0.1
    train_time = 1  # secs

    # # good values
    # # TODO: 0. (cost: 2500-3000)
    activation = tf.nn.relu  # TODO: 1. 학습 가능 여부 (cost: 1700)
    weights_initializer = tf.truncated_normal_initializer  # TODO: 2. 좀더 안정된 분포 (600-700)(min epoch==total epoch)
    n_hiddens = 1000  # TODO: 4. 모델 용량 증가 (cost: 500-700) cost 변화 없음. min epoch < total epoch 이므로 learning_rate를 줄여야 함.
    learning_rate = 0.001  # TODO: 5. 정확도 증가 (cost: 1600) cost 증가했으나 min epoch==total epoch 이므로, cost가 더 줄어들 수 있음 확인. tensorboard 확인
    train_time = 10 * 60  # TODO: 6. 끝까지 학습  (cost: 4-5) (6분)
    # # # TODO: 7. 좀더 줄이려면 어떻게 해야 할까요? (decay)

    log.info('%s -> %s -> %s -> %s -> %s' % (x_train.shape[1], n_hiddens, activation.__name__, n_hiddens, 1))
    log.info('weights_initializer: %s' % weights_initializer.__name__)
    log.info('learning_rate: %.4f' % learning_rate)
    log.info('train_time: %s' % train_time)

    how_many_trains = 3 if train_time < 10 else 1
    log.info('how_many_trains: %s' % how_many_trains)
    for _ in range(how_many_trains):
        time.sleep(1)
        tf.reset_default_graph()  # Clears the default graph stack and resets the global default graph.
        tf.set_random_seed(7942)  # TODO: 3. 결과를 규칙적으로 만들자. (cost: 600-700)

        scope_name = '%s.%s' % (func.__name__, DateUtil.current_yyyymmdd_hhmmss())
        x, y, y_hat, cost, rsme, train_step, summary = build_graph(scope_name, n_features, n_hiddens, n_classes, learning_rate, activation=activation, weights_initializer=weights_initializer,
                                                                   bias_value=bias_value)
        try:
            watch = WatchUtil()

            model_file_saved = False
            model_file = os.path.join(MODELS_DIR, '%s_%s/model' % (os.path.basename(__file__.replace('.py', '')), func.__name__))
            model_dir = os.path.dirname(model_file)
            # log.info('model_file: %s' % model_file)
            if not os.path.exists(model_dir):
                # log.info('model_dir: %s' % model_dir)
                os.makedirs(model_dir)

            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            saver = tf.train.Saver()
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())

                train_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/train', sess.graph)
                valid_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/valid', sess.graph)

                max_cost = 1e10
                best_epoch, best_cost = 0, 1e10
                watch.start('train')

                running, epoch = True, 0
                stop_timer = TimerUtil(interval_secs=train_time)
                valid_timer = TimerUtil(interval_secs=valid_check_interval)
                stop_timer.start()
                valid_timer.start()
                while running:
                    if stop_timer.is_over():
                        break

                    epoch += 1
                    _, train_rsme, train_summary = sess.run([train_step, rsme, summary], feed_dict={x: x_train, y: y_train})
                    train_writer.add_summary(train_summary, global_step=epoch)
                    train_writer.flush()

                    if valid_timer.is_over():
                        valid_rsme, valid_summary = sess.run([rsme, summary], feed_dict={x: x_test, y: y_test})
                        valid_writer.add_summary(valid_summary, global_step=epoch)
                        valid_writer.flush()
                        if valid_rsme < best_cost:
                            best_cost = valid_rsme
                            best_epoch = epoch
                            saver.save(sess, model_file)
                            model_file_saved = True
                            log.info('[epoch: %s] rsme (train/valid): %.1f / %.1f model saved' % (epoch, train_rsme, valid_rsme))
                        else:
                            log.info('[epoch: %s] rsme (train/valid): %.1f / %.1f' % (epoch, train_rsme, valid_rsme))
                        if valid_rsme < early_stop_cost or valid_rsme > max_cost or math.isnan(valid_rsme):
                            running = False
                            break
                watch.stop('train')

                if model_file_saved and os.path.exists(model_file + '.index'):
                    restored = saver.restore(sess, model_file)

                    log.info('')
                    log.info('--------TEST----------')
                    watch.start('test')
                    test_rsme, _y_hat = sess.run([rsme, y_hat], feed_dict={x: x_test, y: y_test})

                    log.info('%s rsme (test): %.1f (epoch best/total: %s/%s), activation: %s, n_hiddens: %s, learning_rate: %s, weights_initializer: %s' % (
                        func.__name__, test_rsme, NumUtil.comma_str(best_epoch), NumUtil.comma_str(epoch), activation.__name__,
                        n_hiddens, learning_rate, weights_initializer.__name__))

                    # _y_hat = np.round(_y_hat)
                    for i in range(min(5, _y_hat.shape[0])):
                        log.info('%s\t->\t%.1f\t(label: %d)' % (x_test[i], _y_hat[i], y_test[i]))
                    watch.stop('test')
                    log.info('--------TEST----------')
            log.info(watch.summary())
        except:
            traceback.print_exc()

    log.info('OK.')
