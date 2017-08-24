import os
import traceback

import tensorflow as tf

from nlp4kor.config import log, SAMPLE_DATA_DIR


def input_pipeline(filenames, batch_size=100, shuffle=True, delim='\t', tokens=2):
    min_after_dequeue = batch_size
    capacity = min_after_dequeue + 3 * batch_size

    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader(skip_header_lines=None)
    _key, value = reader.read(filename_queue)
    x, y = tf.decode_csv(value, field_delim=delim, record_defaults=[[''] for _ in range(tokens)])

    if shuffle:
        features_batch, labels_batch = tf.train.shuffle_batch([x, y], batch_size=batch_size, capacity=capacity,
                                                              min_after_dequeue=min_after_dequeue)
    else:
        features_batch, labels_batch = tf.train.batch([x, y], batch_size=batch_size, capacity=capacity)
    return features_batch, labels_batch


if __name__ == '__main__':
    shuffle = False
    batch_size = 5
    data_file = os.path.join(SAMPLE_DATA_DIR, 'en2kor.tsv')

    if not os.path.exists(data_file):
        log.error('file not exists. %s' % data_file)
        exit()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    filenames = [data_file]
    features_batch, labels_batch = input_pipeline(filenames, batch_size=batch_size, shuffle=shuffle, tokens=2)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        log.info('coordinator: %s' % coordinator)
        log.info('threads: %s, %s' % (len(threads), threads))
        try:
            for nth_batch in range(5):
                if coordinator.should_stop():
                    break

                _features_batch, _labels_batch = sess.run([features_batch, labels_batch])
                log.info('')
                log.info('nth_batch: %s' % nth_batch)
                for _f, _l in zip(_features_batch, _labels_batch):
                    log.info('%s %s' % (_f.decode('utf8'), _l.decode('utf8')))  # decode for print
        except:
            log.info(traceback.format_exc())
        finally:
            coordinator.request_stop()
            coordinator.join(threads)  # Wait for threads to finish.
