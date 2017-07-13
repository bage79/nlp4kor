import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from bage_utils.base_util import is_server
from nlp4kor.config import MNIST_DATA_DIR, MNIST_DAE_MODEL_DIR, log

if __name__ == '__main__':
    mnist_data = os.path.join(MNIST_DATA_DIR, MNIST_DAE_MODEL_DIR)  # input
    device2use = '/gpu:0' if is_server() else '/cpu:0'

    model_file = os.path.join(MNIST_DAE_MODEL_DIR, 'dae_mnist_model≤/model')  # .%s' % max_sentences
    log.info('model_file: %s' % model_file)

    model_dir = os.path.dirname(model_file)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    image_shape = (28, 28)
    mnist = input_data.read_data_sets(mnist_data, one_hot=True)
    assert (mnist.train.images.shape[1] == mnist.test.images.shape[1])
    n_input_dim = mnist.train.images.shape[1]  # MNIST data input (img shape: 28*28)
    n_output_dim = n_input_dim  # MNIST data input (img shape: 28*28)
    n_hidden_1 = 256  # 1st layer num features
    n_hidden_2 = 256  # 2nd layer num features

    log.info('n_input_dim: %s' % n_input_dim)
    log.info('n_output_dim: %s' % n_output_dim)
    log.info('n_hidden_1: %s' % n_hidden_1)
    log.info('n_hidden_2: %s' % n_hidden_2)


    def denoising_autoencoder(_X, _weights, _biases, _keep_prob):
        layer_1 = tf.nn.sigmoid(tf.matmul(_X, _weights['h1']) + _biases['b1'])
        layer_1out = tf.nn.dropout(layer_1, _keep_prob)
        layer_2 = tf.nn.sigmoid(tf.matmul(layer_1out, _weights['h2']) + _biases['b2'])
        layer_2out = tf.nn.dropout(layer_2, _keep_prob)
        return tf.nn.sigmoid(tf.matmul(layer_2out, _weights['out']) + _biases['out'])


    with tf.device(device2use):
        log.info('create graph...')
        # tf Graph input
        x = tf.placeholder('float', [None, n_input_dim])
        y = tf.placeholder('float', [None, n_output_dim])
        dropout_keep_prob = tf.placeholder('float')

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input_dim, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_output_dim]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_output_dim]))
        }

        out = denoising_autoencoder(x, weights, biases, dropout_keep_prob)  # model
        cost = tf.reduce_mean(tf.pow(out - y, 2))  # define loss and optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        init = tf.global_variables_initializer()  # initialize
        log.info('create graph OK.')
        log.info('')

    with tf.device('/cpu:0'):
        saver = tf.train.Saver(max_to_keep=3)  # => cpu

    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess: # original source code
    with tf.Session() as sess:
        sess.run(init)

        training_epochs = 30
        batch_size = 100
        display_step = 5
        plot_step = 10
        noise_rate = 0.3
        dropout_keep = 0.5
        if not os.path.exists(model_file + '.index') or not os.path.exists(model_file + '.meta'):
            log.info('learning...')
            for epoch in range(training_epochs):
                avg_cost = 0.
                num_batch = int(mnist.train.num_examples / batch_size)
                for i in range(num_batch):
                    randidx = np.random.randint(mnist.train.images.shape[0], size=batch_size)
                    batch_xs = mnist.train.images[randidx, :]
                    batch_xs_noisy = batch_xs + noise_rate * np.random.randn(batch_xs.shape[0], n_input_dim)  # add noise

                    feed1 = {x: batch_xs_noisy, y: batch_xs, dropout_keep_prob: dropout_keep}  # dropout_keep_prob=0.5 for learning
                    sess.run(optimizer, feed_dict=feed1)

                    if epoch % display_step == 0:
                        feed2 = {x: batch_xs_noisy, y: batch_xs, dropout_keep_prob: 1.}  # dropout_keep_prob=1 for calculate cost
                        avg_cost += sess.run(cost, feed_dict=feed2) / num_batch

                log.info('Epoch: %3d/%3d cost: %.4f' % (epoch, training_epochs, avg_cost))

                if epoch % plot_step == 0 or epoch == training_epochs - 1:
                    # randidx = np.random.randint(mnist.test.images.shape[0], size=1)
                    # print('randidx:', randidx)

                    image_original = mnist.test.images[[0], :]
                    image_noised = image_original + 0.3 * np.random.randn(1, n_input_dim) # add noise
                    output = sess.run(out, feed_dict={x: image_original, dropout_keep_prob: 1.})
                    image_output = np.reshape(output, image_shape)

                    plt.matshow(np.reshape(image_original, image_shape), cmap=plt.get_cmap('gray'))
                    plt.title('[' + str(epoch) + '] Original Image')
                    plt.colorbar()
                    plt.matshow(np.reshape(image_noised, image_shape), cmap=plt.get_cmap('gray'))
                    plt.title('[' + str(epoch) + '] Input Image')
                    plt.colorbar()
                    plt.matshow(image_output, cmap=plt.get_cmap('gray'))
                    plt.title('[' + str(epoch) + '] Reconstructed Image')
                    plt.colorbar()

                    plt.interactive(False)
                    plt.show(block=True)

                # saver.save(sess, model_file)  # , global_step=epoch)
            log.info('learning OK.')
        else:
            log.info('restore...')
            saver.restore(sess, model_file)
            log.info('restore OK.')
