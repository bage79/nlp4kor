import os
import numpy
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt, gridspec
from tensorflow.examples.tutorials.mnist import input_data
from nlp4kor.config import MNIST_DATA_DIR, MNIST_CNN_MODEL_DIR


def show_image(image: numpy.ndarray, title: str = 'title',
               smoothing=False, relative_color=True, cmap: str = 'Greys'):
    """
    한장의 이미지 출력
    :param image: 2D or 3D array (height, width) or (height, width, channel)
    :param title: label
    :param smoothing: antialiasing 효과 (matplotlibrc 에서 기본값 설정 가능.)
    :param relative_color: 픽셀값이 유사한 경우, 명암의 차이를 크게 한다.
    :param cmap: color map (default: 'Greys'=1 is black.)
    """
    print(image.shape)
    if title:
        plt.title('label: {title}'.format(title=title))

    if smoothing:  # antialiasing
        interpolation = None
    else:
        interpolation = "nearest"

    if relative_color:
        vmin, vmax = None, None
    else:
        vmin, vmax = 0, 1  # for image

    plt.imshow(image, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
    plt.grid(False)
    plt.colorbar()
    plt.show()


def show_images(images, title='title', is_kernel=False, is_pooled_image=False,
                smoothing=False, relative_color=False, cmap='Greys',
                subtitles=[], n_cols=5, max_samples=20):
    """
    이미지(2D array) 또는 커널(feature map)을 이미지로 출력
    :param images: 4D array (NHWC or HWNC)
    :param title: 메인 타이틀
    :param is_kernel: True=커널(HWCC)인 경우 이미지형태로(NHWC) reshape.
    :param is_pooled_image: True=Pooling Layer에서 출력된 이미지
    :param smoothing: antialiasing 효과 (matplotlibrc 에서 기본값 설정 가능.)
    :param relative_color: 픽셀값이 유사한 경우, 명암의 차이를 크게 한다.
    :param cmap: color map (default: 'Greys'=1 is black.)
    :param subtitles: 각 이미지의 타이틀 (라벨 표시용)
    :param n_cols: 여러 개의 이미지를 표시할 때, 가로에 표시될 이미지 개수
    :param max_samples: images에서 몇 개의 데이타를 뽑아서 표시할 지 (max of N)
    """
    if smoothing:  # antialiasing
        interpolation = None
    else:
        interpolation = "nearest"

    if relative_color:
        vmin, vmax = None, None
    else:
        if is_kernel:  # 커널인 경우만 -1=Red, 1=Blue로 강제 지정
            vmin, vmax = -1, 1  # for kernel
            cmap = 'RdBu'
        else:
            vmin, vmax = 0, 1  # for image

    if is_kernel:  # need to reshape (HW NC -> N HW C)
        print('images(is_kernel):', images.shape)
        height, width, input_channels, output_channels = images.shape  # HWNC
        images = images.reshape(height, width, -1)  # height, width, input_channels * output_channels
        images = np.rollaxis(images, 2)  # input_channels * output_channels, height, width
    elif is_pooled_image:  # NHWc -> C HW (14, 14, 32) -> (32, 14, 14)
        print('images(is_pooled_image):', images.shape)
        images = np.rollaxis(images, 2, 0)

    print('images:', images.shape)
    if len(images) > max_samples:
        title = '%s (%s of %s)' % (title, max_samples, len(images))
        images = images[:max_samples]

    if len(images) == 0:
        return
    elif len(images) == 1:
        show_image(images[0], title=title, smoothing=smoothing,
                   relative_color=relative_color, cmap=cmap)
    elif len(images) > 1:
        n_rows = len(images) // n_cols
        if len(images) % n_cols != 0:
            n_rows += 1

        fig = plt.figure(figsize=(10, 10))  # 전체 사이즈 (인치)
        gs = gridspec.GridSpec(n_rows, n_cols)  # 여러 이미지 표시할 그리드
        fig.suptitle(title, fontsize=20)  # 메인 타이틀

        im = plt.imshow(images[0], cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)  # for colorbar
        nth = -1
        for row in range(n_rows):
            if nth >= max_samples:
                break
            for col in range(n_cols):
                nth += 1
                if nth >= max_samples:
                    break
                image = images[nth]  # WHC
                ax = plt.subplot(gs[nth])  # 각 이미지 공간
                ax.grid(False)  # 그리드 출력
                ax.axis('off')  # 가로/세로축 출력
                if len(subtitles) == len(images):
                    ax.set_title(subtitles[nth])
                else:
                    ax.set_title(nth)

                ax.imshow(image, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
        fig.colorbar(im, cax=cax)
        plt.show()


def virtual_kernels(height, width, input_channels, output_channels):
    """
    가상의 커널(feature map)을 생성 (HWNC)
    """
    colors = np.linspace(0, 1, output_channels)
    print('colors:', len(colors), colors)

    images = np.ndarray((output_channels, height, width, input_channels))  # CHWN
    for nth, image in enumerate(images):
        image.fill(colors[nth])  # same color in same channel.
    return np.moveaxis(images, 0, 3)  # HWC


def weight_variable(shape):  # W
    """
    Weight 생성 함수 (kernel)
    :prarm shape: height, width, input channels, output channels (=HWNC)
    """
    # truncated_normal 대칭성을 깨뜨리고 기울기(gradient)가 0이 되는 것을 방지하기 위해 ?
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # b
    """
    Bias 생성 함수 (kernel)
    :prarm shape: height, width, input channels, output channels (=HWNC)
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, strides=[1, 1, 1, 1]):
    """
    Convolution Layer
    :param x: input data
    :param W: Weight(kernel)
    :param strides: batch, height, width, channel (NHWC)
    """
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_2x2(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    """
    Pooling Layer (기본 2x2로 pooling)
    :param x: input data
    :param ksize: batch, height, width, channel (NHWC)
    :param strides: batch, height, width, channel (NHWC)
    """
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME')


if __name__ == '__main__':
    mnist_data = MNIST_DATA_DIR  # input
    mnist_model = MNIST_CNN_MODEL_DIR  # output
    if not os.path.exists(mnist_model):
        os.makedirs(mnist_model)
        print('%s created' % mnist_model)
    print('mnist_data:', mnist_data)  # input
    print('mnist_model:', mnist_model)  # output

    mnist = input_data.read_data_sets(mnist_data, one_hot=True)

    _, height, width, n_channel0 = x_image_shape = [-1, 28, 28, 1]  # gray scale
    # _, height, width, n_channel0 = x_image_shape = [-1, 28, 28, 3] # RGB color
    # N = mnist.test.labels.shape[0]
    n_classes = mnist.test.labels.shape[1]  # 0 ~ 9

    # input data (X)
    x = tf.placeholder(tf.float32, [None, height * width])  # batch, height*width*channel
    x_image = tf.reshape(x, x_image_shape)  # batch, height, width, channel (NHWC)
    print('mnist.test.images:', mnist.test.images.shape)
    print('x:', x)
    print('x_image:', x_image)
    # print('?(N):', N)
    # print('mnist.test.images.image: %s, %s * %s, %s (N, H*W, C)' % (N, width, height, channels))

    # input data (Y)
    print('n_classes:', n_classes)
    y_ = tf.placeholder(tf.float32, [None, n_classes])  # labels, from data
    print('y_:', y_)

    # conv layer 1
    n_channel1 = 32
    W_conv1 = weight_variable([5, 5, n_channel0, n_channel1])  # height, width, input channels, output channels
    b_conv1 = bias_variable([n_channel1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # relu ( conv(x, W) + b )
    h_pool1 = max_pool_2x2(h_conv1)  # 28 * 28 -> 14 * 14
    print('W_conv1:', W_conv1)
    print('b_conv1:', b_conv1)
    print('h_conv1:', h_conv1)
    print('h_pool1:', h_pool1)

    # conv layer 2
    n_channel2 = 64
    W_conv2 = weight_variable([5, 5, n_channel1, n_channel2])
    b_conv2 = bias_variable([n_channel2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # relu ( conv(x, W) + b )
    h_pool2 = max_pool_2x2(h_conv2)  # 14 * 14 -> 7 * 7
    print('W_conv2:', W_conv2)
    print('b_conv2:', b_conv2)
    print('h_conv2:', h_conv2)
    print('h_pool2:', h_pool2)

    # fully connected layer 1 = 3136 -> 1024
    height_pool2, width_pool2, n_channel2 = int(h_pool2.shape[1]), int(h_pool2.shape[2]), int(h_pool2.shape[3])
    print(h_pool2.shape)
    print('height_pool2, width_pool2, n_channel2:', height_pool2, width_pool2, n_channel2)

    n_features = int(height_pool2) * int(width_pool2) * n_channel2  # input of FFNN
    n_hidden = 1024  # neuron of FFNN
    h_pool2_flat = tf.reshape(h_pool2, [-1, n_features])  # input of FFNN
    W_fc1 = weight_variable([n_features, n_hidden])  # Weight of FFNN
    b_fc1 = bias_variable([n_hidden])  # bias of FFNN
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # output of FFNN

    # drop out
    keep_prob = tf.placeholder(tf.float32)  # keep rate for dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropouted output of FFNN

    print('h_pool2_flat:', h_pool2_flat)
    print('W_fc1:', W_fc1)
    print('b_fc1:', b_fc1)
    print('h_fc1:', h_fc1)
    print('h_fc1_drop:', h_fc1_drop)

    # fully connected layer 2 = 1024 -> 10
    n_classes = mnist.test.labels.shape[1]  # number of lables 0 ~ 9
    W_fc2 = weight_variable([n_hidden, n_classes])  # W
    b_fc2 = bias_variable([n_classes])  # b
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # Wx + b
    y = tf.nn.softmax(logits)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=1))  # loss (cost)
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # training

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # evaluation
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # evaluation

    iteration = 10000  # max iteration
    check_accuracy_interval = 10 # min(1000, (iteration / 10))  # less than 1000

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('training...')
        for i in range(1, iteration + 1):  # 1 ~ 10000
            batch = mnist.train.next_batch(50)  # mini batch (size=50)
            batch_x, batch_y = batch[0], batch[1]  # images, labels
            train_step.run(feed_dict={x: batch[0], y_: batch_y, keep_prob: 0.5})  # with dropout
            if i % check_accuracy_interval == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch[1], keep_prob: 1.0})
                print("[step %d] training accuracy %.8f" % (i, train_accuracy))
        print('training OK.')

        print('evaluate...')
        _accuracy, _W_conv1, _W_conv2, _h_pool1, _h_pool2 = sess.run(
            [accuracy, W_conv1, W_conv2, h_pool1, h_pool2],
            feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})  # without dropout
        print("evaluate OK. accuracy %.8f" % _accuracy)

        # print('save models to numpy array...')
        # np.save(os.path.join(mnist_model, 'W_conv1'), _W_conv1)
        # np.save(os.path.join(mnist_model, 'W_conv2'), _W_conv2)
        # np.save(os.path.join(mnist_model, 'h_pool1'), _h_pool1)
        # np.save(os.path.join(mnist_model, 'h_pool2'), _h_pool2)
        # np.save(os.path.join(mnist_model, 'labels'), mnist.test.labels)
        # print('save models to numpy array OK.')
