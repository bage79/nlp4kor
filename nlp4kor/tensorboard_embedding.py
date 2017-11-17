import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from nlp4kor.config import TENSORBOARD_LOG_DIR


def mnist_embedding():
    """
    https://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
    """
    from tensorflow.examples.tutorials.mnist import input_data
    import numpy as np
    import matplotlib.pyplot as plt

    def create_sprite_image(images):
        """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
        if isinstance(images, list):
            images = np.array(images)
        img_h = images.shape[1]
        img_w = images.shape[2]
        n_plots = int(np.ceil(np.sqrt(images.shape[0])))

        spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

        for i in range(n_plots):
            for j in range(n_plots):
                this_filter = i * n_plots + j
                if this_filter < images.shape[0]:
                    this_img = images[this_filter]
                    spriteimage[i * img_h:(i + 1) * img_h,
                    j * img_w:(j + 1) * img_w] = this_img

        return spriteimage

    def vector_to_matrix_mnist(mnist_digits):
        """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
        return np.reshape(mnist_digits, (-1, 28, 28))

    def invert_grayscale(mnist_digits):
        """ Makes black white, and white black """
        return 1 - mnist_digits

    TO_EMBED_COUNT = 500

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    batch_xs, batch_ys = mnist.train.next_batch(TO_EMBED_COUNT)

    embedding_var = tf.Variable(batch_xs, name="mnistembedding")
    summary_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    embedding.metadata_path = os.path.join(TENSORBOARD_LOG_DIR, 'metadata.tsv')
    embedding.sprite.image_path = os.path.join(TENSORBOARD_LOG_DIR, 'mnistdigits.png')
    embedding.sprite.single_image_dim.extend([28, 28])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    to_visualise = batch_xs
    to_visualise = vector_to_matrix_mnist(to_visualise)
    to_visualise = invert_grayscale(to_visualise)

    sprite_image = create_sprite_image(to_visualise)
    plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')


def random_embedding(D=100, N=3):
    """
    https://www.tensorflow.org/versions/r1.1/get_started/embedding_viz
    """
    # Dimensionality of the embedding.
    embedding_var = tf.Variable(tf.random_normal([N, D]), name='word_embedding')

    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(TENSORBOARD_LOG_DIR, 'metadata.tsv')
    # Use the same TENSORBOARD_LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR)
    # The next line writes a projector_config.pbtxt in the TENSORBOARD_LOG_DIR. TensorBoard will read this file during startup.

    projector.visualize_embeddings(summary_writer, config)


if __name__ == '__main__':
    random_embedding()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(TENSORBOARD_LOG_DIR, '%s.ckpt' % __name__), 1)
