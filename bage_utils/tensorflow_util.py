import tensorflow as tf


class TensorflowUtil(object):
    @classmethod
    def variables_list(cls, sess):
        li = []
        for var in tf.all_variables():
            li.append((var.name, sess.run(var)))
        return li
