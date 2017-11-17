import tensorflow as tf


class TensorflowUtil(object):
    @classmethod
    def variables_list(cls, sess):
        li = []
        for var in tf.all_variables():
            li.append((var.name, sess.run(var)))
        return li

    @staticmethod
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    @staticmethod
    def turn_off_tensorflow_logging():
        import os
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tensorflow warnings
        tf.logging.set_verbosity(tf.logging.ERROR)  # ignore tensorflow info (GPU 할당 정보 확인)
