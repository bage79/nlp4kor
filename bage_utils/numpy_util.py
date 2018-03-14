import math

import numpy as np


class NumpyUtil(object):
    @staticmethod
    def cartesian_product(*arrays: np.ndarray):
        """
        # TODO: delete
        :param arrays:
        :return:
        """
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    @staticmethod
    def embeddings(dic_size, embeddings_size=10, min_val=-1., max_val=1., dtype=np.float32) -> np.ndarray:  # memory error, embeddings_size > 10
        """

        :param dic_size: row size
        :param embeddings_size: column size
        :param max_val:
        :param min_val:
        :param dtype:
        :return: array (dic_size, embedding_size)
        """
        if embeddings_size > 30:
            embeddings_size = 30
        nums_in_dim = int(np.ceil(math.log(dic_size, embeddings_size)))
        print('dic_size: %s -> embedding_size: %s -> nums_in_dim: %s -> W: (%s, %s)' % (dic_size, embeddings_size, nums_in_dim, dic_size, embeddings_size))
        dims = np.array([np.linspace(min_val, max_val, nums_in_dim, dtype=dtype) for _ in range(embeddings_size)])
        W = NumpyUtil.cartesian_product(*dims)
        return W[:dic_size]

    @staticmethod
    def combinations(arrays):
        return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


if __name__ == '__main__':
    print(NumpyUtil.combinations([['a', 'b', 'c'], [4, 5], [6, 7]]))
    # dic_size = 16000
    # W = NumpyUtil.embeddings(dic_size=15232, embeddings_size=15)
    # print(W)
    # print(W.shape)
    # print(W.shape)
    # W = np.reshape(W, (-1, embedding_size))
    # W = W[:dic_size]
    # print(W)
    # print(W.shape)
