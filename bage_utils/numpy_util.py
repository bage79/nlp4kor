import numpy as np


def cartesian_product(*arrays: np.ndarray):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def embeddings(dic_size, nums_in_dim=10) -> np.ndarray:
    """

    :param dic_size:
    :param nums_in_dim:
    :return: array (dic_size, embedding_size)
    """
    embedding_size = int(np.ceil(np.log10(dic_size)))
    # print(dic_size, '-> embedding_size: ', embedding_size)
    dims = np.array([np.linspace(0, 0.9, nums_in_dim) for dim in range(embedding_size)])
    W = cartesian_product(*dims)
    return W[:dic_size]


if __name__ == '__main__':
    # dic_size = 16000
    W = embeddings(dic_size=150, nums_in_dim=10)
    print(W)
    print(W.shape)
    # print(W.shape)
    # W = np.reshape(W, (-1, embedding_size))
    # W = W[:dic_size]
    # print(W)
    # print(W.shape)
