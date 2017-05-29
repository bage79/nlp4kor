import math

"""
https://github.com/taki0112/Vector_Similarity/blob/master/python/TS_SS/Vector_Similarity.py
"""


class VectorSimUtil(object):
    """
    various methods for `Vector Similarity`.
    """

    @staticmethod
    def Cosine(vec1, vec2):
        result = VectorSimUtil.InnerProduct(vec1, vec2) / (
            VectorSimUtil.VectorSize(vec1) * VectorSimUtil.VectorSize(vec2))
        return result

    @staticmethod
    def VectorSize(vec):
        return math.sqrt(sum(math.pow(v, 2) for v in vec))

    @staticmethod
    def InnerProduct(vec1, vec2):
        return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

    @staticmethod
    def Euclidean(vec1, vec2):
        return math.sqrt(sum(math.pow((v1 - v2), 2) for v1, v2 in zip(vec1, vec2)))

    @staticmethod
    def Theta(vec1, vec2):
        return math.acos(VectorSimUtil.Cosine(vec1, vec2)) + 10

    @staticmethod
    def Triangle(vec1, vec2):
        theta = math.radians(VectorSimUtil.Theta(vec1, vec2))
        return (VectorSimUtil.VectorSize(vec1) * VectorSimUtil.VectorSize(vec2) * math.sin(theta)) / 2

    @staticmethod
    def Magnitude_Difference(vec1, vec2):
        return abs(VectorSimUtil.VectorSize(vec1) - VectorSimUtil.VectorSize(vec2))

    @staticmethod
    def Sector(vec1, vec2):
        ED = VectorSimUtil.Euclidean(vec1, vec2)
        MD = VectorSimUtil.Magnitude_Difference(vec1, vec2)
        theta = VectorSimUtil.Theta(vec1, vec2)
        return math.pi * math.pow((ED + MD), 2) * theta / 360

    @staticmethod
    def TS_SS(vec1, vec2):
        return VectorSimUtil.Triangle(vec1, vec2) * VectorSimUtil.Sector(vec1, vec2)


if __name__ == '__main__':
    vec1 = (1, 2)

    for vec2 in [
        (1, 3),
        (1, 2),
        (2, 4),
        (4, 8)
    ]:
        print('(%s %s): euclidean: %.3f, cos:%.3f, ts-ss: %.3f' %
              (vec1, vec2, VectorSimUtil.Euclidean(vec1, vec2), VectorSimUtil.Cosine(vec1, vec2),
               VectorSimUtil.TS_SS(vec1, vec2)))
