import hashlib


class EncryptUtil(object):
    @staticmethod
    def md5(s):
        """
        md5
        :param s: string
        :return: 32 characters (128bits=16bytes)
        """
        return hashlib.md5(s.encode()).hexdigest()

    @staticmethod
    def sha1(s):
        """
        sha1
        :param s: string
        :return: 40 characters (160bits=20bytes)
        """
        return hashlib.sha1(s.encode()).hexdigest()

    @staticmethod
    def sha256(s):
        """
        sha1
        :param s: string
        :return: 64 characters (256bits=32bytes)
        """
        return hashlib.sha256(s.encode()).hexdigest()

    @staticmethod
    def sha512(s):
        """
        sha1
        :param s: string
        :return: 128 characters (512bits=64bytes)
        """
        return hashlib.sha512(s.encode()).hexdigest()


if __name__ == '__main__':
    print(len('한글'.encode()))
    h = EncryptUtil.sha512('hello')
    print(len(h), h)
