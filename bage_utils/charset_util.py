# -*- coding: euc-kr -*-
import sys
import chardet


class CharsetUtil(object):
    """
    detect charset for only python2
    """

    @staticmethod
    def detect_encoding(s):  # FIXME: not work
        if sys.version_info.major == 2:
            if isinstance(s, str):
                return str, chardet.detect(s)['encoding'].lower()
            else:
                return type(s), None
        else:
            return type(s), None


if __name__ == '__main__':
    message = u'한글'.encode('euc-kr')
    _type, encoding = CharsetUtil.detect_encoding(message)
    print(_type, encoding, message)

    if encoding is not None:
        message = message.decode(encoding).encode('utf-8')
        _type, encoding = CharsetUtil.detect_encoding(message)
        print(_type, encoding, message)
