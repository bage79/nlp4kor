import base64
import os.path

from bage_utils import base_util
from bage_utils.file_util import FileUtil


class Base64Util(object):
    """ encode/decode with Base64 """

    def __init__(self, s=''):
        self.s = s

    def __repr__(self, *args, **kwargs):
        return self.s

    @staticmethod
    def encodes(s):
        return base64.b64encode(s)

    @staticmethod
    def decodes(s):
        return base64.b64decode(s)

    def encode(self):
        self.s = Base64Util.encodes(self.s)
        return self.s

    def decode(self):
        self.s = Base64Util.decodes(self.s)
        return self.s


if __name__ == '__main__':
    in_file_path = base_util.real_path("input/Penguins.jpg")
    out_data_file_path = base_util.real_path("output/Penguins.jpg.base64.txt")
    out_file_path = base_util.real_path("output/Penguins.jpg")
    if os.path.exists(out_file_path):
        os.remove(out_file_path)
    out_data = Base64Util.encodes(FileUtil.reads(in_file_path, is_binary=True))
    FileUtil.writes(out_data, out_data_file_path)
    out_file = Base64Util.decodes(out_data)
    FileUtil.writes(out_file, out_file_path)

    _in = '박'
    out = Base64Util('박').encode()
    print('Base64Util.encode(%s) -> %s' % (_in, out))
    print('Base64Util.decode(%s) -> %s' % (out, Base64Util(out).decode()))
