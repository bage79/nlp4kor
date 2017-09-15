import codecs
import os.path
import subprocess

import chardet

from bage_utils.num_util import NumUtil


class FileUtil(object):
    def __init__(self, data=None):
        self.data = data

    def __repr__(self):
        return self.data

    def read(self, file_path, is_binary=False):
        self.data = FileUtil.reads(file_path, is_binary)
        return self.data

    def write(self, file_path, is_binary=False):
        FileUtil.writes(self.data, file_path, is_binary)

    @staticmethod
    def delete(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def encoding(file_path):
        try:
            return chardet.detect(open(file_path).read())['encoding']
        except:
            return None

    @staticmethod
    def reads(file_path, is_binary=False):
        if is_binary:
            read_mode = 'rb'
            with open(file_path, mode=read_mode) as f:
                data = ''.join([line for line in f.readlines()])
            return data
        else:  # text data
            read_mode = 'r'
            charset = FileUtil.encoding(file_path)
            with codecs.open(file_path, mode=read_mode, encoding=charset) as f:
                data = ''.join([line for line in f.readlines()])
            return data

    @staticmethod
    def writes(data, file_path, is_binary=False, encoding='UTF-8'):
        d = os.path.dirname(file_path)
        if len(d) > 0 and not os.path.exists(d):
            os.makedirs(d)
        if is_binary:
            write_mode = 'wb'
            if data and len(data) > 0:
                with open(file_path, mode=write_mode) as f:
                    f.write(data)
        else:  # text data
            write_mode = 'w'
            if data and len(data) > 0:
                with codecs.open(file_path, mode=write_mode, encoding=encoding) as f:
                    f.write(data)

    @staticmethod
    def count_lines(filename_or_list):
        lines = 0
        if isinstance(filename_or_list, str):
            filename_list = [filename_or_list]
        else:
            filename_list = filename_or_list

        for filename in filename_list:
            if filename.endswith('.gz') or filename.endswith('.zip'):
                gzip_format = True
            else:
                gzip_format = False

            if gzip_format:
                p = subprocess.Popen('gzip -cd %s | wc -l' % filename, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            else:
                p = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            result, err = p.communicate()
            if p.returncode != 0:
                raise IOError(err)
            lines += int(result.strip().split()[0])
        return lines

    @staticmethod
    def print_n_write(file, s):
        print(s)
        file.write('%s\n' % (s,))

    @staticmethod
    def to_filename(s):
        return s.replace('/', '').replace('"', "'")

    @staticmethod
    def to_filename_from_dict(di, delimeter='_', include=None):
        if include is None:
            include = di.keys()
        return delimeter.join(
            ['{}={}'.format(key.replace(delimeter, ''), NumUtil.to_readable(val)) for key, val in sorted(di.items()) if
             key in include])

    @staticmethod
    def postfix(path, postfix):
        file_name, file_extension = os.path.splitext(path)
        return '%s%s%s' % (file_name, postfix, file_extension)


if __name__ == '__main__':
    name_max = subprocess.check_output("getconf NAME_MAX /", shell=True)
    path_max = subprocess.check_output("getconf PATH_MAX /", shell=True)
    print('name_max:', name_max)  # 255
    print('path_max:', path_max)  # 4096

    with open('output/file_test.txt', 'w') as file:
        FileUtil.print_n_write(file, 'aaa')
        FileUtil.print_n_write(file, ('a', 'b', 'c'))
        # print(FileUtil().read('/root/Documents/pn-a.txt'))
        # data = FileUtil.reads(__file__)
        # out_file_path = base_util.real_path('output/%s.bak' % os.path.basename(__file__))
        # FileUtil.writes(data, out_file_path, is_binary=False)
        # FileUtil().read(__file__).write(base_util.real_path('output/%s.bak' % __file__))
