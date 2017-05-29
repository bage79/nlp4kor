import os
import shutil
import traceback
from os import listdir
from os.path import join, isfile, isdir


class DirUtil(object):
    @staticmethod
    def count_files_recursive(input_dir, file_extension=None):
        count = 0
        try:
            print('input_dir: ' + input_dir)
            for filename in listdir(input_dir):
                input_path = join(input_dir, filename)

                if isdir(input_path):
                    print('%s:%s' % (input_path, DirUtil.count_files_recursive(input_path)))
                    count += DirUtil.count_files_recursive(input_path)
                if isfile(input_path):
                    if file_extension:
                        if filename.endswith(file_extension):
                            count += 1
                            print(input_path)
                    else:
                        count += 1

        except:
            traceback.print_exc()
        return count

    @staticmethod
    def mkdirs(path):
        try:
            if not os.path.isdir(path):
                os.makedirs(path)
        except:
            pass

    @staticmethod
    def rmdirs(path):
        if os.path.isdir(path):
            shutil.rmtree(path)


if __name__ == '__main__':
    print(DirUtil.count_files_recursive('/data/naver/terms.naver.com_2015.02.18', '.html'))
