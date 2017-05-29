import gzip
import os


class DataFileUtil(object):
    @staticmethod
    def read_list(filepath, gzip_format=False):
        if not os.path.exists(filepath):
            raise Exception('"%s" not exists.' % filepath)

        try:
            if gzip_format:
                with gzip.open(filepath, 'rt') as f:
                    values = [c.strip() for c in f]  # e.g. ['ㄱ', 'ㄴ', 'ㄷ', ...]
            else:
                with open(filepath, 'r') as f:
                    values = [c.strip() for c in f]  # e.g. ['ㄱ', 'ㄴ', 'ㄷ', ...]
            return values
        except:
            return []
