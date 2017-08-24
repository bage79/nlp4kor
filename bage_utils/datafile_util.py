import gzip
import os


class DataFileUtil(object):
    @staticmethod
    def read_list(filepath, gzip_format=False, strip=False):
        if not os.path.exists(filepath):
            raise Exception('"%s" not exists.' % filepath)

        try:
            if gzip_format:
                f = gzip.open(filepath, 'rt')
            else:
                f = open(filepath, 'r')

            with f:
                if strip:
                    values = [c.strip() for c in f.read().splitlines()]  # e.g. ['ㄱ', 'ㄴ', 'ㄷ', ...]
                else:
                    values = [c for c in f.read().splitlines()]
                return values
        except:
            return []
