import inspect  # http://docs.python.org/2/library/inspect.html
from pprint import pprint

from bage_utils.dict_util import DictUtil  # @UnusedImport


class InspectUtil(object):
    @staticmethod
    def summary():
        frame = inspect.stack()[1]
        d = {'file': frame[1], 'line': frame[2], 'function': frame[3], 'code': frame[4]}
        return d

    @staticmethod
    def all():
        frame = inspect.stack()[1]
        d = {}
        for key in dir(frame[0]):
            d[key] = getattr(frame[0], key)
        return DictUtil.sort_by_key(d)

    @staticmethod
    def locals():
        frame = inspect.stack()[1]
        d = {}
        for key in frame[0].f_locals:
            d[key] = frame[0].f_locals[key]
        return DictUtil.sort_by_key(d)

    @staticmethod
    def globals():
        frame = inspect.stack()[1]
        d = {}
        for key in frame[0].f_globals:
            d[key] = frame[0].f_globals[key]
        return DictUtil.sort_by_key(d)


def __test():
    pprint(InspectUtil.summary())
    pprint(InspectUtil.locals())


if __name__ == '__main__':
    pprint(InspectUtil.summary())
    # __test()
