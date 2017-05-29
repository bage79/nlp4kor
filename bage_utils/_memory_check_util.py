"""
need more test..
"""
import sys
import types


class MemoryCheckUtil(object):
    @staticmethod
    def __get_refcounts():
        d = {}
        # sys.modules
        # collect all classes
        for m in sys.modules.values():
            for sym in dir(m):
                o = getattr(m, sym)
                if type(o) is types.DynamicClassAttribute:
                    d[o] = sys.getrefcount(o)
        # sort by refcount
        pairs = map(lambda x: (x[1], x[0]), d.items())
        pairs.sort()
        pairs.reverse()
        return pairs

    @staticmethod
    def print_refcounts_top_100():
        print('print_top_100')
        for n, c in MemoryCheckUtil.__get_refcounts()[:100]:
            print('%10d %s' % (n, c.__name__))


if __name__ == '__main__':
    MemoryCheckUtil.print_refcounts_top_100()
