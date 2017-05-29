import collections


class DictUtil(object):
    @staticmethod
    def sort_by_key(di, reverse=False):  # when reverse=False, ASC sort
        return collections.OrderedDict(sorted(di.items(), reverse=reverse))

    @staticmethod
    def sort_by_value(di, reverse=False):  # when reverse=False, ASC sort
        return collections.OrderedDict(sorted(di.items(), key=lambda t: t[1], reverse=reverse))

    @staticmethod
    def strip_values(di):
        for k, v in di.items():
            di[k] = str(v).strip()
        return di


if __name__ == '__main__':
    d = {'2': 'b', '1': '  c  ', '3': 'a'}
    print([v for v in DictUtil.sort_by_key(d).values()])
    # print(DictUtil.sort_by_key(d).keys())
    # print(DictUtil.sort_by_value(d))
    # print(DictUtil.sort_by_value(d).values())
    # print(d)
    # print(DictUtil.strip_values(d))
