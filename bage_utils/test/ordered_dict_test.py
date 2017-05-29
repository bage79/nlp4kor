from collections import OrderedDict

if __name__ == '__main__':
    dic = {
        'A': 1, 'B': 2, 'C': 3
    }
    ordered = OrderedDict(dic)
    for k in ordered.keys():
        print(k)
    for v in ordered.values():
        print(v)
