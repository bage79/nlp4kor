class StackUtil(object):
    """
    - stack implements by python list.
    """

    def __init__(self, li=None):
        if isinstance(li, list) or isinstance(li, tuple):
            self.li = list(li)
        elif li is None:
            self.li = []
        else:
            self.li = [li]

    def put(self, item):
        self.li.append(item)
        return self.li

    def get(self):
        return self.li.pop()

    def __repr__(self):
        return repr(self.li)


if __name__ == '__main__':
    #    s = Stack([1, 2, 3])
    #    s = Stack((1, 2, 3))
    s = StackUtil(range(10))
    print(s.put(10))
    print(s.put(11))
    print(s.get())
