class QueueUtil(object):
    """
    - queue implements by python list.
    """

    def __init__(self, li=None):
        if isinstance(li, list) or isinstance(li, tuple):
            self.li = list(li)
        elif li is None:
            self.li = []
        else:
            self.li = [li]

    def push(self, item):
        self.li.append(item)
        return self.li

    def pop(self):
        return self.li.pop(0)

    def __repr__(self):
        return repr(self.li)


if __name__ == '__main__':
    q = QueueUtil(range(10))
    print(q)
    print(q.push(10))
    print(q.pop())
    print(q)
