class Chunks(object):
    def __init__(self, data):
        self.data = data
        self.index = -1

    def __next__(self):
        self.index += 1
        try:
            return self.data[self.index]
        except IndexError:
            raise StopIteration

    def __iter__(self):
        return self

    def __repr__(self):
        return repr(self.data)
