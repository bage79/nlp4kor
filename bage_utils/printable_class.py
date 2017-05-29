class PrintableClass(object):
    """
    - override `__repr__` method for printing or logging.
    """

    def __repr__(self, *args, **kwargs):
        return str(self.__dict__)


class MetaClass(type):
    def __repr__(self):
        return str({k: self.__dict__[k] for k in self.__dict__.keys() if not k.startswith('__')})


if __name__ == '__main__':
    class A(PrintableClass):
        __private_var = '__private_var'
        class_var = 'class_var'

        def __init__(self):
            self.a = 'A'

        def a_func(self):
            pass


    print(A())
