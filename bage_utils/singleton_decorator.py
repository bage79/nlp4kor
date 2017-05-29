def singleton(class_):
    """
    - decorator for creating singleton instance.
    :param class_: 
    :return: 
    """
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


if __name__ == '__main__':
    @singleton
    class A(object):
        pass


    a1 = A()
    a2 = A()
    print(id(a1), id(a2))
