class Dic2Object(object):
    def __init__(self, __dic=None):
        if not __dic:
            __dic = {}
        self.__dict__.update(__dic)

    def __repr__(self):
        return str(self.__dict__)


if __name__ == '__main__':
    di = {'pk': 123, 'name': '홍길동', 'email': 'kildong.hong@gmail.com'}
    obj = Dic2Object(di)
    # print(type(di), di)
    print(type(obj), obj)
    print(Dic2Object(None))

    # print(obj.pk, obj.name, obj.email
    # print(obj.__dict__
