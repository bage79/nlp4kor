import inspect


class Obj2DictUtil(object):
    """
    - create `dict` from python object.
    """

    @staticmethod
    def to_dict(obj, max_values=10, exclude_private_members=True):
        _di = {}
        attributes = inspect.getmembers(obj, lambda a: not (inspect.isroutine(a)))
        for k, v in attributes:
            if exclude_private_members and k.startswith('_'):
                continue
            v_type = type(v)
            if (v_type is list or v_type is set) and len(v) > max_values:
                v = v[:max_values]
            if (v_type is dict) and len(v) > max_values:
                v = list(v.items())[:max_values]
            _di[k] = v
        return _di
