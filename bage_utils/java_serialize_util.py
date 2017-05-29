import javaobj


class JavaSerializeUtil(object):
    @staticmethod
    def python2java(pobj):
        try:
            jobj = javaobj.dumps(pobj)
            return jobj
        except Exception as e:
            raise e

    @staticmethod
    def java2python(jobj):
        try:
            pobj = javaobj.loads(jobj)
            return pobj
        except Exception as e:
            raise e
