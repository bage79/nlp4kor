import math
import re
from decimal import Decimal


class NumUtil(object):
    """
    - number util
    """
    UK = 100000000  # 억

    @staticmethod
    def int2digit(n, base=10):  # base: 진수
        res = ''
        while n > 0:
            n, r = divmod(n, base)
            res = str(r) + res
        return res

    @staticmethod
    def comma_str(n, decimal_spaces=0):
        """
        :param n: 수치 (문자열 또는 수치형)
        :param decimal_spaces: 소수점 자리수
        """
        try:
            if n is None:
                return None
            if decimal_spaces > 0:
                n = float(n)
            else:
                n = int(float(n))

            if isinstance(n, int):
                return r'{:,d}'.format(n)
            elif isinstance(n, float):
                rule = r'{:,.%df}' % decimal_spaces
                return rule.format(n)
            else:
                return n
        except Exception:  # @UnusedVariable
            return n

    @staticmethod
    def auto_convert(num):
        try:
            return int(num)
        except:
            try:
                return float(num)
            except:
                try:
                    return str(num)
                except:
                    return num

    @staticmethod
    def to_digit(num):
        try:
            _num = ''
            for a in str(num):
                if ('0' <= a <= '9') or a in '-.':
                    _num += a
            _num = _num.lstrip('0')
            return int(_num)
        except:
            return 0

    @staticmethod
    def has_digit(num):
        for a in str(num):
            if a.isdigit():
                return True
        return False

    @staticmethod
    def remove_comma(line):
        return re.sub(r'''(\d),(\d{1})''', r'''\1\2''', line)

    @staticmethod
    def to_readable(n, min_decimal=0.001, max_decimal=1000):
        try:
            if (0 < math.fabs(n) < min_decimal) or (max_decimal < math.fabs(n)):
                return "{:.0e}".format(Decimal(n)).replace('+', '')  # scientific notation
            else:
                return n  # decimal notation
        except:
            return n

    @staticmethod
    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False


if __name__ == '__main__':
    # print(NumUtil.to_digit('-7,097,985.0원'))
    # print(NumUtil.has_digit('-a22a'))
    print(type(NumUtil.auto_convert('20160101')))
    print(type(NumUtil.auto_convert('2016-01-01')))
    print(type(NumUtil.auto_convert('+1')))
    print(type(NumUtil.auto_convert('-1')))
    print(type(NumUtil.auto_convert('-1.1')))
    print(NumUtil.to_digit('0001020'))
    print(NumUtil.to_digit('0000000'))
    #    print(NumUtil.int2digit(8)
    #    print(NumUtil.int2digit(8, 2)
    #    print(NumUtil.int2digit(8, 16)
    #    print(NumUtil.comma_str(-100000)
    # print(NumUtil.remove_comma('123,456,789'))
    # print(NumUtil.remove_comma('브렉시트 충격서 벗어나는 코스피…나흘째 올라 1,970선 회복'))
    # print(NumUtil.remove_comma('가,나,다 123,456,789.00'))
    # print(NumUtil.remove_comma('23,456,789.00억원'))
    # print(NumUtil.to_readable(0.001))
    # print(NumUtil.to_readable(0.0001))
    # print(NumUtil.to_readable(1000))
    # print(NumUtil.to_readable(10000))
    # print(NumUtil.to_readable('title'))
    # print(NumUtil.comma_str(123456789012345.1234))
    # print(NumUtil.comma_str(123456789012345.1234, 2))
    # print(NumUtil.comma_str(123456789012345, 1))
    # print(NumUtil.comma_str('123456789012345.1234'))
    # print(NumUtil.comma_str('123456789012345.1234', 2))
    # print(NumUtil.comma_str('123456789012345', 1))
    pass
