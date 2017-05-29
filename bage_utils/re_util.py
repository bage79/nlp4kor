import re


class ReUtil(object):
    """
    - wrapper of python `re` module.
    """

    @staticmethod
    def displaymatch(match):
        if match is None:
            return None
        return '<Match: %r, groups=%r>' % (match.group(), match.groups())

    @staticmethod
    def remove_first_chars(s, ch):
        return re.sub(r'^' + ch + '*', '', s)

    @staticmethod
    def remove_last_chars(s, ch):
        return re.sub(r'' + ch + r'.*$', '', s)

    @staticmethod
    def is_mobile_number(s):
        patten = re.compile(r'^\d{3}-\d{3,4}-\d{4}$')
        if re.match(patten, s):
            return True
        return False

    @staticmethod
    def is_phone_number(s):
        pattern = re.compile(r'^\d{2,3}-\d{3,4}-\d{4}$')
        if re.match(pattern, s):
            return True
        return False

    @staticmethod
    def is_email_address(s):
        pattern = re.compile(r'^[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*\.[a-zA-Z]{2,3}$')
        if re.match(pattern, s):
            return True
        return False

    @staticmethod
    def is_userid(s):
        pattern = re.compile(r'^[a-z0-9_]{4,20}$')
        if re.match(pattern, s):
            return True
        return False


if __name__ == '__main__':
    print(ReUtil.remove_first_chars(',,,a,b,c', ','))
    print(ReUtil.remove_last_chars('a,b,c,,,,', ','))

    numbers = ["010-7324-7942", "017-123-4567", "00-00-0000", "abc-1111-1111"]
    for number in numbers:
        print("%s -> %s" % (number, ReUtil.is_mobile_number(number)))
