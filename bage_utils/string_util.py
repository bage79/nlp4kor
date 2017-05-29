import re


class StringUtil(object):
    """
    - all about python string.
    """
    REGEX_TOKEN = r''' |=|\(|\)|\[|\]|<|>'''

    @staticmethod
    def mask_passwd_in_url(url):
        """
        e.g. mongodb://root:passwd@db-local:27017/... -> mongodb://root:******@db-local:27017/...
        :param url:
        :return:
        """
        try:
            a, b = url.index(r'://') + len(r'://'), url.index('@')
            head, auth, tail = url[:a], url[a:b], url[b:]
            if len(auth) > 0:
                user, passwd = auth.split(':')
                if len(passwd) > 0:
                    passwd = '*' * len(passwd)
                    auth = '%s:%s' % (user, passwd)
            return head + auth + tail
        except:
            return url

    @staticmethod
    def split_by_bracket(s, regex=REGEX_TOKEN):
        return [t for t in re.split(regex, s) if len(t) > 0]

    @staticmethod
    def from_set(_set):
        s = str(list(_set)).replace(' ', '')
        return s[1: len(s) - 1]

    @staticmethod
    def to_set(s, delim=','):
        return set(s.split(delim))

    @staticmethod
    def to_hex_list(s):
        return [c for c in map(ord, s)]

    # @classmethod
    # def to_hex(cls, char):
    #     return hex(ord(char))

    @staticmethod
    def to_hex_str_list(s):
        return ['{0:x}'.format(c) for c in map(ord, s)]

    @staticmethod
    def extract(s, start_s, end_s):
        if s and s.count(start_s) > 0 and s.count(end_s) > 0:
            start = s.find(start_s) + len(start_s)
            end = s.find(end_s, start)
            if start > 0 and end > 0:
                return s[start:end]
        return ''

    @staticmethod
    def split_by_size(s, size=100):
        lines = s.split('\n')
        result = ''
        for line in lines:
            if len(result) + len(line) > size:
                yield result
                result = line + '\n'
            else:
                result += line + '\n'
        yield result

    @staticmethod
    def merge(x, merge_tabs=False, merge_newlines=False):
        """
        TEXT인 경우, 공백, 탭, 줄바꿈 종류별로 merge를 한다.
        :param x:
        :param merge_tabs:
        :param merge_newlines:
        :return:
        """
        x = re.sub(' +', ' ', x)  # replace multiple blanks to one blank.
        if merge_tabs:
            x = re.sub(r'\t+', r'\t', x)
        if merge_newlines:
            x = re.sub(r'\n+', r'\n', x)
        return x.strip()

    @staticmethod
    def merge_to_one_line(x):
        """
        HTML인 경우, 다수의 공백, 탭, 줄바꿈을 유지할 필요가 없으므로, 모두 단일 공백으로 합쳐준다.
        :param x:
        :return:
        """
        return ' '.join(x.split()).strip()  # replace multiple blanks, tabs, new lines to one blank.

    @staticmethod
    def find_nth(s, x, n, i=0):
        i = s.find(x, i)
        if n == 1 or i == -1:
            return i
        else:
            return StringUtil.find_nth(s, x, n - 1, i + len(x))

    # @staticmethod
    # def split_and_recover(txt, delim):
    #     txt = txt.replace('\t', ' ')
    #     txt = txt.replace(delim, delim + '\t')
    #
    #     parts = txt.split('\t')
    #     return parts


    @staticmethod
    def remove_comment_line(text):
        lines = []
        for line in text.splitlines():
            if not line.startswith('#') and len(line) > 0:
                lines.append(line)
        return ' '.join(lines)


if __name__ == '__main__':
    print(StringUtil.mask_passwd_in_url('mongodb://root:passwd@db-local:27017/admin?authMechanism=MONGODB-CR'))
    print(StringUtil.mask_passwd_in_url('mongodb://root@db-local:27017/admin?authMechanism=MONGODB-CR'))
    print(StringUtil.mask_passwd_in_url('mongodb://db-local:27017/admin?authMechanism=MONGODB-CR'))
    print(StringUtil.mask_passwd_in_url('db-local:27017/admin?authMechanism=MONGODB-CR'))
    # print(StringUtil.split_by_bracket('(서울=포커스뉴스) 송민순 전 외교통상부 장관의 회고록 파문이 일파만파 커지며 정치권의 논란이 가열되고 있다.'))
    # print(StringUtil.merge_to_one_line('081116 스브스 인기가요 롱넘버 캡쳐       어쩌다보니 유천이가 없다..\t\t진짜 롱넘버는 레전드인듯T.T 멋지다구ㅠㅠㅠㅠ 나중에 끝에 가면 홍어삼합도아닌 삼합화음?ㅋㅋㅋㅋ 진심 그거대박 나 그부분 너무 좋음 ㅠㅠㅠ'))
    # for line in StringUtil.split_by_size('1234\n56\n7890', size=5):
    #     print(line)
    # print(StringUtil.merge('a    b  c\t\td\n\ne', merge_tabs=True, merge_newlines=True))
    # s1 = set([])
    # s2 = set([1])
    # s3 = set([1, 2, 3])
    # print(s1, StringUtil.from_set(s1))
    # print(s2, StringUtil.from_set(s2))
    # print(s3, StringUtil.from_set(s3))
    # print(StringUtil.to_hex_list('123'))
    # print(' '.join(StringUtil.to_hex_str_list('123')))
    # print(StringUtil.extract(r"""fn\.link\('(.+?)'""", r"""$$fn.link('/cate/329', '_self', '카테전체보기_에어컨'); return false;"""))
    # print(StringUtil.phone_number('010-2747-6116'))
    # print(StringUtil.phone_number('010+2747_6116'))
    pass
