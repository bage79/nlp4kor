"""
original: https://github.com/sublee/korean/blob/master/korean/hangul.py (sub@subl.ee)
modified: https://github.com/bage79/nlp4kor/blob/master/bage_utils/hangul_util.py (bage79@gmail.com)
"""
import random
import re
import traceback
import warnings

import numpy

from bage_utils.string_util import StringUtil


def to_sequence(*sequences):
    def to_tuple(sequence):
        if not sequence:
            return sequence,
        return tuple(sequence)

    return sum(map(to_tuple, sequences), ())


def _to_one_hot_vector(c: str, li=None) -> numpy.ndarray:
    """
    :param c: 문자 
    :param li: 문자를 찾을 리스트 e.g. ['ㄱ', 'ㄴ', 'ㄷ', ...]
    :return: one hot vector (numpy.ndarray)
    """
    if li is None:
        li = []

    try:
        idx = li.index(c)
    except:
        raise Exception('%s not found in list' % c)

    a = numpy.zeros(len(li), dtype=numpy.int32)
    a[idx] = 1
    return a


warnings.filterwarnings("ignore", category=FutureWarning, append=1)  # for warning in re
# han_eng_re = re.compile('[^ㄱ-ㅎ가-힣0-9a-zA-Z~!@#=\^\$%&_\+~:\";\',\.?/\(\)\{\}\[\]\\s]')
han_eng_re = re.compile('[^ㄱ-ㅎ가-힣0-9a-zA-Z~!@#=^$%&_+:\";\',.?/(){\}\[\]\\s]')


# noinspection PyGlobalUndefined
class HangulUtil(object):
    """
    original: https://github.com/sublee/korean/blob/master/korean/hangul.py
    modified: bage79@gmail.com
    """

    IS_NOT_HANGUL = -1
    global to_sequence

    JA_LIST = to_sequence(u'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')  # 자음
    MO_LIST = to_sequence(u'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')  # 모음
    WANSUNG_LIST = tuple([chr(c) for c in range(ord(u'가'), ord(u'힣') + 1)])  # 완성형

    JA_RANGE = [ord(c) for c in JA_LIST]  # 자음 범위
    MO_RANGE = [ord(c) for c in MO_LIST]  # 모음 범위
    WANSUNG_RANGE = range(ord(u'가'), ord(u'힣') + 1)  # 완성형

    HANGUL_LIST = list(set(list(MO_LIST) + list(JA_LIST) + [chr(c) for c in WANSUNG_RANGE]))
    HANGUL_LIST.sort()
    HANGUL_FIRST = WANSUNG_RANGE[0]

    CHAR2ONE_HOT = list()

    CHO_LIST = to_sequence(u'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')  # 초성
    JUNG_LIST = MO_LIST  # 중성
    JONG_LIST = to_sequence(u'', u'ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ')  # 종성

    CHO_NOISE_LIST = to_sequence(u'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ')  # 초성
    JUNG_NOISE_LIST = to_sequence(u'ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ')  # 모음
    JONG_NOISE_LIST = to_sequence(u'', u'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ')  # 종성

    CHO_JUNG_JONG_LIST = (CHO_LIST, JUNG_LIST, JONG_LIST)  # 초성 중성 종성
    CHO_RANGE = [ord(c) for c in CHO_LIST if len(c) > 0]  # 초성 범위
    JUNG_RANGE = [ord(c) for c in JUNG_LIST if len(c) > 0]  # 중성 범위
    JONG_RANGE = [ord(c) for c in JONG_LIST if len(c) > 0]  # 종성 범위

    HANJA_RANGE = range(ord(u'一'), ord(u'龥') + 1)
    HANJA_LIST = [c for c in HANJA_RANGE]

    ENGLISH_LOWER_RANGE = range(ord(u'a'), ord(u'z') + 1)
    ENGLISH_UPPER_RANGE = range(ord(u'A'), ord(u'Z') + 1)
    ENGLISH_LIST = [chr(c) for c in ENGLISH_LOWER_RANGE] + [chr(c) for c in ENGLISH_UPPER_RANGE]

    NUM_LIST = [str(c) for c in range(10)]
    KEYBOARD_SPECIAL_LIST = list(to_sequence(r"""`~!@#$%^&*()_+-+[]\;',./{}|:"<>?"""))

    CHAR_LIST = HANGUL_LIST + ENGLISH_LIST + NUM_LIST + KEYBOARD_SPECIAL_LIST

    KEYBOARD_ENG_LIST = 'qwertyuiopasdfghjklzxcvbnm' + 'QWERTOP'
    KEYBOARD_HAN_LIST = 'ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡ' + 'ㅃㅉㄸㄲㅆㅒㅖ'
    KEYBOARD_ENG_TO_HAN = dict(zip(KEYBOARD_ENG_LIST, KEYBOARD_HAN_LIST))

    del to_sequence

    @classmethod
    def to_cho_jung_jong_vector(cls, c: str):
        """
        초성(19) + 중성(21) + 종성(28) + 영어(52) + 숫자(10) + 특수문자32)
        :param c: 
        :return: 
        """
        hangul_vector = numpy.zeros(len(cls.CHO_LIST) + len(cls.JUNG_LIST) + len(cls.JONG_LIST))
        english_vector = numpy.zeros(len(cls.ENGLISH_LIST))
        num_vector = numpy.zeros(len(cls.NUM_LIST))
        special_vector = numpy.zeros(len(cls.KEYBOARD_SPECIAL_LIST))

        if cls.is_hangul_char(c):
            cho, jung, jong = cls.split2cho_jung_jong(c)
            if len(cho) == 1:
                cho_vector = _to_one_hot_vector(cho, cls.CHO_LIST)
            else:
                cho_vector = numpy.zeros(len(cls.CHO_LIST))
            if len(jung) == 1:
                jung_vector = _to_one_hot_vector(jung, cls.JUNG_LIST)
            else:
                jung_vector = numpy.zeros(len(cls.JUNG_LIST))
            if len(jong) == 1:
                jong_vector = _to_one_hot_vector(jong, cls.JONG_LIST)
            else:
                jong_vector = numpy.zeros(len(cls.JONG_LIST))
            hangul_vector = numpy.concatenate((cho_vector, jung_vector, jong_vector))
        elif c in cls.ENGLISH_LIST:
            english_vector = _to_one_hot_vector(c, cls.ENGLISH_LIST)
        elif c in cls.NUM_LIST:
            num_vector = _to_one_hot_vector(c, cls.NUM_LIST)
        elif c in cls.KEYBOARD_SPECIAL_LIST:
            special_vector = _to_one_hot_vector(c, cls.KEYBOARD_SPECIAL_LIST)

        return numpy.concatenate((hangul_vector, english_vector, num_vector, special_vector))

    # noinspection PyUnresolvedReferences
    @classmethod
    def to_one_hot_vector(cls, c: str):
        return cls.CHAR2ONE_HOT.get(c, None)

    @classmethod
    def to_one_hot_index(cls, c: str):
        try:
            return cls.CHAR_LIST.index(c)
        except:  # not found
            return -1

    @classmethod
    def load(cls):
        cls.CHAR2ONE_HOT = dict([(c, _to_one_hot_vector(c, li=cls.CHAR_LIST)) for c in cls.CHAR_LIST])
        # print('keys:', cls.HANGUL_ONE_HOT.keys())

    @classmethod
    def to_cho_index(cls, c: str):
        try:
            return cls.CHO_LIST.index(c)
        except:  # not found
            return -1

    @classmethod
    def to_jung_index(cls, c: str):
        try:
            return cls.JUNG_LIST.index(c)
        except:  # not found
            return -1

    @classmethod
    def to_jong_index(cls, c: str):
        try:
            return cls.JONG_LIST.index(c)
        except:  # not found
            return -1

    @staticmethod
    def remain_han_eng(s):
        """
        한글, 영어, 숫자, 일부 기호만 남기고 제거
        :param s:
        :return:
        """
        return ' '.join(han_eng_re.sub(' ', s).split(' ')).strip()

    @classmethod
    def has_hangul(cls, word: object) -> object:
        """
        :param word: 단어
        :return: 한글 음절을 포함하는지 여부
        """
        try:
            for char in word:
                if (ord(char) in cls.WANSUNG_RANGE) or (ord(char) in cls.JA_RANGE) or (ord(char) in cls.MO_RANGE):
                    return True
            return False
        except:
            print(traceback.format_exc())

    @classmethod
    def has_english(cls, word):
        """
        :param word: 단어
        :return: 영어 음절을 포함하는지 여부
        """
        try:
            for char in word:
                if ord(char) in cls.ENGLISH_LOWER_RANGE or ord(char) in cls.ENGLISH_UPPER_RANGE:
                    return True
            return False
        except:
            print(traceback.format_exc())

    @classmethod
    def __char_offset(cls, char):
        """
        :param char: 음절
        :return: offset from u"가".
        """
        if isinstance(char, int):
            offset = char
        else:
            if len(char) != 1:
                return cls.IS_NOT_HANGUL
            elif not cls.is_full_hangul(char):
                return cls.IS_NOT_HANGUL
            offset = ord(char) - cls.HANGUL_FIRST

        if offset >= len(cls.WANSUNG_RANGE):
            return cls.IS_NOT_HANGUL
        else:
            return offset

    @classmethod
    def is_full_hangul(cls, word, exclude_chars='.,'):
        """
        :param exclude_chars: 예외로 둘 문자들 (공백은 기본 포함)
        :param word: 단어
        :return: 모든 음절이 한글인지 여부
        """
        try:
            for char in word:
                if char != ' ' and (char not in exclude_chars) and (ord(char) not in cls.WANSUNG_RANGE):
                    return False
            return True
        except:
            print(traceback.format_exc())

    @classmethod
    def is_full_hangul_or_english(cls, word, exclude_chars=''):
        """
        :param exclude_chars: 예외로 둘 문자들 (공백은 기본 포함)
        :param word: 단어
        :return: 모든 음절이 한글 또는 영어인지 여부
        """
        # print('is_full_hangul(%s)' % word)
        # print('is_full_hangul;', id(log), log.level)
        try:
            for char in word:
                if char != ' ' and (char not in exclude_chars) and (ord(char) not in cls.WANSUNG_RANGE) and (ord(char) not in cls.ENGLISH_UPPER_RANGE) and (
                            ord(char) not in cls.ENGLISH_LOWER_RANGE):
                    return False
            return True
        except:
            print(traceback.format_exc())

    @classmethod
    def has_hangul(cls, word):
        """
        :param word: 단어
        :return: 한글 음절을 포함하는지 여부
        """
        try:
            for char in word:
                if ord(char) in cls.WANSUNG_RANGE:
                    return True
            return False
        except:
            print(traceback.format_exc())

    @classmethod
    def has_hanja(cls, word):
        """
        :param word: 단어
        :return: 한자 음절을 포함하는지 여부
        """
        try:
            for char in word:
                if ord(char) in cls.HANJA_RANGE:
                    return True
            return False
        except:
            print(traceback.format_exc())

    @classmethod
    def is_hangul_char(cls, char):
        """

        :param char: 음절
        :return: 한글 여부
        """
        if len(char) == 0:
            return False
        try:
            if char in cls.HANGUL_LIST:
                # if ord(char) in cls.WANSUNG_RANGE:
                return True
            return False
        except:
            print(traceback.format_exc())

    @classmethod
    def is_english_char(cls, char):
        """

        :param char: 음절
        :return: 영어 여부
        """
        if len(char) == 0:
            return False

        try:
            if ord(char) in cls.ENGLISH_LOWER_RANGE or ord(char) in cls.ENGLISH_UPPER_RANGE:
                return True
            return False
        except:
            print(traceback.format_exc())

    @classmethod
    def is_hanja_char(cls, char):
        """

        :param char: 음절
        :return: 한자 여부
        """
        if len(char) == 0:
            return False

        try:
            if ord(char) in cls.HANJA_RANGE:
                return True
            return False
        except:
            print(traceback.format_exc())

    @classmethod
    def is_moum(cls, char):
        """
        :param char: 음절
        :return: 모음 여부
        """
        return char in cls.MO_LIST

    @classmethod
    def is_jaum(cls, char):
        """
        :param char: 음절
        :return: 자음 여부
        """
        return char in cls.JA_LIST

    @classmethod
    def is_cho(cls, char):
        """
        :param char: 음절
        :return: 초성 여부
        """
        return char in cls.CHO_LIST

    @classmethod
    def is_jung(cls, char):
        """
        :param char: 음절
        :return: 종성 여부
        """
        return cls.is_moum(char)

    @classmethod
    def is_jong(cls, char):
        """
        :param char: 음절
        :return: 종성 여부
        """
        return char in cls.JONG_LIST

    @classmethod
    def get_cho(cls, char):
        """
        :param char: 음절
        :return: 초성
        """
        if cls.is_cho(char):
            return char
        # offset = int(__char_offset(char))
        # if offset == IS_NOT_HANGUL:
        #     return char
        elif cls.is_jung(char) or cls.is_jong(char):
            return ''
        return cls.CHO_LIST[int(cls.__char_offset(char) / int(len(cls.MO_LIST) * len(cls.JONG_LIST)))]

    @classmethod
    def get_jung(cls, char):
        """
        :param char: 음절
        :return: 중성
        """
        if cls.is_jung(char):
            return char
        elif cls.is_cho(char) or cls.is_jong(char):
            return ''
        return cls.MO_LIST[int(cls.__char_offset(char) / int(len(cls.JONG_LIST)) % len(cls.MO_LIST))]

    @classmethod
    def get_jong(cls, char):
        """
        :param char: 음절
        :return: 종성
        """
        if cls.has_jung(char) and cls.is_jong(char):
            return char
        elif cls.is_cho(char) or cls.is_jung(char):
            return ''
        else:
            return cls.JONG_LIST[cls.__char_offset(char) % len(cls.JONG_LIST)]

    @classmethod
    def has_cho(cls, char):
        """
        :param char: 음절
        :return: 초성을 포함하는지 (사용할 일이 있을까?)
        """
        if len(char) > 1:
            char = char[-1]
        return True if len(cls.get_cho(char)) else False

    @classmethod
    def has_jung(cls, char):
        """
        :param char: 음절
        :return: 중성을 포함하는지
        """
        if len(char) > 1:
            char = char[-1]
        return True if len(cls.get_jung(char)) else False

    @classmethod
    def has_jong(cls, char):
        """
        :param char: 음절
        :return: 종성을 포함하는지
        """
        if len(char) > 1:
            char = char[-1]
        return True if len(cls.get_jong(char)) else False

    @classmethod
    def split2cho_jung_jong(cls, char):
        """
        :param char: 음절
        :returns: tuple(초성, 중성, 종성)
        """
        cho = cls.get_cho(char)
        jung = cls.get_jung(char)
        jong = cls.get_jong(char)
        return [cho, jung, jong]

    @classmethod
    def join_cho_jung_jong(cls, cho, jung, jong=' '):
        """
        :param jong: 초성
        :param jung: 중성
        :param cho: 종성
        :return: 음절
        """
        if not (cho and jung):
            return cho or jung

        indexes = [tuple.index(*args) for args in zip(cls.CHO_JUNG_JONG_LIST, (cho, jung, jong))]
        offset = (indexes[0] * len(cls.MO_LIST) + indexes[1]) * len(cls.JONG_LIST) + indexes[2]
        return chr(cls.HANGUL_FIRST + offset)

    @classmethod
    def encode_noise(cls, char):
        jaso_list = cls.split2cho_jung_jong(char)
        x = random.randint(0, 2)

        if x == 0:
            target_jaso = cls.CHO_NOISE_LIST
        elif x == 1:
            target_jaso = cls.JUNG_NOISE_LIST
        else:
            target_jaso = cls.JONG_NOISE_LIST

        randidx = random.randint(0, len(target_jaso) - 1)
        jaso_list[x] = target_jaso[randidx]
        return cls.join_cho_jung_jong(*jaso_list)

    @classmethod
    def join_suffix(cls, char, jong_suffix):
        """
        :param jong_suffix: 음절
        :param char: 종성이면서 접미사 (ㄴ, ㄹ, ....)
        :return: 음절
        """
        cho, jung, jong = cls.split2cho_jung_jong(char)
        if cho != '' and jung != '' and jong == '':
            return cls.join_cho_jung_jong(cho, jung, jong_suffix)
        else:
            return ''.join([char, jong_suffix])

    @classmethod
    def split_string(cls, word):
        """
        :param word: 단어
        :return: list(음절)
        """
        li = []
        for char in word:
            if cls.is_full_hangul(char):
                li.extend(cls.split2cho_jung_jong(char))
            else:
                li.append(char)
        return li

    @classmethod
    def join_string(cls, char_list):
        """
        :param char_list: list(음절)
        :return: 단어
        """
        letters = []
        i = len(char_list) - 1
        try:
            while i >= 0:
                if cls.is_jong(char_list[i]):
                    try:
                        letters.insert(0, cls.join_cho_jung_jong(char_list[i - 2], char_list[i - 1], char_list[i]))
                        i -= 3
                    except:
                        try:
                            letters.insert(0, cls.join_cho_jung_jong(char_list[i - 1], char_list[i]))
                            i -= 2
                        except:
                            letters.insert(0, char_list[i])
                            i -= 1
                elif cls.is_jung(char_list[i]):
                    try:
                        letters.insert(0, cls.join_cho_jung_jong(char_list[i - 1], char_list[i]))
                        i -= 2
                    except:
                        letters.insert(0, char_list[i])
                        i -= 1
                else:
                    letters.insert(0, char_list[i])
                    i -= 1
            return u''.join(letters)
        except:
            return None

    # noinspection PyPep8Naming
    @classmethod
    def qwerty_to_hangul(cls, word):
        chars = []
        for char in word:
            if char in cls.KEYBOARD_ENG_TO_HAN:
                chars.append(cls.KEYBOARD_ENG_TO_HAN[char])
            else:
                chars.append(char)
        hangul = cls.join_string(chars)
        if hangul:
            return hangul
        else:
            return word  # convert failed

    @staticmethod
    def get_except_hangul(value):
        """한글을 제외한 문자열을 리턴함"""
        chars = ''
        for i in value:
            if not HangulUtil.is_hangul_char(i):
                chars += i
        return chars

    @staticmethod
    def get_except_english(value):
        """영어를 제외한 문자열을 리턴함"""
        chars = ''
        for i in value:
            if not HangulUtil.is_english_char(i):
                chars += i
        return chars

    @classmethod
    def text2sentences(cls, text: str, sentence_delim='다.', remove_only_one_word=True, has_hangul=True):
        if isinstance(text, str) and len(sentence_delim) > 0:  # split lines with delimiter
            sentences = []
            lines = text.split(sentence_delim)
            for i, s in enumerate(lines):
                s = StringUtil.merge_to_one_line(s)
                if remove_only_one_word and s.count(' ') == 0:  # only one word in a sentence
                    continue
                if has_hangul and not HangulUtil.has_hangul(s):
                    continue

                if i + 1 != len(lines):
                    s += sentence_delim
                sentences.append(s)
            return sentences
        else:
            return []


HangulUtil.load()

if __name__ == '__main__':
    # print('"%s"' % str(HangulUtil.CHO_LIST))
    # print('"%s"' % str(HangulUtil.JUNG_LIST))
    # print('"%s"' % str(HangulUtil.JONG_LIST))
    for c in ['한', '사']:
        jaso_list = HangulUtil.split2cho_jung_jong(c)
        _c = HangulUtil.encode_noise(c)
        print(c, '->', jaso_list, '->', _c)
    # text = r'''이 숫염소의 이름은 쾰른의 레전드 선수이자 나중에 명감독으로 꼽힌 헤네스 바이스바일러에서 유래하였다.
    # 숫염소 헤네스는 인근의 쾰른 서커스단으로부터 기증받은 것이다.
    # {{축구 클럽팀 정보2| 클럽 이름 = 1. FSV 마인츠 05| 풀 네임 = 1. Fußball- und Sport-Verein Mainz 05 e.V.| 별칭 = Die Nullfünfer (05년) Karnevalsverein (카니발 클럽)| 설립연도 = 1905년 3월 27일| 홈구장 = 코파스 아레나| 수용인원 = 33,500| 회장 = {{국기그림|독일}} 하랄트 슈트르츠| 스포르팅 매니저 = {{국기그림|독일}} 크리스티안 하이델| 감독 = {{국기그림|스위스}} 마틴 슈미트| 리그 = 1 분데스리가| 시즌 = 2013-14| 순위 = 7위| pattern_la1=_fsvmainz1415h| pattern_b1=_fsvmainz1415h| pattern_ra1=_fsvmainz1415h| pattern_sh1
    # | pattern_so1'''
    #     print(HangulUtil.text2sentences(text))
    # print(len(HangulUtil.HANGUL_LIST))
    # print(len(HangulUtil.to_one_hot_vector('ㄱ')))
    # print(len(HangulUtil.to_one_hot_vector('ㄴ')))
    # print(HangulUtil.text2sentences('사과나무()는 장미목 장미과 배나무아과 사과나무속에 딸린 종이다. 그 열매는 사과(沙果; 砂果)라 하며, 세계적으로 가장 널리 재배되는 과일 품종 가운데 하나이다. 사전적으로 평과(苹果)라고도 한다.'))
    # print(len(HangulUtil.CHAR2ONE_HOT))
    # print(HangulUtil.to_one_hot_vector('ㄲ'))
    # print(HangulUtil.to_one_hot_vector('?'))
    # print(HangulUtil.split_char('길'))
    # print(numpy.concatenate((numpy.zeros(3), numpy.ones(4))))
    # print(HangulUtil.to_cho_jung_jong_vector('ㄱ'))
    # print(HangulUtil.to_cho_jung_jong_vector('각'))
    # print(HangulUtil.to_cho_jung_jong_vector('?'))

    pass

# for ch in HangulUtil.HANGUL_RANGE:
#     print(chr(ch), )
# print(HangulUtil.HANGUL_LIST)
# print(HangulUtil.ENGLISH_LOWER_RANGE)
# print(HangulUtil.ENGLISH_UPPER_RANGE)
# print(HangulUtil.has_english('한f ㅎㅎㅎㅎㅎ'))
# print('ㅌㅌㅌㅌ')
# print('%s, %s, %s' % (HangulUtil.get_cho('ㄹ'), HangulUtil.get_jung('ㄹ'), HangulUtil.get_jong('ㄹ')))
# print('%s, %s, %s' % (HangulUtil.get_cho('ㅏ'), HangulUtil.get_jung('ㅏ'), HangulUtil.get_jong('ㅏ')))
# print('%s, %s, %s' % (HangulUtil.get_cho('라'), HangulUtil.get_jung('라'), HangulUtil.get_jong('라')))
# print('%s, %s, %s' % (HangulUtil.get_cho('란'), HangulUtil.get_jung('란'), HangulUtil.get_jong('란')))
# print(HangulUtil.has_jung('ㄹ'))
# print(HangulUtil.is_full_hangul('하 바'))
# print(HangulUtil.has_hanja('벌크선社'[-1]))
# print(HangulUtil.has_hanja('수출입銀'[-1]))
# print(HangulUtil.is_full_hangul('합병한다.'))
# print(HangulUtil.is_full_hangul('차세대-폐렴구균백신'))
# print(HangulUtil.is_full_hangul_or_english("""[헤럴드경제=박혜림 기자] '오늘의 날씨'만 철썩같이 믿고 어제보다 덜 추운 줄 알고 얇게 입고 나왔다가 낭패보신 경험 있으시죠?""", exclude_chars="""=[]\"'.?!"""))
# print(HangulUtil.is_full_hangul_or_english("""[헤럴드경제=박혜림 기자] '오늘의 날씨'만 철썩같이 믿고 어제보다 덜 추운 줄 알고 얇게 입고 나왔다가 낭패보신 경험 있으시죠?""", exclude_chars="""\"'.?!"""))
