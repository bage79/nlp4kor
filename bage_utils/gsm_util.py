class GsmUtil(object):
    gsm_unicode = (
        u"@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞ\x1bÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ¿abcdefghijklmnopqrstuvwxyzäöñüà")
    gsm_ext_unicode = u"^{}\\[~]|€"
    gsm_all_unicode = gsm_unicode + gsm_ext_unicode

    @staticmethod
    def encode_to_gsm(s, out_of_range_char_to='?', charset='utf8'):
        if isinstance(s, str):
            str_type = True
        # elif isinstance(s, unicode):
        #     str_type = False
        else:
            return s

        table = GsmUtil.gsm_all_unicode
        # s = unicode(s)
        res = ""
        for c in s:
            idx = table.find(c)
            if idx != -1:
                res += c
                continue
            else:
                res += out_of_range_char_to

        if str_type:
            return res.encode(charset)
        else:
            return res


def main():
    # print('Ç' * 160  # gsm
    # print('Î' * 160  # no gsm
    # print('Î' * 67  # no gsm
    input_string = r"""abcdef...z 0123456789 ÎÂÇÅÄÛşŞßÜÖîâçåäûüöĞğẞİıÈÉÊËÛÙÏÎÀÂÔèéêëûùïîàâôç ~!`&*;):)<>[]{}><=÷×+'"/ | \ \\ ※♂♀¿ ♡☆ 한글은 안 보임"""
    print(input_string)
    print(GsmUtil.encode_to_gsm(input_string))


if __name__ == '__main__':
    main()
