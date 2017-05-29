from urllib.parse import parse_qs

import HTMLParser
import lxml.html
import requests
from lxml.cssselect import CSSSelector
from lxml.html import fromstring


class HtmlParserUtil(object):
    s = requests.Session()
    h = HTMLParser.HTMLParser()

    def __init__(self, html_or_element, base_url=None, encoding=None):
        self.base_url = base_url
        self.encoding = encoding
        if isinstance(html_or_element, lxml.html.HtmlElement):
            #            print('HtmlElement'
            self.doc = html_or_element
        elif isinstance(html_or_element, str):
            self.doc = fromstring(html_or_element)
        else:
            raise Exception(r"""html_or_element must be 'str' or 'unicode' or 'lxml.html.HtmlElement'""")
        if base_url:
            self.doc.make_links_absolute(base_url)

    @classmethod
    def tostring(cls, element):
        return cls.h.unescape(lxml.html.tostring(element))

    def __repr__(self):
        return lxml.html.tostring(self.doc)

    @classmethod
    def set_headers(cls, di):
        cls.s.headers.update(di)

    @classmethod
    def get(cls, url, encoding='utf8', base_url=None):
        #        r = requests.get(url, allow_redirects=True, headers=headers)
        r = cls.s.get(url, allow_redirects=True)
        r.encoding = encoding
        #        print(r.headers
        #        print('elapsed:', r.elapsed
        return HtmlParserUtil(r.text, base_url, encoding)

    @staticmethod
    def select(css, must_has_text=False):
        sel = CSSSelector(css)
        if must_has_text:
            return [e for e in sel if e.text is not None]
        else:
            return [e for e in sel]

    @staticmethod
    def select_text(css):
        sel = CSSSelector(css)
        return [e.text for e in sel]

    @staticmethod
    def get_param(key, url_query, default_value=None):
        v_list = parse_qs(url_query)
        if key in v_list:
            return v_list.get(key)[0]
        return default_value

    @staticmethod
    def get_param_list(key, url_query, default_value=None):
        if not default_value:
            default_value = []
        v_list = parse_qs(url_query)
        if key in v_list:
            return v_list.get(key)
        return default_value


if __name__ == '__main__':
    p = HtmlParserUtil.get('http://www.naver.com/rules/service.html', base_url='http://www.naver.com', encoding='utf8')
    print(p.select_text('title')[0])
