import html
import re

import chardet
import lxml.html
import requests
from bs4 import BeautifulSoup
from lxml.html.clean import Cleaner

from bage_utils.string_util import StringUtil

html_tag_cleaner = Cleaner(allow_tags=[''], remove_unknown_tags=False)


class HtmlUtil(object):
    @staticmethod
    def remove_tags_in_html(html):
        # return re.sub(r'<.*?>', '', html, count=0, flags=re.DOTALL | re.IGNORECASE)
        return re.sub(r'<[a-zA-Z/][^>]*>', '', html, count=0, flags=re.DOTALL | re.IGNORECASE)

    @staticmethod
    def remove_comments_in_html(html):
        return re.sub(r'<!--.*?-->', '', html, count=0, flags=re.DOTALL | re.IGNORECASE)

    @staticmethod
    def remove_javascripts_in_doc(doc):
        for element in doc.iter("script"):
            element.drop_tree()
        return doc

    @staticmethod
    def remove_elements(doc, css_pattern_list, remove_string_list=[]):
        for p in css_pattern_list:  # remove unnecessary links
            for e in doc.select(p):
                if len(remove_string_list) > 0:
                    if e.text is not None:
                        for string in remove_string_list:
                            if string in e.text:
                                e.extract()
                else:
                    e.extract()
        return doc

    # noinspection PyUnresolvedReferences
    @staticmethod
    def trim(html, prefix_url=None):
        """
        코멘트 제거, 자바스크립트 제거 (100.daum.net 제외)
        \r\n -> \n
        html에 포함된 <br>, <p>를 \n 으로 변환
        다수의 공백, \t, \n 을 하나로 합침
        :param html:
        :param prefix_url:
        :return:
        """
        html = html.replace('\r\n', '\n')
        convert_dic = {'<br>': '\n', '<br/>': '\n', '<br />': '\n', '<p>': '\n', '<p/>': '\n', '<p />': '\n',
                       '<BR>': '\n', '<BR/>': '\n', '<BR />': '\n', '<P>': '\n', '<P/>': '\n', '<P />': '\n'}
        for _from, _to in convert_dic.items():
            html = html.replace(_from, _to)
        html = HtmlUtil.remove_comments_in_html(html)  # remove html comments.
        doc = lxml.html.document_fromstring(html)  # convert to html element.r

        if prefix_url:
            doc.make_links_absolute(prefix_url)  # convert links to absolute links.

        if prefix_url:
            if '100.daum.net' not in prefix_url:  # javascript를 지우면 일부가 안 보이는 HTML도 있다. (100.daum.net)
                doc = HtmlUtil.remove_javascripts_in_doc(doc)  # remove javascript elements.
        else:
            doc = HtmlUtil.remove_javascripts_in_doc(doc)  # remove javascript elements.

        html = lxml.html.tostring(doc, encoding='utf8', include_meta_content_type=True)  # convert to html string.
        html = html.decode('utf8')  # bytes -> string
        html = StringUtil.merge(html)  # replace multiple blanks to one blank.
        return html.strip()

    @staticmethod
    def unescape(_html):
        """
        &로 시작하는 HTML 문자를 원형으로 복원
        &gt; -> >
        &lt; -> <
        :param _html:
        :return:
        """
        return html.unescape(_html)

    @staticmethod
    def charset(_html: bytes) -> str:
        """
        <meta http-equiv="Content-Type" content="text/html; charset=euc-kr"> # HTML < 5.0
        <meta charset="euc-kr"> # HTML >= 5.0
        :param _html: r.content (r= requests.get())
        :return: charset of html
        """
        if type(_html) is not bytes:
            raise Exception('"html" must be bytes in method "charset(html)". (r.content)')

        try:
            soup = BeautifulSoup(_html, 'lxml')
            # e = soup.find('meta', {"http-equiv": 'Content-Type'})
            metas = soup.findAll('meta')
            for meta in metas:
                # print(meta.attrs)
                if 'http-equiv' in meta.attrs and meta.attrs['http-equiv'].lower() == 'content-type':  # HTML < 5.0
                    for s in meta.attrs['content'].split(';'):
                        s = s.strip().lower()
                        if s.startswith('charset='):
                            return s.replace('charset=', '').lower()
                elif 'charset' in meta.attrs:  # HTML >= 5.0
                    return meta.attrs['charset'].lower()
            raise Exception('Not found charset in meta tags')
        except:
            try:
                return chardet.detect(_html)['encoding'].lower()
            except:
                raise Exception('Not found charset by chardet package.')


if __name__ == '__main__':
    # print(HtmlUtil.remove_tags_in_html('<한글><a href="" target="_blank"><한글2>'))
    # print(HtmlUtil.unescape('&lt;역대 대통령&gt;'))
    urls = ['http://www.ajunews.com/view/20160725165138536', 'http://www.ajunews.com/view/20160715081732467']
    for url in urls:
        r = requests.get(url)
        print(url, HtmlUtil.charset(r.content))
        # print(HtmlUtil.trim(r.text, display_none=True))
