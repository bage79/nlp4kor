# from geventhttpclient import httplib; httplib.patch()
# from geventhttpclient.url import URL #@UnusedImport
# from geventhttpclient.client import HTTPClient #@UnusedImport
from urllib.parse import urlencode

from httplib2 import Http


class HttpClientUtil(object):
    DEFAULT_HEADERS = {
        'Accept': 'Accept:text/html,application/xhtml+xml,application/xml;q=0.9,*/*;',
        'Accept-Encoding': 'gzip,deflate',
        'Cache-Control': 'max-age=0',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    @classmethod
    def request(cls, uri, method='GET', postdata=None, cookie='', timeout=10):
        if not postdata:
            postdata = {}
        if method not in ['GET', 'POST']:
            return {}, ''
        h = Http(cache=None, timeout=timeout)
        headers = dict(cls.DEFAULT_HEADERS)
        headers.update(headers or {})
        if method.upper() == 'GET':
            postdata = None
        if method.upper() == 'POST':
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
            postdata = urlencode(postdata)
        if len(cookie) > 0:
            headers['Cookie'] = cookie.replace(',', ';')
        headers, content = h.request(uri, method=method, body=postdata, headers=headers)
        if 'set-cookie' in headers.keys():
            headers['set-cookie'] = headers['set-cookie'].replace(',', ';')
        return headers, content

    @staticmethod
    def parse_http_response(http_resp):
        headers = {}
        _header, content = http_resp.split('\r\n\r\n', 1)
        for line in _header.split('\r\n'):
            line = line.strip()
            try:
                k, v = line.split(':')
                headers[k.strip()] = v.strip()
            except:
                if line.startswith('HTTP'):
                    http_version, status, _ = line.split(' ')
                    headers['http_version'] = http_version
                    headers['status'] = status
        return headers, content


if __name__ == '__main__':
    query_string = r'a=1&b=2&c=8&c=9'
    print(HttpClientUtil.get_param_list('x', query_string))
    # headers, content = HttpClientUtil.request('http://www.naver.com')
    # print(headers, content
