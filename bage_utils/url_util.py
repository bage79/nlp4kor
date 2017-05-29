from urllib.parse import urlparse, urlencode, urljoin


class UrlUtil(object):
    """
    - compose/decompose Url string.
    """

    @staticmethod
    def urljoin(a, b):
        domain = UrlUtil.domain(a)
        return urljoin(domain, urljoin(a[len(domain):], b))

    @staticmethod
    def get_query_dict(url):
        try:
            return dict([x.split('=') for x in urlparse(url).query.split('&')])
        except:
            return ''

    @staticmethod
    def to_query_str(params):
        try:
            return urlencode(params)
        except:
            return dict([])

    @staticmethod
    def parse_url(url):
        return urlparse(url)

    @staticmethod
    def domain(url):
        parsed_url = urlparse(url)
        return '{uri.scheme}://{uri.netloc}'.format(uri=parsed_url)

        # @staticmethod
        #    def to_url(url, params):
        #        return urllib.urlencode(params)


if __name__ == '__main__':
    print(UrlUtil.get_query_dict('SVD_Main.asp?pGB=1&gicode=A000540&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701')[
              'gicode'])
    # print(UrlUtil.urljoin('http://a.com/news', 'xxx'))
    # print(UrlUtil.urljoin('http://a.com/news/cate/', '/xxx'))
    # print(UrlUtil.urljoin('http://a.com/news/cate/', 'xxx'))
    # print(UrlUtil.urljoin('http://a.com/news/cate/', './xxx'))
    # print(UrlUtil.urljoin('http://a.com/news/cate/', '../xxx'))
    # print(UrlUtil.urljoin('http://a.com/news/cate/', '../../xxx'))
    print(UrlUtil.urljoin('http://news.kmib.co.kr/article/', 'view.asp?arcid=0011054231&code=61141411&sid1=eco'))
