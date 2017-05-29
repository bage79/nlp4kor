import bs4
from bs4 import BeautifulSoup
from bs4.element import Comment, Tag

from bage_utils.decorator_util import elapsed


@elapsed
def remain_useful_tags(soup):
    unuseful_tags = ['script', 'style']
    for node in soup.find_all(unuseful_tags):
        node.extract()
    for node in soup.find_all(text=lambda text: isinstance(text, Comment)):
        node.extract()
    return soup


@elapsed
def get_max_node(soup):
    max_len = 0
    max_node = None
    for node in soup.find_all():
        if not node.name in ['span', 'div', 'p', 'li', 'dd']:
            continue
        if len(list(node.children)) > 1:
            continue
        if max_len < get_sibling_text_len(node):
            max_len = get_sibling_text_len(node.text)
            max_node = node.parent
    return max_node


def get_sibling_text_len(node):
    if not isinstance(node, Tag):
        return 0
    if node.parent is None:
        return 0
    length = 0
    print()
    print('node.parent:', node.parent.name)
    for child in node.parent.children:
        if not hasattr(child, 'text'):
            continue
        print()
        print('child:', type(child), child.name)
        print('child.text:', child.string)
        length += len(child.text)
        print('length:', length)
    return length


if __name__ == '__main__':
    print(bs4.__version__)
    soup = BeautifulSoup(open('input/crawl2.html', 'r'), 'lxml')
    print('soup.builder:', soup.builder)
    soup = remain_useful_tags(soup)
    node = get_max_node(soup)
    # print(node)
    # print(soup.title.text)
    # print(soup.br, soup.br.name, type(soup.br))
    # print(len(soup.body.br))
