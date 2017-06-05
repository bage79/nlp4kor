import gzip
import os
import traceback

from bage_utils.file_util import FileUtil
from bage_utils.hangul_util import HangulUtil
from bage_utils.mongodb_util import MongodbUtil
from bage_utils.num_util import NumUtil
from nlp4kor.config import log, MONGO_URL, KO_WIKIPEDIA_ORG_SENTENCES_FILE, KO_WIKIPEDIA_ORG_URLS_FILE, \
    KO_WIKIPEDIA_ORG_CHARACTERS_FILE


class TextPreprocess(object):
    @staticmethod
    def dump_urls(mongo_url, db_name, collection_name, urls_file, mongo_query=None, limit=0):
        if mongo_query is None:
            mongo_query = {}

        corpus_mongo = MongodbUtil(mongo_url, db_name=db_name, collection_name=collection_name)
        total = corpus_mongo.count()
        log.info('%s total: %s' % (corpus_mongo, NumUtil.comma_str(total)))

        output_dir = os.path.basename(urls_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(urls_file, 'wt') as out_f:
            for i, row in enumerate(corpus_mongo.find(mongo_query, limit=limit)):
                if i % 1000 == 0:
                    log.info('%s %.1f%% writed.' % (os.path.basename(urls_file), i / total * 100))
                    out_f.write(row['url'])
                    out_f.write('\n')

    @staticmethod
    def dump_corpus(mongo_url, db_name, collection_name, sentences_file, mongo_query=None, limit=None):
        """
        Mongodb에서 문서를 읽어서, 문장 단위로 저장한다. (단 문장안의 단어가 1개 이거나, 한글이 전혀 없는 문장은 추출하지 않는다.)
        :param mongo_url: mongodb://~~~
        :param db_name: database name of mongodb
        :param collection_name: collection name of mongodb
        :param sentences_file: *.sentence file
        :param mongo_query: default={}
        :param limit:
        :return:
        """
        if mongo_query is None:
            mongo_query = {}

        corpus_mongo = MongodbUtil(mongo_url, db_name=db_name, collection_name=collection_name)
        total = corpus_mongo.count()
        log.info('%s total: %s' % (corpus_mongo, NumUtil.comma_str(total)))

        output_dir = os.path.basename(sentences_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with gzip.open(sentences_file, 'wt') as out_f:
            for i, row in enumerate(corpus_mongo.find(mongo_query, limit=limit)):
                # print('url:', row['url'])
                for c in row['content']:
                    if i % 1000 == 0:
                        log.info('%s %.1f%% writed.' % (os.path.basename(sentences_file), i / total * 100))
                    for s in HangulUtil.text2sentences(c['sentences']):
                        if HangulUtil.has_hangul(s):
                            out_f.write(s)
                            out_f.write('\n')

    @staticmethod
    def collect_characters(sentences_file: str, characters_file: str, max_test: int = 0):
        """
        문장 파일을 읽어서, 유니크한 문자(음절)들을 추출 한다.
        추후 corpus기반으로 one hot vector 생성시 사용한다.
        :param sentences_file: *.sentences file path 
        :param characters_file: *.characters file path
        :param max_test: 0=run all 
        :return: 
        """
        total = FileUtil.count_lines(sentences_file, gzip_format=True)
        log.info('total: %s' % NumUtil.comma_str(total))

        char_set = set()
        with gzip.open(sentences_file, 'rt') as f:
            for i, sentence in enumerate(f):
                i += 1
                if i % 10000 == 0:
                    log.info('%s %.1f%% writed.' % (os.path.basename(characters_file), i / total * 100))
                _char_set = set([c for c in sentence])
                char_set.update(_char_set)
                if 0 < max_test <= i:
                    break

        char_list = list(char_set)
        char_list.sort()
        if max_test == 0:  # 0=full
            with open(characters_file, 'w') as f:
                for c in char_list:
                    f.write(c)
                    f.write('\n')
                log.info('writed to %s OK.' % characters_file)


if __name__ == '__main__':
    urls_file = KO_WIKIPEDIA_ORG_URLS_FILE
    sentences_file = KO_WIKIPEDIA_ORG_SENTENCES_FILE
    characters_file = KO_WIKIPEDIA_ORG_CHARACTERS_FILE
    log.info('urls_file: %s' % urls_file)
    log.info('sentences_file: %s' % sentences_file)
    log.info('characters_file: %s' % characters_file)

    if not os.path.exists(urls_file):
        try:
            log.info('create urls file...')
            TextPreprocess.dump_urls(MONGO_URL, db_name='parsed', collection_name='ko.wikipedia.org', urls_file=urls_file,
                                     mongo_query={})  # mongodb -> text file(url)
            log.info('create urls file OK')
        except:
            log.error(traceback.format_exc())
            if os.path.exists(urls_file):
                os.remove(urls_file)

    if not os.path.exists(sentences_file):
        try:
            log.info('create senences file...')
            TextPreprocess.dump_corpus(MONGO_URL, db_name='parsed', collection_name='ko.wikipedia.org', sentences_file=sentences_file,
                                       mongo_query={})  # mongodb -> text file(corpus)
            log.info('create senences file OK')
        except:
            log.error(traceback.format_exc())
            if os.path.exists(sentences_file):
                os.remove(sentences_file)

    if not os.path.exists(characters_file):
        try:
            log.info('create characters file...')
            TextPreprocess.collect_characters(sentences_file, characters_file)  # text file -> characters(unique features)
            log.info('create characters file OK.')
        except:
            log.error(traceback.format_exc())
            if os.path.exists(characters_file):
                os.remove(characters_file)
