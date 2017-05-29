import gzip
import os
import pickle

from bage_utils.base_util import is_my_pc
from bage_utils.datafile_util import DataFileUtil
from bage_utils.dataset import DataSet
from bage_utils.file_util import FileUtil
from bage_utils.hangul_util import HangulUtil
from bage_utils.mongodb_util import MongodbUtil
from bage_utils.num_util import NumUtil
from bage_utils.one_hot_vector import OneHotVector
from nlp4kor.config import log, DATA_DIR_KO_WIKIPEDIA_ORG, MONGO_URL


class TextPreprocess(object):
    @staticmethod
    def dump_corpus(mongo_url, db_name, collection_name, sentences_file, mongo_query=None):
        """
        Mongodb에서 문서를 읽어서, 문장 단위로 저장한다. (단 문장안의 단어가 1개 이거나, 한글이 전혀 없는 문장은 추출하지 않는다.)
        :param mongo_url: mongodb://~~~ 
        :param db_name: database name of mongodb
        :param collection_name: collection name of mongodb
        :param sentences_file: *.sentence file
        :param mongo_query: default={}
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
            for i, row in enumerate(corpus_mongo.find(mongo_query)):
                # print('url:', row['url'])
                for c in row['content']:
                    if i % 1000 == 0:
                        print('%.1f%% writed.' % (i / total * 100))
                    for s in HangulUtil.text2sentences(c['sentences']):
                        if HangulUtil.has_hangul(s):
                            out_f.write(s)
                            out_f.write('\n')

    @staticmethod
    def collect_characters(input_sentences_file: str, output_chars_file: str, max_test: int = 0):
        """
        문장 파일을 읽어서, 유니크한 문자(음절)들을 추출 한다.
        추후 corpus기반으로 one hot vector 생성시 사용한다.
        :param input_sentences_file: *.sentences file path 
        :param output_chars_file: *.characters file path
        :param max_test: 0=run all 
        :return: 
        """
        total = FileUtil.count_lines(input_sentences_file, gzip_format=True)
        log.info('total: %s' % NumUtil.comma_str(total))

        char_set = set()
        with gzip.open(input_sentences_file, 'rt') as f:
            for i, sentence in enumerate(f):
                i += 1
                if i % 10000 == 0:
                    log.info('%.1f%% readed.' % (i / total * 100))
                _char_set = set([c for c in sentence])
                char_set.update(_char_set)
                if 0 < max_test <= i:
                    break

        char_list = list(char_set)
        char_list.sort()
        if max_test == 0:  # 0=full
            with open(output_chars_file, 'w') as f:
                for c in char_list:
                    f.write(c)
                    f.write('\n')
                log.info('writed to %s OK.' % output_chars_file)


if __name__ == '__main__':
    sentences_file = os.path.join(DATA_DIR_KO_WIKIPEDIA_ORG, 'ko.wikipedia.org.sentences.gz')
    log.info('sentences_file: %s' % sentences_file)
    if not os.path.exists(sentences_file):
        TextPreprocess.dump_corpus(MONGO_URL, db_name='parsed', collection_name='ko.wikipedia.org', sentences_file=sentences_file,
                                   mongo_query={})  # mongodb -> text file(corpus)

    characters_file = os.path.join(DATA_DIR_KO_WIKIPEDIA_ORG, 'ko.wikipedia.org.characters')
    log.info('characters_file: %s' % characters_file)
    if not os.path.exists(characters_file):
        log.info('collect characters...')
        TextPreprocess.collect_characters(sentences_file, characters_file)  # text file -> characters(unique features)
        log.info('collect characters OK.')
