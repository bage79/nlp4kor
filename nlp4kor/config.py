import logging
import os
import sys

from bage_utils.base_util import is_my_pc, db_hostname
from bage_utils.log_util import LogUtil

# warnings.simplefilter(action='ignore', category=FutureWarning)

log = None
if log is None:
    if is_my_pc():
        log = LogUtil.get_logger(None, level=logging.DEBUG, console_mode=True)  # global log
    else:
        log = LogUtil.get_logger(sys.argv[0], level=logging.INFO, console_mode=True)  # global log # console_mode=True for jupyter notebook


MONGO_URL = r'mongodb://%s:%s@%s:%s/%s?authMechanism=MONGODB-CR' % ('root', os.getenv('MONGODB_PASSWD') or 'gPdnd', 'db-local', '27017', 'admin') # FIXME: os.getenv()
MYSQL_URL = {'host': db_hostname(), 'user': 'root', 'passwd': os.getenv('MYSQL_PASSWD'), 'db': 'kr_nlp'}

PROJECT_DIR = os.path.join(os.getenv("HOME"), 'workspace/nlp4kor')
# log.info('PROJECT_DIR: %s' % PROJECT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, 'data/')
# log.info('DATA_DIR: %s' % DATA_DIR)
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

TENSORBOARD_LOG_DIR = os.path.join(os.getenv("HOME"), 'workspace/tensorboard_log/')
# log.info('TENSORBOARD_LOG_DIR: %s' % TENSORBOARD_LOG_DIR)
if not os.path.exists(TENSORBOARD_LOG_DIR):
    os.mkdir(TENSORBOARD_LOG_DIR)

# dataset repositories
MNIST_DATA_DIR = os.path.join(os.getenv('HOME'), 'workspace/nlp4kor-mnist')
KO_WIKIPEDIA_ORG_DATA_DIR = os.path.join(os.getenv('HOME'), 'workspace/nlp4kor-ko.wikipedia.org')
KO_WIKIPEDIA_ORG_SENTENCES_FILE = os.path.join(KO_WIKIPEDIA_ORG_DATA_DIR, 'corpus/ko.wikipedia.org.sentences.gz')
KO_WIKIPEDIA_ORG_URLS_FILE = os.path.join(KO_WIKIPEDIA_ORG_DATA_DIR, 'corpus/ko.wikipedia.org.urls.txt')

if __name__ == '__main__':
    print('DATA_DIR:', MNIST_DATA_DIR)
    print('MONGODB_PASSWD', os.getenv('MONGODB_PASSWD'), os.environ.get('MONGODB_PASSWD'))
