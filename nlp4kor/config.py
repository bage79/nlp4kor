import logging
import os
import sys
import warnings

from bage_utils.base_util import db_hostname, is_my_pc
from bage_utils.log_util import LogUtil

warnings.simplefilter(action='ignore', category=FutureWarning)  # ignore future warnings

log = None
if log is None:
    if len(sys.argv) == 1:  # my pc or pycharm remote
        if is_my_pc():  # my pc
            log = LogUtil.get_logger(None, level=logging.DEBUG, console_mode=True)  # global log
        else:  # pycharm remote
            log = LogUtil.get_logger(sys.argv[0], level=logging.DEBUG, console_mode=True)  # global log # console_mode=True for jupyter
    else:  # by batch script
        log = LogUtil.get_logger(sys.argv[0], level=logging.INFO, console_mode=False)  # global log

MONGO_URL = r'mongodb://%s:%s@%s:%s/%s?authMechanism=MONGODB-CR' % (
    'root', os.getenv('MONGODB_PASSWD'), 'db-local', '27017', 'admin')
MYSQL_URL = {'host': db_hostname(), 'user': 'root', 'passwd': os.getenv('MYSQL_PASSWD'), 'db': 'kr_nlp'}

PROJECT_DIR = os.path.join(os.getenv("HOME"), 'workspace/nlp4kor')

TENSORBOARD_LOG_DIR = os.path.join(os.getenv("HOME"), 'tensorboard_log')
# log.info('TENSORBOARD_LOG_DIR: %s' % TENSORBOARD_LOG_DIR)
if not os.path.exists(TENSORBOARD_LOG_DIR):
    os.mkdir(TENSORBOARD_LOG_DIR)

#################################################
# sample
#################################################
SAMPLE_DATA_DIR = os.path.join(PROJECT_DIR, 'data')
if not os.path.exists(SAMPLE_DATA_DIR):
    os.mkdir(SAMPLE_DATA_DIR)

SAMPLE_MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
if not os.path.exists(SAMPLE_MODELS_DIR):
    os.mkdir(SAMPLE_MODELS_DIR)

#################################################
# mnist
#################################################
MNIST_DIR = os.path.join(os.getenv('HOME'), 'workspace', 'nlp4kor-mnist')
MNIST_DATA_DIR = os.path.join(MNIST_DIR, 'data')
MNIST_CNN_MODEL_DIR = os.path.join(MNIST_DIR, 'models', 'cnn')
MNIST_DAE_MODEL_DIR = os.path.join(MNIST_DIR, 'models', 'dae')

#################################################
# ko.wikipedia.org
#################################################
WIKIPEDIA_DIR = os.path.join(os.getenv('HOME'), 'workspace', 'nlp4kor-ko.wikipedia.org')

# info
WIKIPEDIA_INFO_FILE = os.path.join(WIKIPEDIA_DIR, 'data', 'ko.wikipedia.org.info.txt')
WIKIPEDIA_URLS_FILE = os.path.join(WIKIPEDIA_DIR, 'data', 'ko.wikipedia.org.urls.txt')

# text (with string)
WIKIPEDIA_DATA_DIR = os.path.join(WIKIPEDIA_DIR, 'data')
if not os.path.exists(WIKIPEDIA_DATA_DIR):
    os.mkdir(WIKIPEDIA_DATA_DIR)

WIKIPEDIA_CHARACTERS_FILE = os.path.join(WIKIPEDIA_DATA_DIR, 'ko.wikipedia.org.characters')
WIKIPEDIA_SENTENCES_FILE = os.path.join(WIKIPEDIA_DATA_DIR, 'ko.wikipedia.org.sentences.gz')
WIKIPEDIA_TRAIN_SENTENCES_FILE = os.path.join(WIKIPEDIA_DATA_DIR, 'ko.wikipedia.org.train.sentences.gz')
WIKIPEDIA_VALID_SENTENCES_FILE = os.path.join(WIKIPEDIA_DATA_DIR, 'ko.wikipedia.org.valid.sentences.gz')
WIKIPEDIA_TEST_SENTENCES_FILE = os.path.join(WIKIPEDIA_DATA_DIR, 'ko.wikipedia.org.test.sentences.gz')

# csv (with character id)
WIKIPEDIA_TRAIN_FILE = os.path.join(WIKIPEDIA_DATA_DIR, 'ko.wikipedia.org.train.sentences.cid.gz')  # TODO: csv.gz
WIKIPEDIA_VALID_FILE = os.path.join(WIKIPEDIA_DATA_DIR, 'ko.wikipedia.org.valid.sentences.cid.gz')
WIKIPEDIA_TEST_FILE = os.path.join(WIKIPEDIA_DATA_DIR, 'ko.wikipedia.org.test.sentences.cid.gz')

# csv (with character id) for specific purpose
WIKIPEDIA_DATASET_DIR = os.path.join(WIKIPEDIA_DIR, 'dataset')
if not os.path.exists(WIKIPEDIA_DATASET_DIR):
    os.mkdir(WIKIPEDIA_DATASET_DIR)

WIKIPEDIA_MODELS_DIR = os.path.join(WIKIPEDIA_DIR, 'models')
if not os.path.exists(WIKIPEDIA_MODELS_DIR):
    os.mkdir(WIKIPEDIA_MODELS_DIR)

#################################################
# word spacing
#################################################
WORD_SPACING_DATASET_DIR = os.path.join(WIKIPEDIA_DATASET_DIR, 'word_spacing')
WORD_SPACING_MODEL_DIR = os.path.join(WIKIPEDIA_MODELS_DIR, 'word_spacing')

#################################################
# spelling error correction
#################################################
SPELLING_ERROR_CORRECTION_DATASET_DIR = os.path.join(WIKIPEDIA_DATASET_DIR, 'spelling_error_correction')
if not os.path.exists(SPELLING_ERROR_CORRECTION_DATASET_DIR):
    os.mkdir(SPELLING_ERROR_CORRECTION_DATASET_DIR)

SPELLING_ERROR_CORRECTION_TRAIN_DATASET_FILE = os.path.join(SPELLING_ERROR_CORRECTION_DATASET_DIR, 'ko.wikipedia.org.train.sentences.csv')
SPELLING_ERROR_CORRECTION_VALID_DATASET_FILE = os.path.join(SPELLING_ERROR_CORRECTION_DATASET_DIR, 'ko.wikipedia.org.valid.sentences.csv')
SPELLING_ERROR_CORRECTION_TEST_DATASET_FILE = os.path.join(SPELLING_ERROR_CORRECTION_DATASET_DIR, 'ko.wikipedia.org.test.sentences.csv')

SPELLING_ERROR_CORRECTION_MODEL_DIR = os.path.join(WIKIPEDIA_MODELS_DIR, 'spelling_error_correction')
if not os.path.exists(SPELLING_ERROR_CORRECTION_MODEL_DIR):
    os.mkdir(SPELLING_ERROR_CORRECTION_MODEL_DIR)
