import logging
import multiprocessing
import os.path
import sys

from bage_utils.base_util import is_my_pc
from bage_utils.file_util import FileUtil

lock = multiprocessing.Lock()


class LogUtil(object):
    inited = False
    __default_logger_name = 'root'
    __log = None
    source_filepath = ''

    # ===============================================================================
    # default logging config
    # ===============================================================================
    #    level = logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
    filename = ''
    datefmt = '%Y-%m-%d %H:%M:%S'
    format = '[%(asctime)s][%(levelname)5s] %(message)s'  # '[%(asctime)s] %(message)s'

    @classmethod
    def get_logger(cls, source_filepath=None, level=logging.DEBUG, format=format, datefmt=datefmt, console_mode=False, multiprocess=False):
        """
        get global logging object with multiprocessing-safe.
        :param multiprocess: 
        :param source_filepath:
        :param console_mode: default: False
        :param datefmt:
        :param format:
        :param level: logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL(logging.FATAL
        :return: logging object
        """
        with lock:
            if cls.__log and cls.__log.hasHandlers() and cls.__log.level >= level:
                sys.stderr.write('reuse old logger (id:%s, log_path:%s), level:%s\n' % (
                    id(cls.__log), cls.source_filepath, cls.__log.level))
                return cls.__log

            cls.source_filepath = source_filepath
            if not console_mode and is_my_pc():
                console_mode = True

            if source_filepath is None:
                console_mode = True
                log_path = None
            else:
                log_path = cls.__log_path_from_path(source_filepath)

            error_log_path = None
            if log_path:
                error_log_path = log_path.replace('.log', '.error.log')
                # if delete_old_files:
                #     if log_path:
                #         DirUtil.rmdirs(os.path.dirname(log_path))

                if log_path:  # create log dir
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                if error_log_path:  # create log dir
                    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)

            # print('log_path:', log_path)
            # print('error_log_path:', error_log_path)

            cls.__log = LogUtil.__create_logger(log_path, error_log_path, level, console_mode=console_mode,
                                                format=format, datefmt=datefmt, multiprocess=multiprocess)
            cls.__log.setLevel(level)
            cls.__log.info('log level: %s' % logging.getLevelName(level))

            LogUtil.basigConfig(level, format=format, datefmt=datefmt)
            logging.getLogger("gensim").setLevel(logging.DEBUG)
            logging.getLogger("multiprocessing").setLevel(logging.ERROR)
            logging.getLogger("urllib3").setLevel(logging.ERROR)
            logging.getLogger("requests").setLevel(logging.ERROR)
            logging.getLogger('elasticsearch').setLevel(logging.ERROR)
            logging.getLogger('elasticsearch.trace').setLevel(logging.ERROR)
            logging.getLogger("py2neo").setLevel(logging.ERROR)
            logging.getLogger("py2neo.batch").setLevel(logging.ERROR)
            logging.getLogger("py2neo.cypher").setLevel(logging.ERROR)
            logging.getLogger("httpstream").setLevel(logging.ERROR)
            logging.getLogger("sqlalchemy").setLevel(logging.ERROR)
            # print('craete logger (%s, id=%s, log_path=%s), level=%s' % (cls.__log, id(cls.__log), log_path, level))

            return cls.__log

    @staticmethod
    def basigConfig(level=logging.NOTSET, format=format, datefmt=datefmt):
        logging.basicConfig(
            level=level,
            format=format,
            datefmt=datefmt
        )

    @staticmethod
    def __log_path_from_path(source_file, sub_log_dir='logs'):
        """
        :param sub_log_dir:
        :param source_file:
        :e.g.           ::LogUtil.get_logger(__file__) ./a.py -> ./logs/a.log
        :e.g.           ::LogUtil.get_logger(__file__, sub_log_dir='xx') ./a.py -> ./xx/a.log
        """
        _dir = os.path.join(os.path.dirname(source_file), sub_log_dir)
        _basename = os.path.basename(source_file)
        if len(sys.argv) > 1:
            _basename = '%s.%s' % (_basename, FileUtil.to_filename('.'.join(sys.argv[1:])))
        log_path = os.path.join(_dir, _basename) + '.log'
        return log_path

    @staticmethod
    def __create_logger(log_path, error_log_path=None, level=logging.NOTSET, console_mode=False, format=format,
                        datefmt=datefmt, multiprocess=False):
        if not LogUtil.inited:
            LogUtil.inited = True
            # sys.stderr.write('console_mode: %s\n' % console_mode)
            # sys.stderr.write('level: %s\n' % logging.getLevelName(level))
            if log_path is not None:
                sys.stderr.write('log_path: %s\n' % log_path)
            if error_log_path is not None:
                sys.stderr.write('error_log_path: %s\n' % error_log_path)

        log = logging.getLogger(
            log_path)  # for single process but MultProcTimedRotatingFileHandler support multiprocessing
        # log = multiprocessing.get_logger()  # for multiple processes

        formatter = logging.Formatter(fmt=format, datefmt=datefmt)
        # ===============================================================================
        # logging handlers
        # ===============================================================================
        log.propagate = False
        log.handlers = []

        if console_mode:
            _consoleHandler = logging.StreamHandler(stream=sys.stdout)  # console logger
            _consoleHandler.setLevel(level)
            _consoleHandler.setFormatter(logging.Formatter(fmt='%(message)s', datefmt=datefmt))
            log.addHandler(_consoleHandler)

            _console_errorHandler = logging.StreamHandler(stream=sys.stderr)  # console logger
            _console_errorHandler.setLevel(logging.ERROR)
            _console_errorHandler.setFormatter(logging.Formatter(fmt='%(message)s', datefmt=datefmt))
            log.addHandler(_console_errorHandler)

        if log_path:
            if multiprocess:
                from bage_utils.mult_proc_timed_rotating_file_handler import MultProcTimedRotatingFileHandler
                filehandler_class = MultProcTimedRotatingFileHandler
            else:
                from logging.handlers import TimedRotatingFileHandler
                filehandler_class = TimedRotatingFileHandler

            _fileHandler = filehandler_class(filename=log_path, when='midnight', interval=1, backupCount=0,
                                             encoding='utf8')
            _fileHandler.setLevel(level)
            _fileHandler.setFormatter(formatter)
            log.addHandler(_fileHandler)

            if error_log_path:
                _file_errorHandler = filehandler_class(filename=error_log_path, when='midnight', interval=1,
                                                       backupCount=0, encoding='utf8')
                _file_errorHandler.setLevel(logging.ERROR)
                _file_errorHandler.setFormatter(formatter)
                log.addHandler(_file_errorHandler)
        return log

    @classmethod
    def add_to_app_logger(cls, app):
        for h in cls.__log.handlers:
            app.logger.addHandler(h)
        return app.logger


if __name__ == '__main__':
    log = LogUtil.get_logger(sys.argv[0], level=logging.INFO, console_mode=True)
    log.debug('debug level OK.')
    log.info('info level OK.')
    log.error('error level OK.')
