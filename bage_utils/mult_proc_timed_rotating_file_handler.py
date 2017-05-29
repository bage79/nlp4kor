import datetime
import logging.handlers
import os
import re
import time
from logging.handlers import BaseRotatingHandler
from random import randint

# sibling module than handles all the ugly platform-specific details of file locking
from portalocker import lock, unlock, LOCK_EX

__version__ = '0.0.1'
__author__ = "yorks"


# noinspection PyUnusedLocal
class MultProcTimedRotatingFileHandler(BaseRotatingHandler):
    """
    Handler for logging to a file, rotating the log file at certain timed.
    https://github.com/yorks/mpfhandler
    """

    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False,
                 debug=False):
        """ 
            * interval, backupCount is not working!!! *

        Just Copied from logging.handlers.TimedRotatingFileHandler

        # a rollover occurs.  Current 'when' events supported:
        # S - Seconds
        # M - Minutes
        # H - Hours
        # D - Days
        # midnight - roll over at midnight
        # W{0-6} - roll over on a certain day; 0 - Monday
        #
        # Case of the 'when' specifier is not important; lower or upper case
        # will work.

        """
        BaseRotatingHandler.__init__(self, filename, 'a', encoding, delay)
        self.when = when.upper()
        self.backupCount = backupCount
        self.utc = utc
        self.debug = debug
        self.mylogfile = "%s.%08d" % ('/tmp/mptfhanldler', randint(0, 99999999))

        self.interval = 1  # datetime timedelta only have, days, seconds, microseconds

        if self.when == 'S':
            # self.interval = 1 # one second
            self.suffix = "%Y-%m-%d_%H-%M-%S"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
        elif self.when == 'M':
            self.interval = 60  # one minute
            self.suffix = "%Y-%m-%d_%H-%M"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$"
        elif self.when == 'H':
            self.interval = 60 * 60  # one hour
            self.suffix = "%Y-%m-%d_%H"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}$"
        elif self.when == 'D' or self.when == 'MIDNIGHT':
            # self.interval = 60 * 60 * 24 # one day
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}$"
            self.when = 'D'  # MIDNIGHT is day, use day only
        elif self.when.startswith('W'):
            # self.interval = 60 * 60 * 24 * 7 # one week
            if len(self.when) != 2:
                raise ValueError("You must specify a day for weekly rollover from 0 to 6 (0 is Monday): %s" % self.when)
            if self.when[1] < '0' or self.when[1] > '6':
                raise ValueError("Invalid day specified for weekly rollover: %s" % self.when)
            self.dayOfWeek = int(self.when[1])
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}$"
        else:
            raise ValueError("Invalid rollover interval specified: %s" % self.when)

        self.extMatch = re.compile(self.extMatch)
        # self.interval = self.interval * interval # multiply by units requested
        # self.interval = self.interval * 1  # interval arg is not working

        # lock file, contain next rollover timestamp
        self.stream_lock = None
        self.lock_file = self._getLockFile()

        # read from conf first for inherit the first process
        # if it is the first process, please remove the lock file by hand first
        self.nextRolloverTime = self.getNextRolloverTime()
        if not self.nextRolloverTime:
            self.nextRolloverTime = self.computerNextRolloverTime()
            self.saveNextRolloverTime()

    def _log2mylog(self, msg):
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        msg = str(msg)
        content = "%s [%s]\n" % (time_str, msg)
        fa = open(self.mylogfile, 'a')
        fa.write(content)
        fa.close()

    def _getLockFile(self):
        lock_file = os.path.join(os.path.dirname(self.baseFilename),
                                 '.' + os.path.basename(self.baseFilename) + '.lock')
        return lock_file

    def _openLockFile(self):
        lock_file = self._getLockFile()
        self.stream_lock = open(lock_file, 'w')

    def computerNextRolloverTime(self):
        """ Work out the next rollover time. """
        nextTime = None
        currentDateTime = datetime.datetime.now()
        if self.utc:
            currentDateTime = datetime.datetime.utcnow()

        if self.when == 'D':
            nextDateTime = currentDateTime + datetime.timedelta(days=self.interval)
            nextDate = nextDateTime.date()
            nextTime = int(time.mktime(nextDate.timetuple()))
        elif self.when.startswith('W'):
            days = 0
            currentWeekDay = currentDateTime.weekday()
            if currentWeekDay == self.dayOfWeek:
                days = (self.interval + 7)
            elif currentWeekDay < self.dayOfWeek:
                days = self.dayOfWeek - currentWeekDay
            else:
                days = 6 - currentWeekDay + self.dayOfWeek + 1
            nextDateTime = currentDateTime + datetime.timedelta(days=days)
            nextDate = nextDateTime.date()
            nextTime = int(time.mktime(nextDate.timetuple()))
        else:
            tmpNextDateTime = currentDateTime + datetime.timedelta(seconds=self.interval)
            nextDateTime = tmpNextDateTime.replace(microsecond=0)
            if self.when == 'H':
                nextDateTime = tmpNextDateTime.replace(minute=0, second=0, microsecond=0)
            elif self.when == 'M':
                nextDateTime = tmpNextDateTime.replace(second=0, microsecond=0)

            nextTime = int(time.mktime(nextDateTime.timetuple()))
        return nextTime

    def getNextRolloverTime(self):
        """ get next rollover time stamp from lock file """
        try:
            fp = open(self.lock_file, 'r')
            c = fp.read()
            fp.close()
            return int(c)
        except:
            return False

    def saveNextRolloverTime(self):
        """ save the nextRolloverTimestamp to lock file

            this is a flag for avoid multiple processes to rotate
            the log file again at the same rollovertime.
        """
        if not self.nextRolloverTime:
            return 0
        content = "%d" % self.nextRolloverTime

        if not self.stream_lock:
            self._openLockFile()
        lock(self.stream_lock, LOCK_EX)
        try:
            self.stream_lock.seek(0)
            self.stream_lock.write(content)
            self.stream_lock.flush()
        except:
            if self.debug:
                self._log2mylog('saveNextRT exception!!!')
            pass
        finally:
            unlock(self.stream_lock)
        if self.debug:
            self._log2mylog('saveNextRT:%s' % content)

    def acquire(self):
        """ Acquire thread and file locks.  

            Copid from ConcurrentRotatingFileHandler
        """
        # handle thread lock
        BaseRotatingHandler.acquire(self)
        # Issue a file lock.  (This is inefficient for multiple active threads
        # within a single process. But if you're worried about high-performance,
        # you probably aren't using this log handler.)
        if self.stream_lock:
            # If stream_lock=None, then assume close() was called or something
            # else weird and ignore all file-level locks.
            if self.stream_lock.closed:
                # Daemonization can close all open file descriptors, see
                # https://bugzilla.redhat.com/show_bug.cgi?id=952929
                # Try opening the lock file again.  Should we warn() here?!?
                try:
                    self._openLockFile()
                except Exception:
                    # Don't try to open the stream lock again
                    self.stream_lock = None
                    return
            lock(self.stream_lock, LOCK_EX)
            # Stream will be opened as part by FileHandler.emit()

    def release(self):
        """ Release file and thread locks. 
        """
        try:
            if self.stream_lock and not self.stream_lock.closed:
                unlock(self.stream_lock)
        except Exception:
            pass
        finally:
            # release thread lock
            BaseRotatingHandler.release(self)

    def _close_stream(self):
        """ Close the log file stream """
        if self.stream:
            try:
                if not self.stream.closed:
                    self.stream.flush()
                    self.stream.close()
            finally:
                self.stream = None

    def _close_stream_lock(self):
        """ Close the lock file stream """
        if self.stream_lock:
            try:
                if not self.stream_lock.closed:
                    self.stream_lock.flush()
                    self.stream_lock.close()
            finally:
                self.stream_lock = None

    def close(self):
        """
        Close log stream and stream_lock. """
        try:
            self._close_stream()
            self._close_stream_lock()
        finally:
            self.stream = None
            self.stream_lock = None

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.

        record is not used, as we are just comparing times, but it is needed so
        the method signatures are the same

        Copied from std lib
        """
        t = int(time.time())
        if t >= self.nextRolloverTime:
            return 1
        # print "No need to rollover: %d, %d" % (t, self.rolloverAt)
        return 0

    def doRollover(self):
        """ Do a rollover, 

            0. close stream, stream_lock file handle  
            1. get lock
            2. mv log log.$date
            3. setting up nextRolloverTime
            4. relese lock
        """
        if self.debug:
            self._log2mylog('do Rollover')
        self._close_stream()
        self.acquire()
        try:
            fileNextRolloverTime = self.getNextRolloverTime()
            if not fileNextRolloverTime:
                if self.debug:
                    self._log2mylog('getNextRolloverTime False, skip rotate!')
                self.release()
                return 0
            # avoid other process do rollover again.
            if self.nextRolloverTime < fileNextRolloverTime:
                self.nextRolloverTime = fileNextRolloverTime
                if self.debug:
                    self._log2mylog('already rotated, skip this proc to rotate!')
                self.release()
                return 0
        except Exception as e:
            pass
        # because log is older then self.nextRolloverTime,
        # we need the old log rename to old filename
        #   donot use time.time()-1, 
        #   for between last rollover and nextRolloverTime*N may have none log to record.
        time_tuple = time.localtime(self.nextRolloverTime - 1)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, time_tuple)
        # rename
        if os.path.exists(dfn):
            bakname = dfn + ".bak"
            while os.path.exists(bakname):
                bakname = "%s.%08d" % (bakname, randint(0, 99999999))
            try:
                os.rename(dfn, bakname)
            except:
                pass
        if os.path.exists(self.baseFilename):
            try:
                if self.debug:
                    self._log2mylog('rename %s to %s' % (self.baseFilename, dfn))
                os.rename(self.baseFilename, dfn)
            except:
                pass
        # set new nextRolloverTime
        self.nextRolloverTime = self.computerNextRolloverTime()
        self.saveNextRolloverTime()

        if not self.delay:
            self.stream = self._open()
        self.release()


logging.handlers.MultProcTimedRotatingFileHandler = MultProcTimedRotatingFileHandler
