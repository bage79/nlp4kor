""" original https://github.com/surfly/gevent/blob/master/examples/psycopg2_pool.py """
import gevent
from gevent import monkey

monkey.patch_all()

from gevent.queue import Queue
from _mysql_exceptions import OperationalError
import pymysql
import contextlib
import sys
import threading
import os


class DatabaseConnectionPool(object):
    """
    - .bashrc 또는 .bashprofile 에 MYSQL_PASSWD 를 설정해야 함.
    """

    def __init__(self, max_size, auto_commit, fetchiter_size):
        if not isinstance(max_size, int):
            raise TypeError('Expected integer, got %r' % (max_size,))
        self.max_size = max_size
        self.auto_commit = auto_commit
        self.pool = Queue()
        self.size = 0
        self.fetchiter_size = fetchiter_size

    def get(self):
        #        print('size/max_size: %s/%s' % (self.size, self.max_size)
        #        print('pool.qsize(): %s' % (self.pool.qsize())

        #        if self.size >= self.max_size or self.pool.qsize():
        if self.pool.qsize() >= self.max_size:
            return self.pool.get()
        else:
            self.size += 1
            try:
                new_conn = self.create_connection()
            except:
                self.size -= 1
                raise
            return new_conn

    def put(self, item):
        self.pool.put(item)

    def close_all(self):
        while not self.pool.empty():
            conn = self.pool.get_nowait()
            try:
                conn.close()
            except:
                pass

    def commit_all(self):
        while not self.pool.empty():
            conn = self.pool.get_nowait()
            try:
                conn.commit()
            except:
                pass

    @contextlib.contextmanager
    def connection(self, isolation_level=None):
        conn = self.get()
        try:
            if isolation_level is not None:
                if conn.isolation_level == isolation_level:
                    isolation_level = None
                else:
                    conn.set_isolation_level(isolation_level)
            yield conn
        except:
            if not conn.open:
                conn = None
                self.close_all()
            else:
                conn = self._rollback(conn)
            raise
        else:
            if not conn.open:
                raise OperationalError("Cannot commit because connection was closed: %r" % (conn,))
        finally:
            if conn is not None and conn.open:
                if isolation_level is not None:
                    conn.set_isolation_level(isolation_level)
                self.put(conn)

    @contextlib.contextmanager
    def cursor(self, *args, **kwargs):
        try:
            isolation_level = kwargs.pop('isolation_level', None)
            with self.connection(isolation_level) as conn:
                yield conn.cursor(cursorclass=pymysql.cursors.SSDictCursor, *args, **kwargs)
        except:
            raise

    def _rollback(self, conn):
        try:
            conn.rollback()
        except:
            gevent.hub.get_hub().handle_error(conn, *sys.exc_info())
            return
        return conn

    def execute(self, *args, **kwargs):
        try:
            with self.cursor(**kwargs) as cursor:
                cursor.execute(*args)
                return cursor.rowcount
        except:
            raise

    def executemany(self, *args, **kwargs):
        try:
            with self.cursor(**kwargs) as cursor:
                cursor.executemany(*args)
                return cursor.rowcount
        except:
            raise

    def fetchone(self, *args, **kwargs):
        try:
            with self.cursor(**kwargs) as cursor:
                cursor.execute(*args)
                return cursor.fetchone()
        except:
            raise

    def fetchall(self, *args, **kwargs):
        try:
            with self.cursor(**kwargs) as cursor:
                cursor.execute(*args)
                return cursor.fetchall()
        except:
            raise

    def fetchiter(self, *args, **kwargs):
        try:
            with self.cursor(**kwargs) as cursor:
                cursor.execute(*args)
                while True:
                    items = cursor.fetchmany(size=self.fetchiter_size)
                    if not items:
                        break
                    for item in items:
                        yield item
        except:
            raise


class MySQLPoolGeventUtil(DatabaseConnectionPool):
    DEFAULT_POOL_MAX_SIZE = 10
    FETCHITER_SIZE = 1000

    def __init__(self, host, user, passwd, db, port=3306, charset='utf8', auto_commit=True,
                 max_size=None, fetchiter_size=FETCHITER_SIZE):
        if max_size is None:
            max_size = MySQLPoolGeventUtil.DEFAULT_POOL_MAX_SIZE
        # if DatabaseConnectionPool.CURSOR == pymysql.cursors.SSDictCursor:
        #            auto_commit = False
        #            print('auto_commit = False'
        self.host = host
        self.user = user
        self.db = db
        self.passwd = passwd
        self.port = port
        self.charset = charset
        self.init_command = ''
        self.auto_commit = auto_commit
        self.init_command = 'SET NAMES %s' % charset
        try:
            DatabaseConnectionPool.__init__(self, max_size, auto_commit, fetchiter_size)
        except:
            raise

    def create_connection(self):
        conn = pymysql.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db, port=self.port,
                               charset=self.charset, init_command=self.init_command)
        conn.autocommit(self.auto_commit)
        #        print('create_connection(auto_commit=%s)' % self.auto_commit
        return conn


if __name__ == '__main__':
    db = MySQLPoolGeventUtil(host='localhost', user='root', passwd=os.getenv('MYSQL_PASSWD'), db='test', port=20006,
                             max_size=100)
    room_id = '11112'
    room_name = '홍길동 모임'
    last_message_id = '2222222'


    #    db.execute("""insert into room(room_id, room_name, last_message_id) values(%s, %s, %s)""", room_id, room_name, last_message_id)

    def select_test(no):
        for _i in range(1000):
            #        print('select_test...'
            print('[%s] conns: %s' % (no, db.size))
            row = db.fetchone("""SELECT COUNT(*) cnt from message WHERE device_no=%s""", '00')  # @UnusedVariable


    # print('cnt:', row['cnt']
    #        print('select_test... ok'

    thread_count = 100
    threads = []
    for no in range(thread_count):
        print('t.start()...')
        t = threading.Thread(target=select_test, args=(no,))
        t.start()
        threads.append(t)
        print('t.start() ok.')

    for t in threads:
        t.join()
    print('end!')
    print('db.size:', db.size)
    pass
