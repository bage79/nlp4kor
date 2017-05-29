import os

import eventlet
import pymysql

eventlet.monkey_patch(thread=True, pymysql=True)
from eventlet.db_pool import ConnectionPool
import traceback
import threading


class EventletMySQLPoolUtil(object):
    """
    - .bashrc 또는 .bashprofile 에 MYSQL_PASSWD 를 설정해야 함.
    """
    def __init__(self, host, user, passwd, db, port=3306, charset='utf8', auto_commit=True,
                 min_size=100, max_size=100, max_idle=10, max_age=30, log=None):
        self.log = log
        self.host = host
        self.port = int(port)
        self.db = db
        self.auto_commit = auto_commit
        init_command = None
        if charset == 'utf8':
            init_command = 'SET NAMES UTF8'
        try:
            self.cp = ConnectionPool(pymysql, min_size=int(min_size), max_size=int(max_size), max_idle=int(max_idle),
                                     max_age=int(max_age),
                                     host=host, user=user, passwd=passwd, db=db, port=int(port),
                                     charset=charset, init_command=init_command)
        except Exception as e:
            traceback.print_exc()
            raise e

    def close(self):
        if self.cp:
            self.cp.clear()

    def __del__(self):
        self.close()

    #    def get_conn(self):
    #        conn = self.cp.get()
    #        conn.autocommit(self.auto_commit)
    #        return conn
    #
    #    def put_conn(self, conn):
    #        self.cp.put(conn)

    def execute(self, sql, *params):
        conn = self.cp.get()
        conn.autocommit(self.auto_commit)
        try:
            result = None
            result = conn.cursor(pymysql.cursors.DictCursor).execute(sql, params)
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            self.cp.put(conn)
        return result

    def insert(self, sql, *params):
        conn = self.cp.get()
        conn.autocommit(self.auto_commit)
        try:
            conn.cursor(pymysql.cursors.DictCursor).execute(sql, params)
            rowid = conn.insert_id()
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            self.cp.put(conn)
        return rowid

    def fetchone(self, sql, params=None):
        conn = self.cp.get()
        conn.autocommit(self.auto_commit)
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, params)
            row = cursor.fetchone()
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            self.cp.put(conn)
        return row

    def fetchall(self, sql, params=None):
        conn = self.cp.get()
        conn.autocommit(self.auto_commit)
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, params)
            while True:
                row = cursor.fetchone()
                if not row:
                    break
                yield row
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            self.cp.put(conn)

    def pool_size(self):
        return self.cp.current_size


if __name__ == '__main__':
    db = EventletMySQLPoolUtil('localhost', 'root', os.getenv('MYSQL_PASSWD'), 'test', port=20006, min_size=10,
                               max_size=100)


    def select_test(db):
        print('select_test...')
        print('conns:', db.pool_size())
        # row = db.fetchone("""SELECT COUNT(*) cnt from message WHERE device_no=%s""", '00')
        # print('cnt:', row['cnt']
        print('select_test... ok')


    def insert_test(db, msg):
        r = db.execute(
            """INSERT INTO em_tran (tran_phone, tran_callback, tran_status, tran_date, tran_msg, tran_etc4, tran_type) VALUES ('01073247942', '01073247942', '1', '2014-04-10 15:33:48', '%s', '0', '4')""" % msg)

        print(r)


    threads = []
    for i in range(2):
        print('t.start()...')
        msg = 'TEST_%d' % i
        t = threading.Thread(target=insert_test, args=(db, msg))
        t.start()
        threads.append(t)
        print('t.start() ok.')

    for t in threads:
        t.join()
    print('end!')

    #    for row in db.fetchall("""select * from room"""):
    #        print(row['name']
    #    c1 = db.get_conn()
    #    c2 = db.get_conn()
    #    c3 = db.get_conn()
    #    print(db.pool_size()
    #    db.put_conn(c1)
    #    print(db.pool_size()
    pass
