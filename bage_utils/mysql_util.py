import os
import time
import traceback

import pymysql


class MySQLUtil(object):
    """
    - mysql client wrappter with `pymysql`
    - .bashrc 또는 .bashprofile 에 MYSQL_PASSWD 를 설정해야 함.
    """

    def __init__(self, host, user, passwd, db, port=3306, charset='UTF8', auto_commit=True, auto_connect=True):
        self.init_command = None
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db
        self.port = int(port)
        self.charset = charset.replace('-', '').upper()
        self.auto_commit = auto_commit
        self.conn = None
        self.cursor = None
        if auto_connect:
            self.connect()

    def connect(self):
        if not self.cursor or not self.cursor.connection:
            # print('connect()...')
            self.init_command = 'SET NAMES %s' % self.charset
            self.conn = pymysql.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db, port=self.port, charset=self.charset,
                                        init_command=self.init_command, autocommit=True)
            # self.cursor = self.conn.cursor(pymysql.cursors.SSDictCursor)
            self.cursor = self.conn.cursor(pymysql.cursors.DictCursor)
            # self.conn.autocommit(self.auto_commit)
            # print('connect() OK.')

    def __repr__(self):
        return '%s@%s:%s/%s' % (self.user, self.host, self.port, self.db)

    def __del__(self):
        if self.cursor:
            self.conn.close()

    # def __check_connection(self):
    #     try:
    #         self.cursor.execute("SELECT 1")
    #     except (AttributeError, pymysql.OperationalError):
    #         self.connect()

    def affected_rows(self):
        # when using 'insert .. on duplicate key update ..'
        # 0 if an existing row is set to its current values
        # 1 if the row is inserted as a new row
        # 2 if an existing row is updated
        return self.conn.affected_rows()

    @property
    def rowcount(self):
        return self.cursor.rowcount

    @staticmethod
    def addslashes(field):
        return field.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')

    def execute(self, query):
        if not self.cursor:
            self.connect()
        try:
            self.cursor.execute(query)
        except pymysql.OperationalError:
            self.connect()  # reconnect
            self.cursor.execute(query)
        except Exception as e:
            print('에러:', e)
            raise e

    def select_one(self, query):
        if not self.cursor:
            self.connect()
        try:
            self.cursor.execute(query)
            return self.cursor.fetchone()
        except pymysql.OperationalError:
            self.connect()  # reconnect
            self.cursor.execute(query)
            return self.cursor.fetchone()
        except Exception as e:
            raise e

    def select(self, query):
        if not self.cursor:
            self.connect()
        try:
            self.cursor.execute(query)
            while True:
                row = self.cursor.fetchone()
                if not row:
                    break
                yield row
        except pymysql.OperationalError:
            self.connect()  # reconnect
            self.cursor.execute(query)
            while True:
                row = self.cursor.fetchone()
                if not row:
                    break
                yield row
        except Exception as e:
            raise e


if __name__ == '__main__':
    # noinspection PyShadowingNames
    __MYSQL = {
        'host': '1.2.3.4',
        'user': 'root',
        'passwd': os.getenv('MYSQL_PASSWD'),
        'db': 'test',
        'port': 3306,
        'charset': 'utf8'
    }

    # field = ''' \\ ' " '''
    # print(field)
    # print(MySQLUtil.addslashes(field))

    start = time.time()
    # db = MySQLUtil(host='1.2.3.4', user='user', passwd='passwd', db='db_name', port=3306, charset='utf8', )
    try:
        # input_collection = MySQLUtil(host='localhost', user='root', passwd='user_passwd', db='wikipedia_korean', port=3306, charset='utf8', auto_connect=False)
        input_collection = MySQLUtil(**__MYSQL)
        # noinspection SqlDialectInspection
        for row in input_collection.select('SELECT prefix_url FROM news_doc_rule LIMIT 10'):
            print(row)
    except:
        traceback.print_exc()
    print('OK')
    print(time.time() - start)
    # for row in db.execute("select pncode, region from region_pncode order by pncode"):
    #     pn = str(row['pncode'])
    # db.execute("""insert into room(room_id, name, last_message_id) values(%s, %s, %s)""", room_id, name, last_message_id)
