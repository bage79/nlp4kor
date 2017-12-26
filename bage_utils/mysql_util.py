import os
import traceback
import numpy as np
import pymysql


class MySQLUtil(object):
    """
    - mysql client wrappter with `pymysql`
    - .bashrc 또는 .bashprofile 에 MYSQL_PASSWD 를 설정해야 함.
    """

    def __init__(self, host, user, passwd, db, port=3306, charset='UTF8', auto_connect=True):
        self.init_command = None
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db
        self.port = int(port)
        self.charset = charset.replace('-', '').upper()
        self.conn = None
        self.cursor = None
        self.queries = []
        if auto_connect:
            self.connect()

    def connect(self):
        if not self.cursor or not self.cursor.connection:
            self.init_command = 'SET NAMES %s' % self.charset
            self.conn = pymysql.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db, port=self.port, charset=self.charset,
                                        init_command=self.init_command, autocommit=True)
            self.cursor = self.conn.cursor(pymysql.cursors.DictCursor)

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

    @classmethod
    def mysql_type2numpy_type(cls, mysql_type):
        if mysql_type is not None:
            mysql_type = mysql_type.lower()
            if mysql_type.startswith('int') or mysql_type.startswith('bigint'):
                return np.int64
            elif mysql_type.startswith('float'):
                return np.float64
            else:
                return np.unicode_
        else:
            return None

    @classmethod
    def columns2numpy_types(cls, mysql, table_name) -> (list, list):
        columns, types = [], []
        for row in mysql.select('SHOW FIELDS FROM `%s`' % table_name):
            columns.append(row['Field'])
            types.append(cls.mysql_type2numpy_type(row['Type']))
        return columns, types

    @staticmethod
    def addslashes(field):
        return field.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')

    def bulk_execute(self, query, bulk_size=100, force_execute=False):
        if force_execute:
            full_query = ';'.join(self.queries)
            if full_query and len(full_query) > 0:
                # print(full_query)
                self.execute(full_query)
        else:
            self.queries.append(query)

            # print(len(self.queries))
            if len(self.queries) >= bulk_size:
                full_query = ';'.join(self.queries)
                # print(full_query)
                self.execute(full_query)
                self.queries = []

    def execute(self, query):
        if not self.cursor:
            self.connect()
        try:
            self.cursor.execute(query)
        except pymysql.OperationalError or pymysql.err.InterfaceError:
            self.connect()  # reconnect
            self.cursor.execute(query)
        except Exception as e:
            raise e

    def select_one(self, query):
        if not self.cursor:
            self.connect()
        try:
            self.cursor.execute(query)
            return self.cursor.fetchone()
        except pymysql.OperationalError or pymysql.err.InterfaceError:
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
        except pymysql.OperationalError or pymysql.err.InterfaceError:
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
    __MYSQL = {
        'host': '192.168.0.12',
        'user': 'root',
        'passwd': os.getenv('MYSQL_PASSWD'),
        'db': 'test',
        'port': 3306,
        'charset': 'utf8'
    }
    print(__MYSQL)
    try:
        mysql = MySQLUtil(**__MYSQL)
        for row in mysql.select('SELECT * FROM test_table'):
            print(row)
    except:
        traceback.print_exc()
