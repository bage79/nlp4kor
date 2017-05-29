import os

from pymongo import MongoClient, ASCENDING

from bage_utils.string_util import StringUtil


class MongodbUtil(object):
    """
    - .bashrc 또는 .bashprofile 에 MYSQL_PASSWD 를 설정해야 함.
    """

    def __init__(self, mongo_url, db_name, collection_name, auto_connect=False):
        """
        :param mongo_url: host, port, username, password, auth db
        :param db_name: database name
        :param collection_name: collection name
        :param auto_connect: default do not connect for multiprocessing (http://api.mongodb.com/python/current/faq.html#using-pymongo-with-multiprocessing)
        """
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.collection_name = collection_name
        self.auto_connect = auto_connect
        self.collection = MongoClient(mongo_url, socketKeepAlive=True, connect=auto_connect)[db_name][collection_name]

    def __repr__(self):
        return '%s (db_name:%s, collection_name:%s, auto_connect:%s)' % (
            StringUtil.mask_passwd_in_url(self.mongo_url), self.db_name, self.collection_name, self.auto_connect)

    def __str__(self):
        return self.__repr__()

    def find(self, query=None, sort=None, limit=0):
        if query is None:
            query = {}
        if sort is None:
            sort = [('_id', ASCENDING)]

        for row in self.collection.find(query, no_cursor_timeout=True).sort(sort).limit(limit):
            yield row

    def count(self, query=None):
        if query is None:
            query = {}
        return self.collection.count(query, no_cursor_timeout=True)

    def find_one(self, query: dict, limit=0) -> dict:
        if limit > 0:
            return self.collection.find_one(query, no_cursor_timeout=True).limit(limit)
        else:
            return self.collection.find_one(query, no_cursor_timeout=True)

    def create_index(self, field_list=None, unique=False):
        if field_list is None:
            field_list = []
        for field in field_list:
            self.collection.create_index([(field, ASCENDING)], background=True, unique=unique)
        return

    def insert(self, row: dict):
        return self.collection.insert_one(row)

    def update_one(self, where_query: dict, update_content: dict, upsert=False):
        return self.collection.update_one(
            where_query,
            update_content,
            upsert=upsert
        )

    def update(self, where_query: dict, update_content: dict, upsert=False):
        return self.collection.update_many(
            where_query,
            update_content,
            upsert=upsert
        )

    def save(self, row):
        return self.collection.save(row)

    def delete(self, where_query: dict):
        result = self.collection.delete_one(where_query)
        if result:
            return result.deleted_count
        return 0

    def drop(self):
        return self.collection.drop()


if __name__ == '__main__':
    MONGO_URL = r'mongodb://%s:%s@%s:%s/%s?authMechanism=MONGODB-CR' % (
        'root', os.getenv('MONGODB_PASSWD'), 'localhost', '27017', 'admin')
    mongo = MongodbUtil(MONGO_URL, 'root', 'test')
    mongo.create_index(['title', 'content'])

    # docs = [{'_id': 1, 'url':'', 'title': '테스트', 'content': '내용'}, {'_id': 2, 'title': '테스트', 'content': '내용2'}]
    # for doc in docs:
    #     mongo.insert(doc)
    # for row in mongo.find({'title': '테스트'}, limit=10):
    #     print(row)
    # mongo.update({'_id': 2}, {'title': 'test', 'content': '내용 테스트'})

    # query = "{'url': 'http://news.mk.co.kr/newsRead.php?sc=30000016&year=2016&no=603924'}"
    query = [{"title": ""}, {"content": ""}]
    for row in mongo.find(query, limit=10):
        print(row)
