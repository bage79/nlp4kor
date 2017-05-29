import os
import pickle

import redis

from bage_utils.base_util import db_hostname


class RedisUtil(object):
    """
    - Redis Client Wrapper
    - .bashrc 또는 .bashprofile 에 REDIS_PASSWD 를 설정해야 함.
    """
    TEST_DB_NO = 99  # redic.conf 에서 DB 개수를 100개로 설정해야 함.

    @staticmethod
    def get_client(host='localhost', port=6379, db=TEST_DB_NO, password=os.getenv('REDIS_PASSWD'),
                   decode_responses=True, encoding='utf8'):
        pool = redis.ConnectionPool(host=host, port=port, db=db, password=password, decode_responses=decode_responses,
                                    encoding=encoding)
        return redis.Redis(connection_pool=pool)


if __name__ == '__main__':
    _redis = RedisUtil.get_client(host=db_hostname(), decode_responses=False)
    li = [1, 2, 3]
    _redis.rpush('test_queue', pickle.dumps(li))
    item = _redis.lpop('test_queue')
    print(pickle.loads(item))

    _redis = RedisUtil.get_client(host=db_hostname(), decode_responses=True)
    print(_redis.info())
    _redis.delete('test')
    _redis.hset('test', '1', 100)
    _redis.hset('test', '2', 200)
    _redis.hdel('test', '1')
    print(type(_redis.hkeys('test')))
    _redis.set('test', 10)
    value = _redis.get('test')
    print(type(value), value)
    _redis.delete('test')
