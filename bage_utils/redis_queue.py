from bage_utils.base_util import redis_hostname
from bage_utils.redis_util import RedisUtil


# noinspection PyMethodMayBeStatic
class RedisQueue(object):
    """
    - queue with `Redis`
    """

    def __init__(self, _redis, queue_name, len_max=100):
        self.MESSAGE_STOP = '[STOP][%s]' % queue_name
        self.redis = _redis
        self.queue_name = queue_name
        self.stop = False
        self._len_max = len_max

    def __repr__(self):
        return 'RedisQueue queue_name:{queue_name}, len(cur/max):{len}/{len_man}, redis:{redis}'.format(
            redis=self.redis, queue_name=self.queue_name, len=self.len(), len_man=self._len_max)

    def put(self, value):
        self.redis.rpush(self.queue_name, value)

    def get(self):
        value = self.redis.lpop(self.queue_name)
        if value == self.MESSAGE_STOP:
            self.stop = True
        return value

    def can_put(self):
        if self.redis.llen(self.queue_name) < self._len_max:
            return True
        else:
            return False

    def len_max(self):
        return self._len_max

    def is_stop(self):
        return self.stop

    def is_empty(self):
        return self.redis.llen(self.queue_name) == 0

    def put_stop(self):
        self.put(self.MESSAGE_STOP)

    def len(self):
        return self.redis.llen(self.queue_name)

    def clear(self):
        self.redis.delete(self.queue_name)


if __name__ == '__main__':
    queue = RedisQueue(RedisUtil.get_client(host=redis_hostname()), 'test_queue')
    queue.put('test_value')
    print('len(queue):', queue.len())

    value = queue.get()
    print('value:', value)
    if queue.is_stop():
        print('queue stoped.')
    elif queue.is_empty():
        print('queue is empty')

    print('len(queue):', queue.len())
    if queue.is_stop():
        print('queue stoped.')
    elif queue.is_empty():
        print('queue is empty')
