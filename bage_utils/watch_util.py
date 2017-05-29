import time

from pytools.stopwatch import StopWatch

from bage_utils.date_util import DateUtil


class WatchUtil(object):
    """
    - wrapper of 'pytools.stopwatch.StopWatch`
    - support multiple watches.
    """
    DEFAULT_WATCH_NAME = ''  # ''__WATCH__'

    def __init__(self, auto_stop=True):
        """
        :param auto_stop: elapsed() 또는 elapsed_string() 호출시, 자동으로 stop() 호출됨.
        """
        self.__watches = {}
        self.__cnt = {}
        self.auto_stop = auto_stop

    def __get(self, name=DEFAULT_WATCH_NAME) -> StopWatch:
        if name not in self.__watches:
            self.__watches[name] = StopWatch()
            self.__cnt[name] = 0
        return self.__watches[name]

    def del_watch(self, name=DEFAULT_WATCH_NAME) -> bool:
        if name in self.__watches:
            del self.__watches[name]
            return True
        else:
            return False

    def start(self, name=DEFAULT_WATCH_NAME) -> StopWatch:
        self.__get(name).start()
        self.__cnt[name] += 1
        return self.__get(name)

    def stop(self, name=DEFAULT_WATCH_NAME) -> StopWatch:
        return self.__get(name).stop()

    def elapsed(self, name=DEFAULT_WATCH_NAME) -> int:
        if self.auto_stop:
            try:
                self.__get(name).stop()  # must call start() later.
            except AssertionError:
                return 0
        return self.__get(name).elapsed()

    def elapsed_string(self, name=DEFAULT_WATCH_NAME) -> str:
        return DateUtil.secs_to_string(self.elapsed(name))

    def summary(self, prefix=''):
        import operator
        li = [(name, self.__watches[name].elapsed()) for name in self.__watches]
        li = sorted(li, key=operator.itemgetter(1), reverse=True)
        s = ''
        # for name, total_milli_secs in li:
        #     s += 'total [%s] %s\n' % (DateUtil.millisecs_to_string(total_milli_secs), name)
        for name, total_milli_secs in li:
            if self.__cnt[name] > 0:
                if len(prefix) > 0:
                    s += '%s average [%s] %s\n' % (
                        prefix, DateUtil.millisecs_to_string(float(total_milli_secs) / float(self.__cnt[name])), name)
                else:
                    s += 'average [%s] %s\n' % (
                        DateUtil.millisecs_to_string(float(total_milli_secs) / float(self.__cnt[name])), name)
        return s


if __name__ == '__main__':
    watches = WatchUtil(auto_stop=True)

    watches.start('A')
    time.sleep(1)
    print(watches.elapsed_string('A'))

    watches.del_watch('A')
    watches.start('B')
    time.sleep(3)
    print(watches.elapsed_string('B'))

    watches.start('A')
    time.sleep(3)
    print(watches.elapsed_string('A'))

    watches.start('A')
    time.sleep(2)
    print(watches.elapsed_string('A'))

    print(watches.summary())
