import datetime
import time


class TimerUtil(object):
    def __init__(self, interval_secs=1):
        self.interval_secs = interval_secs
        self.count = 0
        self.next_time = 0

    def start(self):
        self.next_time = datetime.datetime.now() + datetime.timedelta(seconds=self.interval_secs)
        return self

    def is_over(self):
        if datetime.datetime.now() > self.next_time:
            self.count += 1
            self.next_time = datetime.datetime.now() + datetime.timedelta(seconds=self.interval_secs)
            return True
        else:
            return False


if __name__ == '__main__':
    timer = TimerUtil(interval_secs=2).start()
    for i in range(100):
        time.sleep(1)
        print(timer.is_over())