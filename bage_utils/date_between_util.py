import datetime

from bage_utils.date_util import DateUtil


class DateBetweenUtil(object):
    """ between two days """

    def __init__(self, from_date: datetime.date, to_date: datetime.date):
        if not (isinstance(from_date, datetime.datetime) and isinstance(to_date, datetime.datetime)) and not (
                    isinstance(from_date, datetime.date) and isinstance(to_date, datetime.date)):
            raise Exception(
                "from_date, to_date must be datetime.datetime instances.")
        self.from_date = from_date
        self.to_date = to_date
        self.delta = self.to_date - self.from_date

    def delta(self):
        return self.delta

    def days(self):
        return self.delta.days

    def date_list(self):
        _list = []
        for i in range(self.delta.days + 1):
            date = self.from_date + datetime.timedelta(days=i)
            _list.append(date)
        return _list

    def date_split(self, max_days=30, from_start=True):
        li = []
        if from_start:
            start_date = self.from_date
            while start_date <= self.to_date:
                end_date = start_date + datetime.timedelta(days=max_days)
                if end_date > self.to_date:
                    end_date = self.to_date
                li.append((start_date, end_date))
                start_date = end_date + datetime.timedelta(days=1)
        else:
            end_date = self.to_date
            while end_date >= self.from_date:
                start_date = end_date - datetime.timedelta(days=max_days)
                if start_date < self.from_date:
                    start_date = self.from_date
                li.append((start_date, end_date))
                end_date = start_date - datetime.timedelta(days=1)
        return li


if __name__ == '__main__':
    # start_date, end_date = DateUtil.string_to_datetime('2015-11-29 00:00:00'), DateUtil.string_to_datetime('2016-01-01 00:00:00')
    start_date, end_date = DateUtil.string_to_date('2015-11-01'), DateUtil.string_to_date('2016-01-05')
    between = DateBetweenUtil(start_date, end_date)
    # for a, b in between.date_split(from_start=True):
    #     print(a, b)
    # print()
    for a, b in between.date_split(from_start=False):
        print(a, b)
        # between = DateBetweenUtil(datetime.datetime(2013, 1, 31, 1), datetime.datetime(2013, 2, 2, 4))
        # print(between.days())
        # for d in between.date_list():
        #     print(str(d), type(d))
        # print(between.hour_list())
        # check exception raised.
        # assert Exception, DateBetweenUtil(10, datetime.datetime(2013, 2, 2, 4))
