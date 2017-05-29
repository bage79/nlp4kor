import datetime
import re
import time


class DateUtil(object):
    DATE_START = '2000-01-01'  # 이전의 데이터는 무시함.
    DATE_END = '2016-11-30'

    DATETIME_START = '2000-01-01 23:59:59'  # 이전의 데이터는 무시함.
    DATETIME_END = '2016-09-30 23:59:59'

    @staticmethod
    def yyyymmdd2mysql_date(yyyymmdd):
        if len(yyyymmdd) == 8:
            return '%4d-%02d-%02d' % (int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:]))
        else:
            return ''

    @staticmethod
    def mysql_date2yyyymmdd(mysql_date):
        tokens = mysql_date.split('-')
        if len(mysql_date) == 10 and len(tokens) == 3:
            return '%4d%02d%02d' % (int(tokens[0]), int(tokens[1]), int(tokens[2]))
        else:
            return ''

    @staticmethod
    def mysql_date2date(mysql_date):
        tokens = mysql_date.split('-')
        if len(mysql_date) == 10 and len(tokens) == 3:
            return datetime.date(int(tokens[0]), int(tokens[1]), int(tokens[2]))
        else:
            return datetime.date.today()

    @staticmethod
    def secs_to_string(secs):  # @ReservedAssignment
        secs = int(secs)
        mins, secs = divmod(secs, 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)
        if days > 0:
            return '%ddays %02d:%02d:%02d' % (days, hours, mins, secs)
        else:
            return '%02d:%02d:%02d' % (hours, mins, secs)

    @staticmethod
    def millisecs_to_string(_secs):  # @ReservedAssignment
        secs = int(_secs)
        mili_secs = float(_secs) - secs
        mins, secs = divmod(secs, 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)
        if days > 0:
            return '%ddays %02d:%02d:%02d %.4f' % (days, hours, mins, secs, mili_secs)
        else:
            return '%02d:%02d:%02d %.4f' % (hours, mins, secs, mili_secs)

    @staticmethod
    def current_datetime_string():
        now = time.localtime()
        return '%04d-%02d-%02d %02d:%02d:%02d' % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    @staticmethod
    def to_datetime_string(_datetime: datetime.datetime, format='%04d-%02d-%02d %02d:%02d:%02d'):
        return format % (
            _datetime.year, _datetime.month, _datetime.day, _datetime.hour, _datetime.minute, _datetime.second)

    @staticmethod
    def to_date_string(_date: datetime.date, format='%04d-%02d-%02d'):
        return format % (_date.year, _date.month, _date.day)

    @staticmethod
    def date2mysql_date(_date: datetime.date):
        return DateUtil.to_date_string(_date)

    @staticmethod
    def datetime2yyyymmdd(_datetime: datetime.datetime):
        return '%04d%02d%02d' % (_datetime.year, _datetime.month, _datetime.day)

    @staticmethod
    def date2yyyymmdd(_datetime: datetime.date):
        return '%04d%02d%02d' % (_datetime.year, _datetime.month, _datetime.day)

    @staticmethod
    def current_date_string():
        now = time.localtime()
        return '%04d-%02d-%02d' % (now.tm_year, now.tm_mon, now.tm_mday)

    @staticmethod
    def current_yyyymm():
        now = time.localtime()
        return '%04d%02d' % (now.tm_year, now.tm_mon)

    @staticmethod
    def current_yyyymmdd():
        now = time.localtime()
        return '%04d%02d%02d' % (now.tm_year, now.tm_mon, now.tm_mday)

    @staticmethod
    def current_yyyymmddhhmmss():
        now = time.localtime()
        return '%04d%02d%02d%02d%02d%02d' % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    @staticmethod
    def current_yyyymmdd_hhmm():
        now = time.localtime()
        return '%04d%02d%02d_%02d%02d' % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

    @staticmethod
    def current_hhmmss():
        now = time.localtime()
        return '%d:%d:%d' % (now.tm_hour, now.tm_min, now.tm_sec)

    @staticmethod
    def string_to_unixtime(date_str, time_format='%Y-%m-%d %H:%M:%S'):
        date = DateUtil.string_to_datetime(date_str, time_format)
        return int(date.timestamp())

    @staticmethod
    def current_hhmm00():
        now = time.localtime()
        return '%d:%d:00' % (now.tm_hour, now.tm_min)

    @staticmethod
    def current_millisecs():
        return int(round(time.time() * 1000))

    @staticmethod
    def current_microsecs():
        return int(round(time.time() * 1000000))

    @staticmethod
    def current_unixtimestamp():
        return int(time.time())

    @staticmethod
    def is_mysql_date_format(date_str):
        try:
            DateUtil.string_to_date(date_str)
            return True
        except:
            return False

    @staticmethod
    def is_mysql_datetime_format(date_str):
        try:
            DateUtil.string_to_datetime(date_str)
            return True
        except:
            return False

    @staticmethod
    def string_to_datetime(date_str, time_format='%Y-%m-%d %H:%M:%S'):
        return datetime.datetime.strptime(date_str, time_format)

    @staticmethod
    def string_to_date(date_str, time_format='%Y-%m-%d'):
        return datetime.datetime.strptime(date_str, time_format).date()

    @staticmethod
    def now():
        return datetime.datetime.now()

    @staticmethod
    def today():
        return datetime.datetime.now().date()

    @staticmethod
    def is_valid_date_string(date_string, date_start='1960-01-01', date_end=None):
        if not date_end:
            date_end = DateUtil.current_date_string()

        try:
            match = re.match(
                '^(((\d{4})(-)(0[13578]|10|11|12)(-)(0[1-9]|[12][0-9]|3[01]))|((\d{4})(-)(0[469]|11)(-)([0][1-9]|[12][0-9]|30))|((\d{4})(-)(0[2])(-)(0[1-9]|1[0-9]|2[0-8]))|(([02468][048]00)(-)(02)(-)(29))|(([13579][26]00)(-)(02)(-)(29))|(([0-9][0-9][0][48])(-)(02)(-)(29))|(([0-9][0-9][2468][048])(-)(02)(-)(29))|(([0-9][0-9][13579][26])(-)(02)(-)(29)))$',
                date_string)
            if not match:
                return False

            # time.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")
            if date_start <= date_string <= date_end:
                return True
            else:
                return False
        except ValueError:
            print('ValueError')
            return False

    @staticmethod
    def is_valid_datetime_string(datetime_string, datetime_start='1960-01-01 00:00:00', datetime_end=None, delta: datetime.timedelta = None):
        if not datetime_string or len(datetime_string) == 0:
            return False

        if not datetime_end:
            if delta:
                datetime_end = DateUtil.to_datetime_string(datetime.datetime.now() + delta)
            else:
                datetime_end = DateUtil.current_datetime_string()
        try:
            match = re.match(
                '^(((\d{4})(-)(0[13578]|10|11|12)(-)(0[1-9]|[12][0-9]|3[01]))|((\d{4})(-)(0[469]|11)(-)([0][1-9]|[12][0-9]|30))|((\d{4})(-)(0[2])(-)(0[1-9]|1[0-9]|2[0-8]))|(([02468][048]00)(-)(02)(-)(29))|(([13579][26]00)(-)(02)(-)(29))|(([0-9][0-9][0][48])(-)(02)(-)(29))|(([0-9][0-9][2468][048])(-)(02)(-)(29))|(([0-9][0-9][13579][26])(-)(02)(-)(29)))(\s([0-1][0-9]|2[0-4]):([0-5][0-9]):([0-5][0-9]))$',
                datetime_string)
            if not match:
                return False

            # time.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")
            if datetime_start <= datetime_string <= datetime_end:
                return True
            else:
                return False
        except ValueError:
            print('ValueError')
            return False

    @staticmethod
    def weekday_string(date, lang='ko'):
        if lang == 'ko':
            WEEK_DAY = ['월', '화', '수', '목', '금', '토', '일']
        else:
            WEEK_DAY = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        return WEEK_DAY[date.weekday()]


if __name__ == '__main__':
    # print(DateUtil.string_to_datetime('2017-03-16', time_format='%Y-%m-%d'))
    # time = datetime.datetime.time(datetime.datetime.now())
    # print('time:', time)
    # print(time < datetime.time(0, 30))
    # print(datetime.time(11, 30) < time)
    date = datetime.datetime.now() + datetime.timedelta(days=1)
    print(DateUtil.to_datetime_string(date))
    # print(DateUtil.to_datetime_string(datetime.now()))
    # print(DateUtil.current_datetime_string())
    # print('1970-01-01 23:59:59', DateUtil.is_valid_datetime_string('1970-01-01 23:59:59', datetime_start='1990-01-01 23:59:59'))
    # print('1970-01-01 23:59:59', DateUtil.is_valid_datetime_string('1970-01-01 23:59:59'))
    # for datetime_string in ['xx', '9141-1-3 7:10:0', '2000-01-01 23:59:59', '1970-01-01 23:59:59', '2016-02-29 14:21:00']:
    #     print(datetime_string, DateUtil.is_valid_datetime_string(datetime_string, datetime_end='1990-01-01 23:59:59'))

    # print(DateUtil.string_to_unixtime('2017-01-01 00:00:00'))
    # print(DateUtil.current_yyyymmdd_hhmm())
    # @formatter:off
    # print(type(DateUtil.today()), DateUtil.today())
    # print(DateUtil.secs_to_string(1000))
    # print(DateUtil.current_time_string(), len(DateUtil.current_time_string()))
    # print(DateUtil.current_yyyymm(), len(DateUtil.current_yyyymm()))
    # print(DateUtil.current_yyyymmddhhmmss(), len(DateUtil.current_yyyymmddhhmmss()))
    # print(DateUtil.current_unixtimestamp(), len(str(DateUtil.current_unixtimestamp())))
    # print(DateUtil.current_millisecs(), len(str(DateUtil.current_millisecs())))
    # print(DateUtil.current_microsecs(), len(str(DateUtil.current_microsecs())))
    # d = DateUtil.string_to_date('2015-10-20 21:30:40')
    # print(type(d), d, '->', d.year, d.month, d.day, d.hour, d.minute, d.second, d.weekday())
    # print(type(datetime.now()))
    # print('%.10f' % time.time()
