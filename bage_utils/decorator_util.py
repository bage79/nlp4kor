import time


def try_except(f):
    def decorator(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise e

    return decorator


def elapsed(decimal_place=6):
    def __print_elapsed(f):
        def decorator(*args, **kwargs):
            start_time = time.time()
            result = f(*args, **kwargs)
            end_time = time.time()
            time_format = 'elapsed time: %%.%df secs' % decimal_place
            print(time_format % (end_time - start_time), 'in %s()' % f.__name__)
            return result

        return decorator

    return __print_elapsed


@elapsed(decimal_place=3)
def __sample():
    total = 0
    for i in range(0, 100000):
        total += i
    return total


@try_except
def loop():
    while True:
        time.sleep(1)
        print('running...')


if __name__ == '__main__':
    try:
        loop()
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    pass
