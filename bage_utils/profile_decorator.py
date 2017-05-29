import time
from functools import wraps

PROF_DATA = {}


def profile(fn):
    """
    - profiling number of called and execution time of a function.
    :param fn: 
    :return: 
    """

    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()
        ret = fn(*args, **kwargs)
        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling


def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print("Function %s called %d times. " % (fname, data[0]), )
        print('Execution time max: %.3f, average: %.3f' % (avg_time, max_time))


def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}


if __name__ == '__main__':
    @profile
    def __test_fn_a():
        i = 0
        while i < 100000:
            i += 1


    @profile
    def __test_fn_b():
        __test_fn_a()


    __test_fn_a()
    __test_fn_b()
    print_prof_data()
