"""
need more test..
"""

import platform
import resource


# gc.set_debug(gc.DEBUG_LEAK)

class MemoryUtil(object):
    def __init__(self, enable=True):
        self.last_total_memory = self.total_memory()

    @staticmethod
    def total_memory():
        total = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system().lower() == 'darwin':  # OSX
            total /= 1024
        return int(total)

    def increased_bytes(self):
        # print('memory increased: %s bytes' % (NumUtil.comma_str(self.total_memory() - self.last_total_memory)))
        total_memory = self.total_memory()
        increased = total_memory - self.last_total_memory
        self.last_total_memory = total_memory
        return increased


if __name__ == '__main__':
    pass
    # memory = MemoryUtil(check_memory_leak=False)
    # memory.snapshot('start')
    # print('\nafter')
    # memory.snapshot('end')
    # memory.snapshot('after gc.collect()')
    # print(memory.summary())
