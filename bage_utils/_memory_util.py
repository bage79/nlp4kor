"""
need more test..
"""

import platform
import resource


# gc.set_debug(gc.DEBUG_LEAK)

class MemoryUtil(object):
    def __init__(self, enable=True):
        self.last_total_memory = self.total_memory()
        # self.__enable = enable
        # self.__shots = []
        # self.init_memory = self.total_memory()
        # if self.__enable:
        #     tracemalloc.start()

    def total_memory(self):
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

        # def snapshot(self, label=''):  # FIXME: snapshot 자체의 메모리 용량을 제외 해야 함.
        #     label = str(label)
        #     if self.__enable:
        #         total_memory = self.total_memory()
        #         print(label, 'shots:', sys.getsizeof(self.__shots), 'total_memory:', total_memory)
        #         self.__shots.append((label, total_memory, tracemalloc.take_snapshot()))
        #
        # def summary(self, show_lines=False, lines_per_leak=10):  # FIXME: snapshot 자체의 메모리 용량을 제외 해야 함.
        #     if self.__enable:
        #         s = '[memory snashots]\n'
        #         total_leak_memory = 0
        #         total_lines = 0
        #         for i in range(1, len(self.__shots)):
        #             label, total_memory, shot = self.__shots[i]
        #             label_, total_memory_, shot_ = self.__shots[i - 1]
        #             leak_memory = total_memory - total_memory_
        #             total_lines += 1
        #             total_leak_memory += leak_memory
        #             if show_lines and (leak_memory > 0):
        #                 s += '"%s" vs "%s" \n' % (label, label_)
        #                 s += '%s bytes = %s - %s \n' % (NumUtil.comma_str(leak_memory), NumUtil.comma_str(total_memory), NumUtil.comma_str(total_memory_))
        #                 for line in shot.compare_to(shot_, 'lineno')[:lines_per_leak]:
        #                     line = str(line)
        #                     if ('tracemalloc.py' not in line) and ('memory_util.py' not in line) and ('memory_leak_util.py' not in line) and ('linecache.py' not in line):
        #                         s += '\t%s\n' % line
        #                         try:
        #                             file_line = line.split(' ')[0]
        #                             filepath, lineno, _ = file_line.split(':')
        #                             s += '\t\t%s' % linecache.getline(filepath, int(lineno))
        #                         except:
        #                             pass
        #                 s += '\n'
        #         s += 'total lines: %s \n' % NumUtil.comma_str(total_lines)
        #         s += 'total leak: %s bytes \n' % NumUtil.comma_str(total_leak_memory)
        #         return s
        #     else:
        #         return ''
        #
        # def __display_top(self, snapshot, group_by='lineno', limit=10):
        #     s = ''
        #     snapshot = snapshot.filter_traces((
        #         tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        #         tracemalloc.Filter(False, "<unknown>"),
        #     ))
        #     top_stats = snapshot.statistics(group_by)
        #
        #     s += "Top %s lines" % limit + '\n'
        #     for index, stat in enumerate(top_stats[:limit], 1):
        #         frame = stat.traceback[0]
        #         # replace "/path/to/module/file.py" with "module/file.py"
        #         filename = tracemalloc.os.sep.join(frame.filename.split(os.sep)[-2:])
        #         s += "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024) + '\n'
        #         line = linecache.getline(frame.filename, frame.lineno).strip()
        #         if line:
        #             s += '    %s' % line + '\n'
        #
        #     other = top_stats[limit:]
        #     if other:
        #         size = sum(stat.size for stat in other)
        #         s += "%s other: %.1f KiB" % (len(other), size / 1024) + '\n'
        #     total = sum(stat.size for stat in top_stats)
        #     s += "Total allocated size: %.1f KiB" % (total / 1024) + '\n'


if __name__ == '__main__':
    pass
    # memory = MemoryUtil(check_memory_leak=False)
    # memory.snapshot('start')
    # print('\nafter')
    # memory.snapshot('end')
    # memory.snapshot('after gc.collect()')
    # print(memory.summary())
