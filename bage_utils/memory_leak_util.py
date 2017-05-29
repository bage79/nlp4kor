import gc
import linecache
import os

import objgraph

from bage_utils._memory_util import MemoryUtil
from bage_utils.base_util import get_username
from bage_utils.num_util import NumUtil


class MemoryLeakUtil(object):
    def __init__(self, output_path='/Users/%s/Downloads' % get_username(), enable=True, show_lines=False):
        self.enable = enable
        if self.enable:
            self.output_path = output_path
            self.last_garbage_len = 0
            if not os.path.isdir(self.output_path):
                os.mkdir(self.output_path)
            gc.enable()
            gc.set_debug(gc.DEBUG_SAVEALL)
            self.show_lines = show_lines
            if self.show_lines:
                self.total_lines = []
            else:
                self.total_lines = None
            self.total_leaks_count = 0
            self.memory = MemoryUtil()
            self.total_increased_bytes = 0

    def gabage_len(self):
        if self.enable:
            return gc.collect()
            # return len(gc.garbage)

    def start(self):
        if self.last_garbage_len == 0:
            self.last_garbage_len = self.gabage_len()

    def check(self, filepath, lineno):
        """
        2번 이상 memory leak 이 발생한 경우만 저장한다.
        따라서, 테스트코드에서 같은 명령을 2회 이상 실행해야 한다.
        :param filepath:
        :param lineno:
        :return:
        """
        if self.enable:
            lineno = int(lineno)
            garbage_len = self.gabage_len()
            leaks_count = garbage_len - self.last_garbage_len
            self.last_garbage_len = garbage_len

            # print('total_byes:', self.memory.total_memory())
            if leaks_count > 0:  # detect memory leak
                increased_bytes = self.memory.increased_bytes()
                print('increased_bytes:', increased_bytes)
                self.total_leaks_count += leaks_count
                self.total_increased_bytes += increased_bytes
                if self.show_lines:
                    line = 'leaks: %s bytes(%s)\n' % (
                        NumUtil.comma_str(increased_bytes), NumUtil.comma_str(leaks_count))
                    for i in range(lineno - 1, lineno):
                        # print(linecache.getline(filepath, i).strip())
                        line += '\tincreased:%s bytes\t%s:%s\t%s\n' % (
                            NumUtil.comma_str(increased_bytes), filepath, i, linecache.getline(filepath, i).strip())
                    self.total_lines.append(line)
                return increased_bytes
        return 0

    def summary(self):
        print('total_byes:', self.memory.total_memory())
        if self.enable:
            if self.show_lines:
                summary = '[leak summary]\ntotal bytes: %s, total lines: %s, increased: %s bytes\n' % (
                    NumUtil.comma_str(self.memory.total_memory()), len(self.total_lines),
                    NumUtil.comma_str(self.total_increased_bytes))
                return '\n'.join(self.total_lines) + summary
            else:
                summary = '[leak summary]\ntotal bytes: %s, total increased: %s bytes\n' % (
                    NumUtil.comma_str(self.memory.total_memory()), NumUtil.comma_str(self.total_increased_bytes))
                return summary
        else:
            return ''

    def save_graph(self, obj_name, max_depth=5):
        if self.enable:
            if type(obj_name) is not str:
                print('input class name into "obj_name" parameter.')
                return
            refs_filepath = os.path.join(self.output_path, obj_name + '.refs..png')
            backrefs_filepath = os.path.join(self.output_path, obj_name + '.backrefs..png')
            objgraph.show_backrefs(objgraph.by_type(obj_name), max_depth=max_depth, filename=backrefs_filepath)
            objgraph.show_refs(objgraph.by_type(obj_name), max_depth=max_depth, filename=refs_filepath)


if __name__ == '__main__':
    pass
    # print('graph written to "%s".' % filepath)
    # objgraph.show_most_common_types(limit=3)

    # print('\nleaking_objects')
    # roots = objgraph.get_leaking_objects()
    # print(len(roots))
    # print(roots[0])
    # objgraph.show_most_common_types(objects=roots, limit=3)
    # objgraph.show_refs(roots[:3], refcounts=True, filename='/Users/%s/Downloads/roots.png' % get_username())

    # objgraph.show_growth()
    # objgraph.show_chain(
    #     objgraph.find_backref_chain(
    #         random.choice(objgraph.by_type('MyBigFatObject')),
    #         objgraph.is_proper_module),
    #     filename='~/Downloads/chain2.png')
