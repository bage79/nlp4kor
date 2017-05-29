import multiprocessing
import os
import platform

import psutil


class TasksetUtil(object):
    """ 
    - Linux taskset command for process priority. 
    """

    @staticmethod
    def get_rss_mb(pid):
        return float(psutil.Process(pid)._proc.memory_info()[0]) / 1024 / 1024

    @staticmethod
    def get_cpu_no(pid, log=None):
        cmd = 'taskset -cp %s' % pid
        if log:
            log.info(cmd)
        return int(os.popen(cmd).read().strip().split(':')[1].strip())  # include my process

    @staticmethod
    def get_process_count(program_filepath, log=None):
        cmd = 'ps aux | grep %s | grep python | grep -v grep | wc -l' % os.path.basename(program_filepath)
        if log:
            log.info(cmd)
        return int(os.popen(cmd).read().strip())  # include my process

    @staticmethod
    def set_my_process_on_new_cpu(program_filepath, target_cpu, log=None):  # 리눅스에서만 동작함. (mac에서는 실행 불가)
        """
        run a process on each cpu.
        :param log:
        :param target_cpu:
        :param program_filepath:
        :return: target_cpu number(0~).
        """
        if platform.system().lower() == 'darwin':  # OSX
            if log:
                log.info('This is OSX.')
            return 0

        if log:
            log.info('filepath: %s' % program_filepath)
        total_cpu = multiprocessing.cpu_count() - 1
        target_cpu %= total_cpu

        cmd = 'taskset -cp %d %d' % (target_cpu, os.getpid())
        if log:
            log.info('total cpu: %s' % total_cpu)
            log.info('target cpu: %s' % target_cpu)
            log.info(cmd)

        os.system(cmd)
        return target_cpu


if __name__ == '__main__':
    cpu_no = TasksetUtil.set_my_process_on_new_cpu(__file__, target_cpu=0, log=None)  # this process number
    print('cpu_no: %s' % cpu_no)
