import os
import signal
import subprocess


class ShellUtil(object):
    """
    - run shell command.
    - see also `bage_util.SshUtil`
    """

    @staticmethod
    def kill_processes(cmd):
        p = subprocess.Popen(['ps', '-ef'], stdout=subprocess.PIPE)
        out, _err = p.communicate()
        for line in out.splitlines():
            if line.count(' ' + cmd) > 0:
                pid = int(line.split(None, 2)[1])
                os.kill(pid, signal.SIGKILL)

    @staticmethod
    def call(cmd):
        try:
            p = subprocess.Popen(cmd.split(' ', 1), stdout=subprocess.PIPE)
            out, _err = p.communicate()
            lines = []
            for line in out.splitlines():
                lines.append(line.decode('utf8'))
            return '\n'.join(lines)
        except FileNotFoundError as e:
            print("FileNotFoundError: ", e)


if __name__ == '__main__':
    # ShellUtil.kill_processes('vim')
    print(ShellUtil.call('ps -rf'))
    # ShellUtil.call('1')
