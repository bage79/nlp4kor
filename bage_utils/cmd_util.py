import os


class CmdUtil(object):  # SshUtil 도 같은 기능을 한다. 다만 CmdUtil은 connectionless.
    def __init__(self, host=None, print_input=False):
        """
        원격 접속시, ssh로 암호 없이 접속하도록 설정 필요.
        https://opentutorials.org/module/432/3742
        :param host:
        :param print_input:
        """
        self.host = host
        self.print_input = print_input

    @staticmethod
    def __run_ssh(cmd):
        result = ''
        p = os.popen(cmd, "r")
        for line in p.readlines():
            result += line
        p.close()
        return result.rstrip('\n')

    def run(self, cmd: object, print_input: object = False) -> object:
        if self.host:  # remote
            cmd = cmd.replace("$", "\\$")
            cmd = cmd.replace(r'"', r'\"')
            #    cmd = cmd.replace("%", "%%")
            cmd = r"""ssh %s "%s" """ % (self.host, cmd)
            if self.print_input or print_input:
                print('[%s]' % self.host, cmd)
        else:  # local
            if self.print_input or print_input:
                print('[%s]' % 'localhost', cmd)
        return self.__run_ssh(cmd)


if __name__ == '__main__':
    print(CmdUtil('api-live').run('hostname', print_input=True))
    print(CmdUtil().run('hostname', print_input=True))
