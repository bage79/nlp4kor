import traceback
import warnings

import paramiko

warnings.filterwarnings("ignore")


class SshUtil(object):
    """
    - Connect remote by SSH and run specific command.
    - See also `bage_util.SellUtil`
    """

    def __init__(self, username, password, hostname, port=22, log=None):
        self.client = None
        self.log = log
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.WarningPolicy())
            #            self.client.load_system_host_keys()
            #            keys = paramiko.util.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
            #            key = keys[hostname]
            #            print('keys:', keys
            #            self.client.get_host_keys().add(hostname, 'ssh-rsa', key)
            self.client.connect(hostname, port, username, password)
        except Exception:
            self.close()
            if self.log:
                self.log.error(traceback.format_exc())
            else:
                traceback.print_exc()

    def close(self):
        try:
            if self.client:
                self.client.close()
        except Exception:
            if self.log:
                self.log.error(traceback.format_exc)
            else:
                traceback.print_exc()

    def execute_n_print(self, command):
        _stdin, stdout, _stderr = self.client.exec_command(command)
        for line in stdout:
            print(line.strip('\n'))

    def execute(self, command):
        _stdin, stdout, _stderr = self.client.exec_command(command)
        return stdout

    def transfer(self, from_site, to_site):
        pass


if __name__ == '__main__':
    ssh = SshUtil('root', 'xxxx', 'server1')
    ssh.execute_n_print('ls -l')
