import os

from bage_utils.base_util import is_my_pc, get_username
from bage_utils.cmd_util import CmdUtil


class TelegramUtil(object):
    """
    - Telegram client with `telegram-cli`
    """

    @staticmethod
    def send_telegram(msg, username='길동_홍'):
        telegram_cli_path = '%s/tg/bin/telegram-cli' % os.getenv('HOME')
        server_pub_path = '%s/tg/tg-server.pub' % os.getenv('HOME')
        if is_my_pc():  # 내 컴퓨터에서 구동할 때
            telegram_cli_path = '%s/tg/bin/telegram-cli' % os.getenv('HOME')
            server_pub_path = '%s/tg/tg-server.pub' % os.getenv('HOME')

        cmd = r'''%s -k %s -W -e "msg %s %s" ''' % (telegram_cli_path, server_pub_path, username, msg)
        print('cmd:', cmd)
        CmdUtil().run(cmd)
        # exit_code = os.system(cmd.encode(encoding='utf8'))    # hang
        # subprocess.call(cmd.encode(encoding='utf8'), shell=False, timeout=2)
        # if exit_code == 0:
        #     return True
        # else:
        #     return False


if __name__ == '__main__':
    TelegramUtil.send_telegram('텔레그램 테스트...', username='길동_홍')
