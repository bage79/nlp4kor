import multiprocessing
import os
import time

from slackclient import SlackClient

from bage_utils.base_util import is_linux_os

lock = multiprocessing.Lock()


class SlackUtil(object):
    """
    - Slack client with `slackclient.SlackClient`
    - 메시지 전송전에 슬랙봇을 해당 채널에 초대 해야 함.
    - .bashrc 또는 .bashprofile 에 SLACK_API_TOKEN 를 설정해야 함.
    """
    DEFAULT_CHANNEL = 'general'
    API_TOKEN = os.getenv('SLACK_API_TOKEN')
    slack_client = SlackClient(API_TOKEN)

    @classmethod
    def send_message(cls, message, channel=DEFAULT_CHANNEL):
        """
        메시지 전송
        :param message: 메시지 
        :param channel: 채널
        :return: 
        """
        if is_linux_os():
            with lock:
                time.sleep(0.1)
                cls.slack_client.api_call("chat.postMessage", channel=channel, text=message)


if __name__ == '__main__':
    SlackUtil.send_message('...', channel='general')
