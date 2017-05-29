import getpass
import inspect
import os.path
import platform
import socket


def real_path(filepath=None):
    """ 이 함수를 호출한 소스의 상대 경로와 filepath의 상대경로를 조합하여, 절대 경로를 생성함. (shell command 용) """
    frame = inspect.stack()[1]
    caller = inspect.getmodule(frame[0])
    if filepath:
        return os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(caller.__file__)), filepath))
    else:
        return os.path.realpath(caller.__file__)


# noinspection PyUnresolvedReferences
def get_local_address():
    if not hasattr(get_local_address, 'LOCAL_ADDRESS'):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('gmail.com', 80))
        setattr(get_local_address, 'LOCAL_ADDRESS', s.getsockname()[0])
        s.close()
    return get_local_address.LOCAL_ADDRESS


def get_username():
    """shell 로그인 계정"""
    return getpass.getuser()


def is_osx_os():
    """ Mac """
    return platform.system().lower() == 'darwin'


def is_linux_os():
    """ Linux """
    return platform.system().lower() == 'linux'


def is_windows_os():
    """ Windows """
    return platform.system().lower() == 'windows'


def is_my_pc():
    """ 맥이면 내 PC로 판단함."""
    return is_osx_os()


def is_server():
    """ ubuntu or windows는 서버로 판단함."""
    return not is_my_pc()


def hostname():
    """ hostname 확인 (linux, max 정상 동작. windows 확인 필요 """
    return socket.gethostname()


def api_server_hostname():
    """ 접속 가능한 API서버의 hostname을 리턴 """
    if is_my_pc():
        return 'localhost'  # test server
    else:
        return 'api-local'


def db_hostname():
    """ DB 서버 hostname 조회 """
    if is_my_pc():
        return 'localhost'  # test server
    else:
        return 'db-local'


def elasticsearch_hostname():
    return 'db-local'


def neo4j_hostname():
    return 'db-local'


def redis_hostname():
    return 'db-local'


def hts_server_hostname():
    """ HTS(windows) hostname 조회 """
    if is_osx_os():  # I'm paralles host
        return 'vm_guest'
    elif is_windows_os():  # I'm HTS server
        return 'localhost'
    else:  # I'm ubuntu
        return 'hts-local'  # 호스트에서 port-forwarding으로 게스트 포트 열어줌. 10.211.55.4(parallels),    10.0.2.1(virtualbox)


if __name__ == '__main__':
    print(hostname())
    print(get_local_address())
    # print(platform.system().lower())
    print(real_path('logs'))
