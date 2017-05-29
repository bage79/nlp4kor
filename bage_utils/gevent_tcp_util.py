from gevent import monkey

monkey.patch_all()
import traceback
import socket
from gevent.server import StreamServer


class GeventTcpUtil(object):
    def __init__(self, sock=None, address=None, timeout=10, buffer_size=4096, log=None):
        self.timeout = timeout
        self.log = log
        self.buffer_size = buffer_size
        self.address = address
        if sock:
            self.sock = sock
        else:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(timeout)
            except:
                if self.log:
                    self.log.error(traceback.format_exc())

    @staticmethod
    def listen(port, handler):  # __echo_handler is a sample handler
        server = StreamServer(('0.0.0.0', port), handler)
        server.serve_forever()

    def connect(self, host, port):
        try:
            if self.log:
                self.log.info('connect to (host=%s, port=%d)...' % (host, port))
            self.sock.connect((host, port))
            if self.log:
                self.log.info('connect OK.')
        except:
            if self.log:
                self.log.error(traceback.format_exc())

    def write(self, packet):
        try:
            #            for p in packets:
            self.sock.sendall(bytearray(packet, "utf-8"))
        except:
            if self.log:
                self.log.error(traceback.format_exc())

    def read_bytes(self, length=1):
        try:
            data = ''
            while length > 0:
                part = self.sock.recv(length)
                data += part
                length -= len(part)
            return data
        except:
            if self.log:
                self.log.error(traceback.format_exc())

    def read_to_delim(self, delim='\n'):
        _buffer = ''
        data = True
        while data:
            data = self.sock.recv(self.buffer_size)
            _buffer += data

            while _buffer.find(delim) != -1:
                line, _buffer = _buffer.split(delim, 1)
                yield line
        return

    def readlines(self):
        return self.read_to_delim('\r\n')

    def close(self):
        try:
            self.sock.close()
        except:
            if self.log:
                self.log.error(traceback.format_exc())


def __client_sample():
    """ example for client """
    tcp = GeventTcpUtil()
    tcp.connect('www.naver.com', 80)
    tcp.write("GET %s HTTP/1.0\r\nHost: %s\r\n\r\n" % ('/', 'www.naver.com'))
    for line in tcp.read_to_delim():
        print(line)
    tcp.close()


def __server_sample():
    def __echo_handler(sock, address):
        tcp = GeventTcpUtil(sock, address)
        for line in tcp.readlines():
            if line.strip().lower() == 'quit':
                print("client quit")
                break
            tcp.write(line)
            tcp.write('\n')
            print(("echoed %r" % line))

    """ example for server """
    GeventTcpUtil.listen(port=7942, handler=__echo_handler)


if __name__ == '__main__':
    #    __client_sample()
    __server_sample()
    pass
