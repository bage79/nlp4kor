# import eventlet; eventlet.monkey_patch()
import socket

import eventlet


class EventletTcpUtil(object):
    def __init__(self, host, port, sock=None, address=None, timeout=10, buffer_size=4096):
        self.host, self.port = host, port
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.address = address
        if sock:
            self.sock = sock
        else:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            except Exception as e:
                raise e

    @staticmethod
    def listen(port, handler):  # __echo_handler is a sample handler
        server = eventlet.listen(('0.0.0.0', port))
        pool = eventlet.GreenPool()
        while True:
            try:
                sock, address = server.accept()
                #            print(sock, address
                pool.spawn(handler, EventletTcpUtil(sock, address))
            except Exception as e:
                raise e

    def connect(self):
        try:
            if self.host and self.port:
                self.sock.connect((self.host, self.port))
        except Exception as e:
            raise e

    def write(self, packet):
        try:
            #            self.sock.sendall(bytearray(packet, "utf-8"))
            self.sock.sendall(packet)
            return True
        except Exception as e:
            raise e

    def read_bytes(self, length=1):
        try:
            data = ''
            #            self.sock.setblocking(False)
            while length > 0:
                part = self.sock.recv(length)
                data += part
                length -= len(part)
            # print('data: %s, length: %s, part: %s' % (data, length, part)
            return data
        except Exception as e:
            raise e

    def read_to_delim(self, delim='\n'):
        try:
            _buffer = ''
            data = True
            while data:
                data = self.sock.recv(self.buffer_size)
                _buffer += data

                while _buffer.find(delim) != -1:
                    line, _buffer = _buffer.split(delim, 1)
                    yield line
            return
        except Exception as e:
            raise e

    def readlines(self):
        return self.read_to_delim('\r\n')

    def close(self):
        try:
            self.sock.close()
        except Exception as e:
            raise e


def __client_sample():
    """ example for client """
    tcp = EventletTcpUtil('www.naver.com', 80)
    tcp.connect()
    tcp.write("GET %s HTTP/1.0\r\nHost: %s\r\n\r\n" % ('/', 'www.naver.com'))
    for line in tcp.read_to_delim():
        print(line)
    tcp.close()


def __server_sample():
    def __echo_handler(tcp):
        for line in tcp.readlines():
            if line.strip().lower() == 'quit':
                print("client quit")
                break
            tcp.write(line)
            tcp.write('\n')
            print(("echoed %r" % line))

    """ example for server """
    EventletTcpUtil.listen(port=7942, handler=__echo_handler)


if __name__ == '__main__':
    #    __client_sample()
    __server_sample()
    pass
