import socket


class DnsUtil(object):
    @staticmethod
    def domain2ip(domain):
        return socket.gethostbyname(domain)
