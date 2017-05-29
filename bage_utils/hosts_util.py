import re


class HostsUtil(object):
    __HOST2IP = {}
    __IP2HOST = {}

    @classmethod
    def __host2ip_from_hosts_file(cls):
        if len(cls.__HOST2IP.keys()) > 0 and len(cls.__IP2HOST.keys()) > 0:
            return

        with open('/etc/hosts', 'r') as f:
            for line in f.xreadlines():
                if line.startswith('#'):  # line.startswith(':')
                    continue

                line = line.rstrip('\n')  # remove new line characters.
                line = line.replace(' ', '\t')  # replace blank with tab char.

                line = re.sub('\t{2,}', '\t', line)  # replace multiple tabs with single tab char.
                line = line.strip('\t')  # remove first or last tab char in line.

                tokens = line.split('\t')
                if len(tokens) >= 2:
                    ip = tokens[0]
                    for i in range(1, len(tokens)):
                        host = tokens[i]
                        print(ip, '->', host)
                        if host not in cls.__HOST2IP:
                            cls.__HOST2IP[host] = ip
                        if ip not in cls.__IP2HOST:  # 줄 첫번째 항목으로 지정
                            cls.__IP2HOST[ip] = host

    @classmethod
    def get_ip(cls, host):
        cls.__host2ip_from_hosts_file()
        return cls.__HOST2IP.get(host, None)

    @classmethod
    def get_host(cls, ip):
        cls.__host2ip_from_hosts_file()
        return cls.__IP2HOST.get(ip, None)


if __name__ == '__main__':
    print('hostname of %s is %s.' % ('127.0.0.1', HostsUtil.get_host('127.0.0.1')))
    print('ip of %s is %s.' % ('localhost', HostsUtil.get_ip('localhost')))
