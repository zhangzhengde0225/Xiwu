import socket

import psutil
import socket

def get_ip_addresses():
    ip_addresses = {}
    addrs = psutil.net_if_addrs()
    for interface, iface_addrs in addrs.items():
        for addr in iface_addrs:
            if addr.family == socket.AF_INET:
                ip_addresses[interface] = addr.address
    return ip_addresses

ip_addresses = get_ip_addresses()
for interface, ip in ip_addresses.items():
    print(f"Interface {interface} has IP address {ip}")
exit()

def get_ip_addresses():
    ips = []
    hostnames = [hostname for hostname in socket.gethostbyname_ex(socket.gethostname())[2] if not hostname.startswith("127.")][:1]
    # print(hostnames)
    for hostname in hostnames:
        ips.append(hostname)
    return ips

ips = get_ip_addresses()
print("IP addresses:", ips)

exit()

def auto_worker_address(worker_address, host, port):
    import socket
    if worker_address != 'auto':
        return worker_address
    if host in ['localhost', '127.0.0.1']:
        return f'http://{host}:{port}'
    elif host == '0.0.0.0':
        ## TODO，此处需要改进，获取本机ip
        # 获取本机的外部 IP 地址是使用一个与外部世界的连接
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # s.connect(("10.255.255.255", 1))
            s.connect(("172.16.255.255", 1))
            ip = s.getsockname()[0]
        return f'http://{ip}:{port}'
    else:
        raise ValueError(f'host {host} is not supported')
    
    
if __name__ == '__main__':
    worker_address = 'auto'
    host = '0.0.0.0'
    res = auto_worker_address(worker_address, host, port=42900)
    print(res)