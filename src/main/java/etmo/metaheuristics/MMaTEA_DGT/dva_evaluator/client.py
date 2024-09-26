from socket import socket, AF_INET, SOCK_DGRAM

if __name__ == '__main__':
    s = socket(AF_INET, SOCK_DGRAM)

    arr = [1,2,3,4,5]

    print(s.sendto(bytes(arr), ('localhost', 20000)))
    
    print(s.recvfrom(8192))
    