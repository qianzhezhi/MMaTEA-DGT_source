from socket import *
from socketserver import BaseRequestHandler, TCPServer
from threading import Thread

ip = "localhost"
port = 19999

class Handler(BaseRequestHandler):

    def handle(self):
        print('connection init')
                
        msg = self.request.recv(8012)
        print(f'[{self.client_address}] Received: {msg}')

        # response the result
        resp = '1.234'
        print(f'[{self.client_address}] Response: {resp}')
        self.request.sendall(f'{str(resp)}\n\r'.encode('utf-8'))


if __name__ == '__main__':

    socket_server = socket(AF_INET, SOCK_STREAM)
    socket_server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

    NWORKERS = 5

    serv = TCPServer((ip, port), Handler)
    print('server simple start')
    for n in range(NWORKERS):
        t = Thread(target=serv.serve_forever)
        t.daemon = True
        t.start()
    serv.serve_forever()