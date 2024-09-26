from socket import *
from socketserver import BaseRequestHandler, TCPServer
from multiprocess import Process
import time
from datetime import datetime
from evaluator import process, evaluate

# server setting
ip = 'localhost'
port = 20001

parallel = 5

# evaluate cache
evaluators = {}

def process_data(raw_data):
    variables, city, risk_type = process(raw_data)
    result = evaluate(evaluators, variables, risk_type, city, variant='omicron')
    return result


def recv_data(new_socket, client_info):
    start = time.time()
    print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Connection from {client_info}')

    raw_data = bytearray()

    while True:
        raw_data.clear()
        while True:
            raw_tmp = new_socket.recv(1024)
            raw_data += raw_tmp
            # end with LF
            if len(raw_tmp) == 0 or raw_tmp[-1] == 10:
                break

        if raw_data.startswith(b'exit'):
            print('received exit')
            break

        print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Received: {len(raw_data)}')

        result = process_data(raw_data)

        print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Response: {result}')
        new_socket.sendall(f'{str(result)}\n\r'.encode('utf-8'))
        

    print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Connection close ({time.time() - start} s)')
    new_socket.close()
    return 0

def main():
    socket_server = socket(AF_INET, SOCK_STREAM)
    socket_server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    socket_server.bind((ip, port))
    socket_server.listen(parallel)
    print(f'server is runnning on {ip}:{port}')
    while True:
        new_socket, client_info = socket_server.accept()
        p = Process(target=recv_data, args=(new_socket, client_info))
        p.start()
        new_socket.close()
        

if __name__ == '__main__':

    main()


