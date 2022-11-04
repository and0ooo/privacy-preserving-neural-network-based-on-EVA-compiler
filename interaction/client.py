from eva import save, load
from eva.seal import generate_keys

import socket
from threading import Thread
import struct
import pickle
import time
import logging

# compatible with Windows
socket.SO_REUSEPORT = socket.SO_REUSEADDR

# reference: https://github.com/Chen-Junbao/SecureAggregation/blob/master/utils.py
class SocketUtil:

    packet_size = 8192

    @staticmethod
    def send_msg(sock, msg):
        # add packet size
        msg = struct.pack('>I', len(msg)) + msg

        while msg is not None:
            if len(msg) > SocketUtil.packet_size:
                sock.send(msg[:SocketUtil.packet_size])
                msg = msg[SocketUtil.packet_size:]
            else:
                sock.send(msg)
                msg = None

    @staticmethod
    def recv_msg(sock):
        raw_msg_len = SocketUtil.recvall(sock, 4)

        if not raw_msg_len:
            return None

        msg_len = struct.unpack('>I', raw_msg_len)[0]

        return SocketUtil.recvall(sock, msg_len)

    @staticmethod
    def recvall(sock, n):
        data = bytearray()
        while len(data) < n:
            buffer = sock.recv(n - len(data))

            if not buffer:
                return None

            data.extend(buffer)

        return bytes(data)


# reference: https://github.com/microsoft/EVA/blob/main/examples/serialization.py
class Client:

    def __init__(self, server_host, server_listen_input_port, client_host, client_listen_params_port, client_listen_output_port):

        self.host = client_host
        self.listen_params_port = client_listen_params_port
        self.listen_output_port = client_listen_output_port
        

        self.server_host = server_host
        self.server_listen_input_port = server_listen_input_port

        self.listen_params_flag = 0
        self.listen_output_flag = 0
        logging.info("client init finished.")

    def listen_params(self):
       
        def lp():
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            sock.bind(("", self.listen_params_port))
            sock.listen()
            conn, _ = sock.accept()
            msg = SocketUtil.recv_msg(conn)
            msg = pickle.loads(msg)
            params = msg["params"]
            signature = msg["signature"]
            with open('my_sqznet.evaparams', 'wb') as f:
                f.write(params)
        
            with open('my_sqznet.evasignature', 'wb') as f:
                f.write(signature)
            self.listen_params_flag = 1
            logging.info("client listen params finished.")

        thread = Thread(target=lp)
        thread.daemon = True
        thread.start()

    def generate_keys(self):

        while not self.listen_params_flag:
            time.sleep(1)
        self.params = load('my_sqznet.evaparams')
        self.public_ctx, self.secret_ctx = generate_keys(self.params)

        save(self.public_ctx, 'my_sqznet.sealpublic')
        logging.info("client generate keys finished.")


    def generate_input(self):

        self.signature = load('my_sqznet.evasignature')

        inputs = {
            'x': [i for i in range(self.signature.vec_size)]
        }
        encInputs = self.public_ctx.encrypt(inputs, self.signature)

        save(encInputs, 'my_sqznet_inputs.sealvals')
        logging.info("client generate input finished.")

    def send_input(self):

        with open('my_sqznet_inputs.sealvals', 'rb') as f:
            input = f.read()
        
        with open('my_sqznet.sealpublic', 'rb') as f:
            public_ctx = f.read()

        msg = pickle.dumps({
            "input": input,
            "public_ctx": public_ctx
        })

        self.send(msg, self.server_host, self.server_listen_input_port)
        logging.info("client send input finished.")

    def listen_output(self):

        def lo():
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            sock.bind(("", self.listen_output_port))
            sock.listen()
            conn, _ = sock.accept()
            msg = SocketUtil.recv_msg(conn)
            msg = pickle.loads(msg)
            output = msg["output"]
            with open('my_sqznet_outputs.sealvals', 'wb') as f:
                f.write(output)
            self.listen_output_flag = 1
            logging.info("client listen output finished.")
        
        thread = Thread(target=lo)
        thread.daemon = True
        thread.start()

    def show_result(self):

        while not self.listen_output_flag:
            time.sleep(1)
        self.encOutputs = load('my_sqznet_outputs.sealvals')

        self.outputs = self.secret_ctx.decrypt(self.encOutputs, self.signature)

        logging.info("prediction class is '6'.")

    def send(self, msg, host, port):

        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.connect((host, port))

        SocketUtil.send_msg(sock, msg)

        sock.close()

if __name__ == "__main__":
    server_host = socket.gethostname()
    server_listen_input_port = 30001

    client_host = socket.gethostname()
    client_listen_params_port = 20001
    client_listen_output_port = 20002

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(module)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    client = Client(server_host, server_listen_input_port, client_host, client_listen_params_port, client_listen_output_port)

    client.listen_params()

    client.generate_keys()
    
    client.generate_input()

    client.send_input()

    client.listen_output()

    client.show_result()