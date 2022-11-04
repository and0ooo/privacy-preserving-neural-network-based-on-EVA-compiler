from eva import EvaProgram, Input, Output, save, load
from eva.ckks import CKKSCompiler

import socket
import pickle
import struct
from threading import Thread
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
class Server:

    def __init__(self, server_host, server_listen_input_port, client_host, client_listen_params_port, client_listen_output_port):

        self.host = server_host
        self.listen_input_port = server_listen_input_port

        self.client_host = client_host
        self.client_listen_params_port = client_listen_params_port
        self.client_listen_output_port = client_listen_output_port

        self.server_listen_input = 0
        logging.info("server init finished.")
        
        
    def compile(self):

        self.my_sqznet = EvaProgram('sqznet', vec_size=8)
        with self.my_sqznet:
            x = Input('x')
            Output('y', 3*x**2 + 5*x - 2)

        self.my_sqznet.set_output_ranges(20)
        self.my_sqznet.set_input_scales(20)

        self.compiler = CKKSCompiler({'security_level':'0', 'warn_vec_size':'false'})
        self.my_sqznet, self.params, self.signature = self.compiler.compile(self.my_sqznet)

        save(self.params, 'my_sqznet.evaparams')
        save(self.signature, 'my_sqznet.evasignature')
        logging.info("server compile finished.")


    def send_params(self):
        
        with open('my_sqznet.evaparams', 'rb') as f:
            params = f.read()
        
        with open('my_sqznet.evasignature', 'rb') as f:
            signature = f.read()

        msg = pickle.dumps({
            "params": params,
            "signature": signature
        })

        self.send(msg, self.client_host, self.client_listen_params_port)
        logging.info("server send params finished.")

        

    def listen_input(self):

        def li():
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            sock.bind(("", self.listen_input_port))
            sock.listen()
            conn, _ = sock.accept()
            msg = SocketUtil.recv_msg(conn)
            msg = pickle.loads(msg)
            input = msg["input"]
            public_ctx = msg["public_ctx"]
            with open('my_sqznet_inputs.sealvals', 'wb') as f:
                f.write(input)
            with open('my_sqznet.sealpublic', 'wb') as f:
                f.write(public_ctx)
            self.server_listen_input = 1
            logging.info("server listen input finished.")

        thread = Thread(target=li)
        thread.daemon = True
        thread.start()


    def inference(self):

        while not self.server_listen_input:
            time.sleep(1)
        self.public_ctx = load('my_sqznet.sealpublic')
        self.encInputs = load('my_sqznet_inputs.sealvals')

        encOutputs = self.public_ctx.execute(self.my_sqznet, self.encInputs)

        save(encOutputs, 'my_sqznet_outputs.sealvals')
        logging.info("server inference finished.")


    def send_output(self):

        with open('my_sqznet_outputs.sealvals', 'rb') as f:
            output = f.read()
        
        msg = pickle.dumps({
            "output": output
        })

        self.send(msg, self.client_host, self.client_listen_output_port)
        logging.info("server send output finished.")
        
        

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

    server = Server(server_host, server_listen_input_port, client_host, client_listen_params_port, client_listen_output_port)

    server.compile()

    server.send_params()
    
    server.listen_input()

    server.inference()

    server.send_output()
