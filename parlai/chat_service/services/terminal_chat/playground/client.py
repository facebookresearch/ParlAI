#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import socket

# Create a TCP/IP socket


# Connect the socket to the port where the server is listening
server_address = ('localhost', 12344)
print('connecting to {} port {}'.format(*server_address))

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    while True:
        # Send data
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(server_address)

        message = (input("Enter message here: ") + '\n').encode()
        print('sending {!r}'.format(message.decode()))
        sock.sendall(message)
        if message.decode() == "EXIT\n":
            break
        
        data = sock.recv(1024)
        if data is not None:
            data = data.decode()
            print('received {!r}'.format(data))
        sock.close()
finally:
    print('closing socket')
    sock.close()

