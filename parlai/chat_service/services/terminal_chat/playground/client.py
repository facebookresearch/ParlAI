#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import socket

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('localhost', 12345)
print('connecting to {} port {}'.format(*server_address))
sock.connect(server_address)

try:
    while True:
        # Send data
        message = input("Enter message here: ").encode()
        if message.decode() == "EXIT":
            break
        print('sending {!r}'.format(message.decode()))
        sock.sendall(message)

        # Look for the response
        amount_received = 0
        amount_expected = len(message)

        while amount_received < amount_expected:
            data = sock.recv(1024)
            amount_received += len(data)
            print('received {!r}'.format(data))

finally:
    print('closing socket')
    sock.close()
