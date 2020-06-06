#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import errno
import json
import logging
import threading
import time
import websocket
import parlai.chat_service.utils.logging as log_utils

SOCKET_TIMEOUT = 6


# Socket handler
class ChatServiceMessageSocket:
    """
    ChatServiceMessageSocket is a wrapper around websocket to forward messages from the
    remote server to the ChatServiceManager.
    """

    def __init__(self, server_url, port, message_callback):
        """
        server_url:           url at which the server is to be run
        port:                 port for the socket to operate on
        message_callback:     function to be called on incoming message objects (format: message_callback(self, data))
        """
        self.server_url = server_url
        self.port = port
        self.message_callback = message_callback

        self.ws = None
        self.last_pong = None
        self.alive = False

        # initialize the state
        self.listen_thread = None

        # setup the socket
        self.keep_running = True
        self._setup_socket()

    def _safe_send(self, data, force=False):
        if not self.alive and not force:
            # Try to wait a second to send a packet
            timeout = 1
            while timeout > 0 and not self.alive:
                time.sleep(0.1)
                timeout -= 0.1
            if not self.alive:
                # don't try to send a packet if we're still dead
                return False
        try:
            self.ws.send(data)
        except websocket.WebSocketConnectionClosedException:
            # The channel died mid-send, wait for it to come back up
            return False
        return True

    def _ensure_closed(self):
        try:
            self.ws.close()
        except websocket.WebSocketConnectionClosedException:
            pass

    def _send_world_alive(self):
        """
        Registers world with the passthrough server.
        """
        self._safe_send(
            json.dumps(
                {
                    'type': 'world_alive',
                    'content': {'id': 'WORLD_ALIVE', 'sender_id': 'world'},
                }
            ),
            force=True,
        )

    def _setup_socket(self):
        """
        Create socket handlers and registers the socket.
        """

        def on_socket_open(*args):
            log_utils.print_and_log(logging.DEBUG, 'Socket open: {}'.format(args))
            self._send_world_alive()

        def on_error(ws, error):
            try:
                if error.errno == errno.ECONNREFUSED:
                    self._ensure_closed()
                    self.use_socket = False
                    raise Exception("Socket refused connection, cancelling")
                else:
                    log_utils.print_and_log(
                        logging.WARN, 'Socket logged error: {}'.format(repr(error))
                    )
            except BaseException:
                if type(error) is websocket.WebSocketConnectionClosedException:
                    return  # Connection closed is noop
                log_utils.print_and_log(
                    logging.WARN,
                    'Socket logged error: {} Restarting'.format(repr(error)),
                )
                self._ensure_closed()

        def on_disconnect(*args):
            """
            Disconnect event is a no-op for us, as the server reconnects automatically
            on a retry.
            """
            log_utils.print_and_log(
                logging.INFO, 'World server disconnected: {}'.format(args)
            )
            self.alive = False
            self._ensure_closed()

        def on_message(*args):
            """
            Incoming message handler for messages from the FB user.
            """
            packet_dict = json.loads(args[1])
            if packet_dict['type'] == 'conn_success':
                self.alive = True
                return  # No action for successful connection
            if packet_dict['type'] == 'pong':
                self.last_pong = time.time()
                return  # No further action for pongs
            message_data = packet_dict['content']
            log_utils.print_and_log(
                logging.DEBUG, 'Message data received: {}'.format(message_data)
            )
            for message_packet in message_data['entry']:
                for message in message_packet['messaging']:
                    self.message_callback(message)

        def run_socket(*args):
            url_base_name = self.server_url.split('https://')[1]
            while self.keep_running:
                try:
                    sock_addr = "wss://{}/".format(url_base_name)
                    self.ws = websocket.WebSocketApp(
                        sock_addr,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_disconnect,
                    )
                    self.ws.on_open = on_socket_open
                    self.ws.run_forever(ping_interval=1, ping_timeout=0.9)
                except Exception as e:
                    log_utils.print_and_log(
                        logging.WARN,
                        'Socket error {}, attempting restart'.format(repr(e)),
                    )
                time.sleep(0.2)

        # Start listening thread
        self.listen_thread = threading.Thread(
            target=run_socket, name='Main-Socket-Thread'
        )
        self.listen_thread.daemon = True
        self.listen_thread.start()
        time.sleep(1.2)
        while not self.alive:
            try:
                self._send_world_alive()
            except Exception:
                pass
            time.sleep(0.8)
