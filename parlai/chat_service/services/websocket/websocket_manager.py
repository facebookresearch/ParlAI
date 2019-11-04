#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Websocket Manager Module
Contains implementation of the WebsocketManager which helps run ParlAI via
websockets
"""

import time
import logging
import parlai.chat_service.services.messenger.shared_utils as shared_utils
from parlai.chat_service.services.websocket.sockets import (
    MessageSocketHandler,
)
import tornado
from tornado.options import define, options

PORT = 35496


class WebsocketManager:
    """
    Manages interactions between agents on a websocket as well as direct interactions
    between agents and an overworld.
    """
    def __init__(self, opt):
        """Create a WebsocketManager using the given setup options"""
        self.subs = []
        self.port = opt.get('port', PORT)
        self.opt = opt
        self.app = None
        self.debug = opt.get('debug', True)

    def start_task(self):
        """Begin handling task.
        """
        self.running = True
        logger = logging.getLogger("MainLogger")
        self.app = self._make_app()
        self.app.listen(self.port)
        tornado.ioloop.IOLoop.current().start()

    def shutdown(self):
        """Defined to shutown the tornado application"""
        tornado.ioloop.IOLoop.current().stop()

    def _new_message(self, message, senderId):
        """Callback when a new message is received
        Args:
            message: string. Message from client
            senderId: UUID. UUID of message sender
        """
        logging.info("Manager got new message!")

    def _make_app(self):
        """
        Starts the tornado application
        """
        message_callback = self._new_message

        options['log_to_stderr'] = True
        tornado.options.parse_command_line([])

        return tornado.web.Application([
            (r"/websocket", MessageSocketHandler, {'message_callback': message_callback}),
        ], debug=self.debug)
