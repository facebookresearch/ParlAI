#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, TypeVar
from tornado.websocket import WebSocketHandler
import uuid
import logging
import json


def get_rand_id():
    return str(uuid.uuid4())


T = TypeVar('T', bound='MessageSocketHandler')


class MessageSocketHandler(WebSocketHandler):
    def __init__(self: T, *args, **kwargs):
        self.subs: Dict[int, T] = kwargs.pop('subs')

        def _default_callback(message, socketID):
            logging.warn(f"No callback defined for new WebSocket messages.")

        self.message_callback = kwargs.pop('message_callback', _default_callback)
        self.sid = '1'
        super().__init__(*args, **kwargs)

    def open(self):
        """
        Opens a websocket and assigns a random UUID that is stored in the class-level
        `subs` variable.
        """
        if self.sid not in self.subs.values():
            self.subs[self.sid] = self
            self.set_nodelay(True)

    def on_close(self):
        """
        Runs when a socket is closed.
        """
        if self.sid in self.subs:
            del self.subs[self.sid]

    def on_message(self, message_text):
        """
        Callback that runs when a new message is received from a client See the
        chat_service README for the resultant message structure.

        Args:
            message_text: A stringified JSON object with a text or attachment key.
                `text` should contain a string message and `attachment` is a dict.
                See `WebsocketAgent.put_data` for more information about the
                attachment dict structure.
        """
        logging.info('websocket message from client: {}'.format(message_text))
        message = json.loads(message_text)
        message_history = message.get('message_history', [])

        message = {
            'text': message.get('text', ''),
            'message_history': message_history,
            'payload': message.get('payload'),
            'sender': {'id': self.sid},
            'recipient': {'id': 0},
        }
        self.message_callback(message)

    def check_origin(self, origin):
        return True
