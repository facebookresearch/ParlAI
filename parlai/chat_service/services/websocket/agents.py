#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import base64
from io import BytesIO
from parlai.chat_service.core.agents import ChatServiceAgent
from PIL import Image


class WebsocketAgent(ChatServiceAgent):
    """Class for a person that can act in a ParlAI world via websockets"""

    def __init__(self, opt, manager, receiver_id, task_id):
        super().__init__(opt, manager, receiver_id, task_id)
        self.message_partners = []
        self.action_id = 1

    def observe(self, act):
        """Send an agent a message through the websocket manager

        The message sent to the WebsocketManager is a string or dict for a payload.
        Payloads have the following fields:
            `type`: similar to the `act` payload below
            `body`: where Base 64 encoded content should be provided
            `mime_type`: where the mime type of the payload should be specified

        Since currently images are the only supported payloads, `body` will be a
        base 64 encoded image and `mime_type` will be an image mime type.

        Args:
            act: dict. See the `chat_services` README for more details on the
                format of this argument. For the `payload`, this agent expects
                a 'type' key, where the value 'image' is the only type currently
                supported. If the payload is an image, a `path` key must be
                specified for the path to the image.
        """
        logging.info(f"Sending new message: {act}")
        if 'payload' in act:
            payload = act['payload']
            if payload['type'] == 'image' and 'path' in payload:
                buffered = BytesIO()
                image = Image.open(payload['path'])
                image.save(buffered, format=image.format)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                msg = {
                    'type': 'image',
                    'body': img_str,
                    'mime_type': Image.MIME[image.format],
                }
            else:
                raise ValueError("Payload not supported")
        else:
            msg = act['text']

        quick_replies = act.get('quick_replies', None)
        self.manager.observe_message(self.id, msg, quick_replies)

    def put_data(self, message):
        """Put data into the message queue"""
        logging.info(f"Received new message: {message}")
        message = json.loads(message['text'])
        action = {'episode_done': False, 'text': message['body']}
        self._queue_action(action, self.action_id)
        self.action_id += 1
