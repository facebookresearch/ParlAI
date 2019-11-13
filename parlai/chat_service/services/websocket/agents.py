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

        Only payloads of type 'image' are currently supported. In the case of
        images, the resultant message will have a `body` field which will be a
        base 64 encoded image and `mime_type` which will be an image mime type.

        Args:
            act: dict. See the `chat_services` README for more details on the
                format of this argument. For the `payload`, this agent expects
                a 'type' key, where the value 'image' is the only type currently
                supported. If the payload is an image, either a `path` key must be
                specified for the path to the image, or an 'image' key holding a
                PIL Image.
        """
        logging.info(f"Sending new message: {act}")
        if 'payload' in act:
            payload = act['payload']
            if payload['type'] == 'image':
                if 'path' in payload:
                    image = Image.open(payload['path'])
                    msg = self._get_message_from_image(image)
                elif 'image' in payload:
                    msg = self._get_message_from_image(payload['image'])
                else:
                    raise ValueError("Invalid payload for type 'image'")
            else:
                raise ValueError(f"Payload type {payload['type']} not supported")
        else:
            msg = act['text']

        quick_replies = act.get('quick_replies', None)
        self.manager.observe_message(self.id, msg, quick_replies)

    def _get_message_from_image(self, image):
        """Gets the message dict for sending the provided image

        Args:
            image: PIL Image. Image to be sent in the message

        Returns a message struct with the fields `type`, `body` and `mime_type`.
        See `observe` for more information
        """
        buffered = BytesIO()
        image.save(buffered, format=image.format)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        msg = {
            'type': 'image',
            'body': img_str,
            'mime_type': Image.MIME[image.format],
        }
        return msg

    def put_data(self, message):
        """Put data into the message queue"""
        logging.info(f"Received new message: {message}")
        message = json.loads(message['text'])
        action = {'episode_done': False, 'image': message['image'], 'text': message['body']}
        self._queue_action(action, self.action_id)
        self.action_id += 1
