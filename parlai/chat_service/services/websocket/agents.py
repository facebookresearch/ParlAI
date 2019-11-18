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

        Only payloads of type 'image' are currently supported. In the case of
        images, the resultant message will have a `text` field which will be a
        base 64 encoded image and `mime_type` which will be an image mime type.

        Returned payloads have a 'image' boolean field, a 'text' field for the
        message contents, and a 'mime_type' field for the message content type.

        Args:
            act: dict. If act contains a payload, then a dict should be provided.
                Otherwise, act should be a dict with the `text` key for the content.
                For the 'payload' dict, this agent expects an 'image' key, which
                specifies whether or not the payload is an image.
                If the payload is an image, either a 'path' key must be specified
                for the path to the image, or a 'data' key holding a PIL Image.
                A `quick_replies` key can be provided with a list of string quick
                replies for both payload and text messages.
        """
        logging.info(f"Sending new message: {act}")
        quick_replies = act.get('quick_replies', None)
        if 'payload' in act:
            payload = act['payload']
            if payload['image']:
                if 'path' in payload:
                    image = Image.open(payload['path'])
                    msg = self._get_message_from_image(image)
                elif 'data' in payload:
                    msg = self._get_message_from_image(payload['data'])
                else:
                    raise ValueError("Invalid payload for type 'image'")
            else:
                raise ValueError(f"Payload type {payload['type']} not supported")

            self.manager.observe_payload(self.id, msg, quick_replies)
        else:
            self.manager.observe_message(self.id, act['text'], quick_replies)

    def _get_message_from_image(self, image):
        """Gets the message dict for sending the provided image

        Args:
            image: PIL Image. Image to be sent in the message

        Returns a message struct with the fields `image`, `text` and `mime_type`.
        See `observe` for more information
        """
        buffered = BytesIO()
        image.save(buffered, format=image.format)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        msg = {'image': True, 'text': img_str, 'mime_type': Image.MIME[image.format]}
        return msg

    def put_data(self, message):
        """Put data into the message queue

        Args:
            message: dict. An incoming websocket message, where the message content
                is in the 'text' field. The message content is expected to be
                stringified JSON. See `observe` for usable keys of the JSON
                content. `See MessageSocketHandler.on_message` for more
                information about the message structure.
        """
        logging.info(f"Received new message: {message}")
        message = json.loads(message['text'])
        action = {
            'episode_done': False,
            'image': message.get('image', False),
            'text': message.get('text', ''),
            'mime_type': message.get('mime_type')
        }
        self._queue_action(action, self.action_id)
        self.action_id += 1
