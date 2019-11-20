#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

        Only attachments of type `image` are currently supported. In the case of
        images, the resultant message will have a `text` field which will be a
        base 64 encoded image and `mime_type` which will be an image mime type.

        Args:
            act: dict. If act contains an `attachment` key, then a dict should be
                provided for the value in `attachment`. Otherwise, act should be
                a dict with the key `text` for the message.
                For the `attachment` dict, this agent expects a `type` key, which
                specifies whether or not the attachment is an image. If the
                attachment is an image, either a `path` key must be specified
                for the path to the image, or a `data` key holding a PIL Image.
                A `quick_replies` key can be provided with a list of string quick
                replies for any message
        """
        logging.info(f"Sending new message: {act}")
        quick_replies = act.get('quick_replies', None)
        attachment_msg = self._get_attachment_msg(act)
        if attachment_msg is not None:
            self.manager.observe_payload(self.id, attachment_msg, quick_replies)
        else:
            self.manager.observe_message(self.id, act['text'], quick_replies)

    def _get_attachment_msg(self, act):
        """
        Gets the message for an attachment given an act

        Args:
            act: dict. See `observe` for the structure of this dict

        If the act does not have an attachment key, returns None. Otherwise,
        returns the message for observe.
        """
        if not act.get('attachment'):
            return None

        attachment = act['attachment']
        assert attachment['type'] == 'image', "Only image attachments supported"

        if 'path' in attachment:
            image = Image.open(attachment['path'])
            msg = self._get_message_from_image(image)
        elif 'data' in attachment:
            msg = self._get_message_from_image(attachment['data'])
        else:
            raise ValueError("Invalid attachment format for type 'image'")

        return msg

    def _get_message_from_image(self, image):
        """Gets the message dict for sending the provided image

        Args:
            image: PIL Image. Image to be sent in the message

        Returns a message struct with the fields `type`, `text` and `mime_type`.
        See `observe` for more information
        """
        buffered = BytesIO()
        image.save(buffered, format=image.format)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        msg = {'type': 'image', 'text': img_str, 'mime_type': Image.MIME[image.format]}
        return msg

    def put_data(self, message):
        """Put data into the message queue

        Args:
            message: dict. An incoming websocket message. See the chat_services
                README for the message structure.
        """
        logging.info(f"Received new message: {message}")
        action = {
            'episode_done': False,
            'text': message.get('text', ''),
            'attachment': message.get('attachment'),
        }

        self._queue_action(action, self.action_id)
        self.action_id += 1
