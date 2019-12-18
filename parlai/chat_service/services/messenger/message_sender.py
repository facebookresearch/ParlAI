#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import requests

import parlai.mturk.core.shared_utils as shared_utils

MAX_QUICK_REPLIES = 10
MAX_TEXT_CHARS = 640
MAX_QUICK_REPLY_TITLE_CHARS = 20
MAX_POSTBACK_CHARS = 1000


# Arbitrary attachments can be created as long as they adhere to the docs
# developers.facebook.com/docs/messenger-platform/send-messages/templates

# Message builders
def create_attachment(payload):
    """
    Create a simple url-based attachment.
    """
    # TODO support data-based attachments?
    assert payload['type'] in [
        'image',
        'video',
        'file',
        'audio',
    ], 'unsupported attachment type'
    assert (
        'url' in payload or 'attachment_id' in payload
    ), 'unsupported attachment method: must contain url or attachment_id'
    if 'url' in payload:
        return {'type': payload['type'], 'payload': {'url': payload['url']}}
    elif 'attachment_id' in payload:
        return {
            'type': payload['type'],
            'payload': {'attachment_id': payload['attachment_id']},
        }


def create_reply_option(title, payload=''):
    """
    Create a quick reply option.

    Takes in display title and optionally extra custom data.
    """
    assert (
        len(title) <= MAX_QUICK_REPLY_TITLE_CHARS
    ), 'Quick reply title length {} greater than the max of {}'.format(
        len(title), MAX_QUICK_REPLY_TITLE_CHARS
    )
    assert (
        len(payload) <= MAX_POSTBACK_CHARS
    ), 'Payload length {} greater than the max of {}'.format(
        len(payload), MAX_POSTBACK_CHARS
    )
    return {'content_type': 'text', 'title': title, 'payload': payload}


def create_text_message(text, quick_replies=None):
    """
    Return a list of text messages from the given text.

    If the message is too long it is split into multiple messages. quick_replies should
    be a list of options made with create_reply_option.
    """

    def _message(text_content, replies):
        payload = {'text': text_content[:MAX_TEXT_CHARS]}
        if replies:
            payload['quick_replies'] = replies
        return payload

    tokens = [s[:MAX_TEXT_CHARS] for s in text.split(' ')]
    splits = []
    cutoff = 0
    curr_length = 0
    if quick_replies:
        assert (
            len(quick_replies) <= MAX_QUICK_REPLIES
        ), 'Number of quick replies {} greater than the max of {}'.format(
            len(quick_replies), MAX_QUICK_REPLIES
        )
    for i in range(len(tokens)):
        if tokens[i] == '[*SPLIT*]':
            if ' '.join(tokens[cutoff : i - 1]).strip() != '':
                splits.append(_message(' '.join(tokens[cutoff:i]), None))
                cutoff = i + 1
                curr_length = 0
        if curr_length + len(tokens[i]) > MAX_TEXT_CHARS:
            splits.append(_message(' '.join(tokens[cutoff:i]), None))
            cutoff = i
            curr_length = 0
        curr_length += len(tokens[i]) + 1
    if cutoff < len(tokens):
        splits.append(_message(' '.join(tokens[cutoff:]), quick_replies))
    return splits


def create_attachment_message(attachment_item, quick_replies=None):
    """
    Create a message list made with only an attachment.

    quick_replies should be a list of options made with create_reply_option.
    """
    payload = {'attachment': attachment_item}
    if quick_replies:
        assert (
            len(quick_replies) <= MAX_QUICK_REPLIES
        ), 'Number of quick replies {} greater than the max of {}'.format(
            len(quick_replies), MAX_QUICK_REPLIES
        )
        payload['quick_replies'] = quick_replies
    return [payload]


def create_list_element(element):
    assert 'title' in element, 'List elems must have a title'
    ret_elem = {
        'title': element['title'],
        'subtitle': '',
        'default_action': {
            'type': 'postback',
            'title': element['title'],
            'payload': element['title'],
        },
    }
    if 'subtitle' in element:
        ret_elem['subtitle'] = element['subtitle']
    return ret_elem


def create_compact_list_message(raw_elems):
    elements = [create_list_element(elem) for elem in raw_elems]
    return {
        'type': 'template',
        'payload': {
            'template_type': 'list',
            'top_element_style': 'COMPACT',
            'elements': elements,
        },
    }


class MessageSender:
    """
    MesageSender is a wrapper around the facebook messenger requests that simplifies the
    process of sending content.
    """

    def __init__(self, secret_token):
        """
        server_url:           url at which the server is to be run
        port:                 port for the socket to operate on
        message_callback:     function to be called on incoming message objects
                              format: message_callback(self, data)
        """
        self.auth_args = {'access_token': secret_token}

    def send_sender_action(self, receiver_id, action, persona_id=None):
        api_address = 'https://graph.facebook.com/v2.6/me/messages'
        message = {'recipient': {'id': receiver_id}, "sender_action": action}
        if persona_id is not None:
            message['persona_id'] = persona_id
        requests.post(api_address, params=self.auth_args, json=message)

    def send_read(self, receiver_id):
        self.send_sender_action(receiver_id, "mark_seen")

    def typing_on(self, receiver_id, persona_id=None):
        self.send_sender_action(receiver_id, "typing_on", persona_id)

    def send_fb_payload(
        self, receiver_id, payload, quick_replies=None, persona_id=None
    ):
        """
        Sends a payload to messenger, processes it if we can.
        """
        api_address = 'https://graph.facebook.com/v2.6/me/messages'
        if payload['type'] == 'list':
            data = create_compact_list_message(payload['data'])
        elif payload['type'] in ['image', 'video', 'file', 'audio']:
            data = create_attachment(payload)
        else:
            data = payload['data']

        message = {
            "messaging_type": 'RESPONSE',
            "recipient": {"id": receiver_id},
            "message": {"attachment": data},
        }
        if quick_replies is not None:
            quick_replies = [create_reply_option(x, x) for x in quick_replies]
            message['message']['quick_replies'] = quick_replies
        if persona_id is not None:
            payload['persona_id'] = persona_id
        response = requests.post(api_address, params=self.auth_args, json=message)
        result = response.json()
        if 'error' in result:
            if result['error']['code'] == 1200:
                # temporary error please retry
                response = requests.post(
                    api_address, params=self.auth_args, json=message
                )
                result = response.json()
        shared_utils.print_and_log(
            logging.INFO, '"Facebook response from message send: {}"'.format(result)
        )
        return result

    def send_fb_message(
        self, receiver_id, message, is_response, quick_replies=None, persona_id=None
    ):
        """
        Sends a message directly to messenger.
        """
        api_address = 'https://graph.facebook.com/v2.6/me/messages'
        if quick_replies is not None:
            quick_replies = [create_reply_option(x, x) for x in quick_replies]
        ms = create_text_message(message, quick_replies)
        results = []
        for m in ms:
            if m['text'] == '':
                continue  # Skip blank messages
            payload = {
                "messaging_type": 'RESPONSE' if is_response else 'UPDATE',
                "recipient": {"id": receiver_id},
                "message": m,
            }
            if persona_id is not None:
                payload['persona_id'] = persona_id
            response = requests.post(api_address, params=self.auth_args, json=payload)
            result = response.json()
            if 'error' in result:
                if result['error']['code'] == 1200:
                    # temporary error please retry
                    response = requests.post(
                        api_address, params=self.auth_args, json=payload
                    )
                    result = response.json()
            shared_utils.print_and_log(
                logging.INFO, '"Facebook response from message send: {}"'.format(result)
            )
            results.append(result)
        return results

    def create_persona(self, name, image_url):
        """
        Creates a new persona and returns persona_id.
        """
        api_address = 'https://graph.facebook.com/me/personas'
        message = {'name': name, "profile_picture_url": image_url}
        response = requests.post(api_address, params=self.auth_args, json=message)
        result = response.json()
        shared_utils.print_and_log(
            logging.INFO, '"Facebook response from create persona: {}"'.format(result)
        )
        return result

    def delete_persona(self, persona_id):
        """
        Deletes the persona.
        """
        api_address = 'https://graph.facebook.com/' + persona_id
        response = requests.delete(api_address, params=self.auth_args)
        result = response.json()
        shared_utils.print_and_log(
            logging.INFO, '"Facebook response from delete persona: {}"'.format(result)
        )
        return result

    def upload_fb_attachment(self, payload):
        """
        Uploads an attachment using the Attachment Upload API and returns an attachment
        ID.
        """
        api_address = 'https://graph.facebook.com/v2.6/me/message_attachments'
        assert payload['type'] in [
            'image',
            'video',
            'file',
            'audio',
        ], 'unsupported attachment type'
        if 'url' in payload:
            message = {
                "message": {
                    "attachment": {
                        "type": payload['type'],
                        "payload": {"is_reusable": "true", "url": payload['url']},
                    }
                }
            }
            response = requests.post(api_address, params=self.auth_args, json=message)
        elif 'filename' in payload:
            message = {
                "attachment": {
                    "type": payload['type'],
                    "payload": {"is_reusable": "true"},
                }
            }
            with open(payload['filename'], 'rb') as f:
                filedata = {
                    "filedata": (
                        payload['filename'],
                        f,
                        payload['type'] + '/' + payload['format'],
                    )
                }
                response = requests.post(
                    api_address,
                    params=self.auth_args,
                    data={"message": json.dumps(message)},
                    files=filedata,
                )
        result = response.json()
        shared_utils.print_and_log(
            logging.INFO,
            '"Facebook response from attachment upload: {}"'.format(result),
        )
        return result
