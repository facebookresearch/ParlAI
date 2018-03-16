# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import errno
import json
import logging
import requests
import threading
import time
import websocket

import parlai.mturk.core.shared_utils as shared_utils

MAX_QUICK_REPLIES = 10
MAX_TEXT_CHARS = 640
MAX_QUICK_REPLY_TITLE_CHARS = 20
MAX_POSTBACK_CHARS = 1000
SOCKET_TIMEOUT = 6

# Arbitrary attachments can be created as long as they adhere to the docs
# developers.facebook.com/docs/messenger-platform/send-messages/templates

# Message builders
def create_attachment(payload):
    """Create a simple url-based attachment"""
    # TODO support data-based attachments?
    assert payload['type'] in ['image', 'video', 'file', 'audio'], \
        'unsupported attachment type'
    assert ('url' in payload or 'attachment_id' in payload), \
        'unsupported attachment method: must contain url or attachment_id'
    if 'url' in payload:
        return {'type': payload['type'],
                'payload': {'url': payload['url']}}
    elif 'attachment_id' in payload:
        return {'type': payload['type'],
                'payload': {'attachment_id': payload['attachment_id']}}


def create_reply_option(title, payload=''):
    """Create a quick reply option. Takes in display title and optionally extra
    custom data.
    """
    assert len(title) <= MAX_QUICK_REPLY_TITLE_CHARS, (
        'Quick reply title length {} greater than the max of {}'.format(
            len(title), MAX_QUICK_REPLY_TITLE_CHARS
        )
    )
    assert len(payload) <= MAX_POSTBACK_CHARS, (
        'Payload length {} greater than the max of {}'.format(
            len(payload), MAX_POSTBACK_CHARS
        )
    )
    return {'content_type': 'text', 'title': title, 'payload': payload}


def create_text_message(text, quick_replies=None):
    """Return a list of text messages from the given text. If the message is
    too long it is split into multiple messages.
    quick_replies should be a list of options made with create_reply_option.
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
        assert len(quick_replies) <= MAX_QUICK_REPLIES, (
            'Number of quick replies {} greater than the max of {}'.format(
                len(quick_replies), MAX_QUICK_REPLIES
            )
        )
    for i in range(len(tokens)):
        if (curr_length + len(tokens[i]) > MAX_TEXT_CHARS):
            splits.append(_message(' '.join(tokens[cutoff:i]), None))
            cutoff = i + 1
            curr_length = 0
        curr_length += len(tokens[i]) + 1
    if cutoff < len(tokens):
        splits.append(_message(' '.join(tokens[cutoff:]), quick_replies))
    return splits


def create_attachment_message(attachment_item, quick_replies=None):
    """Create a message list made with only an attachment.
    quick_replies should be a list of options made with create_reply_option.
    """
    payload = {'attachment': attachment_item}
    if quick_replies:
        assert len(quick_replies) <= MAX_QUICK_REPLIES, (
            'Number of quick replies {} greater than the max of {}'.format(
                len(quick_replies), MAX_QUICK_REPLIES
            )
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
        }
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
        }
    }


# Socket handler
class MessageSocket():
    """MessageSocket is a wrapper around websocket to simplify message sends
    and receives into parlai from FB messenger.
    """

    def __init__(self, server_url, port, secret_token, message_callback):
        """
        server_url:           url at which the server is to be run
        port:                 port for the socket to operate on
        message_callback:     function to be called on incoming message objects
                              format: message_callback(self, data)
        """
        self.server_url = server_url
        self.port = port
        self.message_callback = message_callback

        self.ws = None
        self.last_pong = None
        self.alive = False
        self.auth_args = {'access_token': secret_token}

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
        """Registers world with the passthrough server"""
        self._safe_send(json.dumps({
            'type': 'world_alive',
            'content': {'id': 'WORLD_ALIVE', 'sender_id': 'world'},
        }), force=True)

    def send_sender_action(self, receiver_id, action):
        api_address = 'https://graph.facebook.com/v2.6/me/messages'
        message = {
            'recipient': {
                'id': receiver_id
            },
            "sender_action": action,
        }
        requests.post(
            api_address,
            params=self.auth_args,
            json=message,
        )

    def send_read(self, receiver_id):
        self.send_sender_action(receiver_id, "mark_seen")

    def typing_on(self, receiver_id):
        self.send_sender_action(receiver_id, "typing_on")

    def send_fb_payload(self, receiver_id, payload):
        """Sends a payload to messenger, processes it if we can"""
        api_address = 'https://graph.facebook.com/v2.6/me/messages'
        if payload['type'] == 'list':
            data = create_compact_list_message(payload['data'])
        elif payload['type'] in ['image', 'video', 'file', 'audio']:
            data = create_attachment(payload)
        else:
            data = payload['data']
        message = {
            "messaging_type": 'RESPONSE',
            "recipient": {
                "id": receiver_id
            },
            "message": {
                "attachment": data,
            }
        }
        response = requests.post(
            api_address,
            params=self.auth_args,
            json=message,
        )
        result = response.json()
        shared_utils.print_and_log(
            logging.INFO,
            '"Facebook response from message send: {}"'.format(result)
        )
        return result

    def send_fb_message(self, receiver_id, message, is_response,
                        quick_replies=None):
        """Sends a message directly to messenger"""
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
                "recipient": {
                    "id": receiver_id
                },
                "message": m
            }
            response = requests.post(
                api_address,
                params=self.auth_args,
                json=payload
            )
            result = response.json()
            shared_utils.print_and_log(
                logging.INFO,
                '"Facebook response from message send: {}"'.format(result)
            )
            results.append(result)
        return results

    def upload_fb_attachment(self, payload):
        """Uploads an attachment using the Attachment Upload API and returns
        an attachment ID.
        """
        api_address = 'https://graph.facebook.com/v2.6/me/message_attachments'
        assert payload['type'] in ['image', 'video', 'file', 'audio'], \
            'unsupported attachment type'
        if 'url' in payload:
            message = {
                "message": {
                    "attachment": {
                        "type": payload['type'],
                        "payload": {
                            "is_reusable": "true",
                            "url": payload['url']
                        }
                    }
                }
            }
            response = requests.post(
                api_address,
                params=self.auth_args,
                json=message,
            )
        elif 'filename' in payload:
            message = {
                "attachment": {
                    "type": payload['type'],
                    "payload": {
                        "is_reusable": "true",
                    }
                }
            }
            filedata= {"filedata": (payload['filename'], open(payload['filename'], 'rb'), payload['type']+'/'+payload['format'])}
            response = requests.post(
                api_address,
                params=self.auth_args,
                data={"message": json.dumps(message)},
                files=filedata
            )
        result = response.json()
        shared_utils.print_and_log(
            logging.INFO,
            '"Facebook response from attachment upload: {}"'.format(result)
        )
        return result


    def _setup_socket(self):
        """Create socket handlers and registers the socket"""
        def on_socket_open(*args):
            shared_utils.print_and_log(
                logging.DEBUG,
                'Socket open: {}'.format(args)
            )
            self._send_world_alive()

        def on_error(ws, error):
            try:
                if error.errno == errno.ECONNREFUSED:
                    self._ensure_closed()
                    self.use_socket = False
                    raise Exception("Socket refused connection, cancelling")
                else:
                    shared_utils.print_and_log(
                        logging.WARN,
                        'Socket logged error: {}'.format(repr(error)),
                    )
            except BaseException:
                if type(error) is websocket.WebSocketConnectionClosedException:
                    return  # Connection closed is noop
                shared_utils.print_and_log(
                    logging.WARN,
                    'Socket logged error: {} Restarting'.format(repr(error)),
                )
                self._ensure_closed()

        def on_disconnect(*args):
            """Disconnect event is a no-op for us, as the server reconnects
            automatically on a retry"""
            shared_utils.print_and_log(
                logging.INFO,
                'World server disconnected: {}'.format(args)
            )
            self.alive = False
            self._ensure_closed()

        def on_message(*args):
            """Incoming message handler for messages from the FB user"""
            packet_dict = json.loads(args[1])
            if packet_dict['type'] == 'conn_success':
                self.alive = True
                return  # No action for successful connection
            if packet_dict['type'] == 'pong':
                self.last_pong = time.time()
                return  # No further action for pongs
            message_data = packet_dict['content']
            shared_utils.print_and_log(
                logging.DEBUG,
                'Message data received: {}'.format(message_data)
            )
            for message_packet in message_data['entry']:
                for message in message_packet['messaging']:
                    self.message_callback(message)

        def run_socket(*args):
            url_base_name = self.server_url.split('https://')[1]
            while self.keep_running:
                try:
                    sock_addr = "ws://{}/".format(
                        url_base_name)
                    self.ws = websocket.WebSocketApp(
                        sock_addr,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_disconnect,
                    )
                    self.ws.on_open = on_socket_open
                    self.ws.run_forever(ping_interval=1, ping_timeout=0.9)
                except Exception as e:
                    shared_utils.print_and_log(
                        logging.WARN,
                        'Socket error {}, attempting restart'.format(repr(e))
                    )
                time.sleep(0.2)

        # Start listening thread
        self.listen_thread = threading.Thread(
            target=run_socket,
            name='Main-Socket-Thread'
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
