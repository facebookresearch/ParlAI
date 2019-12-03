#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import uuid
import websocket
from parlai.core.params import ParlaiParser

try:
    import thread
except ImportError:
    import _thread as thread
import time


def get_rand_id():
    """
    :return: The string of a random id using uuid4
    """
    return str(uuid.uuid4())


def prBlueBG(text):
    """
    Print given in text with a blue background.

    :param text: The text to be printed
    """
    print("\033[44m{}\033[0m".format(text), sep="")


def on_message(ws, message):
    """
    Prints the incoming message from the server.

    :param ws: a WebSocketApp
    :param message: json with 'text' field to be printed
    """
    incoming_message = json.loads(message)
    print("\033[0m\nBot: " + incoming_message['text'], "\033[44m\n")


def on_error(ws, error):
    """
    Prints an error, if occurs.

    :param ws: WebSocketApp
    :param error: An error
    """
    print(error)


def on_close(ws):
    """
    Cleanup before closing connection
    :param ws: WebSocketApp
    """
    # Reset color formatting if necessary
    print("\033[0m")
    print("Connection closed")


def on_open(ws):
    """
    Starts a new thread that loops, taking user input and sending it to the websocket.

    :param ws: websocket.WebSocketApp that sends messages to a terminal_manager
    """
    id = get_rand_id()

    def run(*args):
        while True:
            x = input("\033[44m Me: ")
            print("\033[0m", end="")
            data = {}
            data['id'] = id
            data['text'] = x
            json_data = json.dumps(data)
            ws.send(json_data)
            time.sleep(0.75)
            if x == "[DONE]":
                break
        ws.close()

    thread.start_new_thread(run, ())


def setup_args():
    """
    Set up args, specifically for the port number.

    :return: A parser that parses the port from commandline arguments.
    """
    parser = ParlaiParser(False, False)
    parser.add_terminal_chat_args(is_client=True)
    return parser.parse_args()


if __name__ == "__main__":
    opt = setup_args()
    port = opt.get('port', 34596)
    ws = websocket.WebSocketApp(
        "ws://localhost:{}/websocket".format(port),
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open
    ws.run_forever()
