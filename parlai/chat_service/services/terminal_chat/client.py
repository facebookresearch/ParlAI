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
    return str(uuid.uuid4())


def prBlueBG(text):
    """
    Print given in text with a blue background.

    :param text: The text to be printed
    """
    print("\033[44m{}\033[0m".format(text), sep="")


def on_message(ws, message):
    incoming_message = json.loads(message)
    print("\033[0m\nBot: " + incoming_message['text'], "\033[44m\n")


def on_error(ws, error):
    print(error)


def on_close(ws):
    # Reset color formatting if necessary
    print("\033[0m")
    print("Connection closed")


def on_open(ws):
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
    Set up args.
    """
    parser = ParlaiParser(False, False)
    parser.add_terminal_args(is_client=True)
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
