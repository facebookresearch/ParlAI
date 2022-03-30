#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import uuid
import websocket
import time
import threading
from parlai.core.params import ParlaiParser

# the socket callback functions operate asynchronously.
# upon exit of a chat, we do not want the user to view any additional messages from the server.
# alas, it is necessary to send two messages ([DONE], and EXIT) in order to fully exist the world pool
# to prevent receiving a message after sending [DONE], we track the user's state with
# this global variable.
RUNNING = True


def _get_rand_id():
    """
    :return: The string of a random id using uuid4
    """
    return str(uuid.uuid4())


def _prBlueBG(text):
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
    if not RUNNING:
        return
    incoming_message = json.loads(message)
    print("\033[0m\n")
    print("Bot: " + incoming_message['text'])
    quick_replies = incoming_message.get('quick_replies')
    if quick_replies is not None and len(quick_replies) > 0:
        print(f"\nOptions: [{'|'.join(quick_replies)}]")
    print("\033[44m\n")


def on_error(ws, error):
    """
    Prints an error, if occurs.

    :param ws: WebSocketApp
    :param error: An error
    """
    print(error)


def on_close(ws):
    """
    Cleanup before closing connection.

    :param ws: WebSocketApp
    """
    # Reset color formatting if necessary
    print("\033[0m")
    print("Connection closed")


def _run(ws, id):
    """
    Takes user input and sends it to a websocket.

    :param ws: websocket.WebSocketApp
    """
    global RUNNING
    while True:
        x = input("\033[44m Me: ")
        print("\033[0m", end="")
        data = {}
        data['id'] = id
        data['text'] = x
        if x == "[DONE]":
            RUNNING = False
        json_data = json.dumps(data)
        ws.send(json_data)
        time.sleep(1)
        if x == "[DONE]":
            time.sleep(1)
            data['text'] = 'EXIT'
            ws.send(json.dumps(data))
            break
    ws.close()


def on_open(ws):
    """
    Starts a new thread that loops, taking user input and sending it to the websocket.

    :param ws: websocket.WebSocketApp that sends messages to a terminal_manager
    """
    id = _get_rand_id()
    threading.Thread(target=_run, args=(ws, id)).start()


def setup_args():
    """
    Set up args, specifically for the port number.

    :return: A parser that parses the port from commandline arguments.
    """
    parser = ParlaiParser(False, False)
    parser_grp = parser.add_argument_group('Terminal Chat')
    parser_grp.add_argument(
        '--port', default=35496, type=int, help='Port to run the terminal chat server'
    )
    parser_grp.add_argument(
        '--host', default='localhost', type=str, help='Host to connect to.'
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = setup_args()
    port = opt.get('port', 34596)
    host = opt.get('host', 'localhost')
    print("Connecting to port: ", port)
    ws = websocket.WebSocketApp(
        f"ws://{host}:{port}/websocket",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open
    ws.run_forever()
