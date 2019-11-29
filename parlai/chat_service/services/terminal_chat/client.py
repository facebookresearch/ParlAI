#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import uuid
import websocket

try:
    import thread
except ImportError:
    import _thread as thread
import time

# STEP 1: RUN: python parlai/chat_service/services/terminal_chat/run.py --config-path parlai/chat_service/tasks/chatbot/config.yml
# STEP 2: RUN: python client.py
# STEP 3: Interact


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
    print("error", error)


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
            time.sleep(1.5)
            if x == "[DONE]":
                break
        try:
            ws.close()
        except e:
            print("eee", e)

    thread.start_new_thread(run, ())


if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        "ws://localhost:35496/websocket",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open
    ws.run_forever()
