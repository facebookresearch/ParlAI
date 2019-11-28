#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import websocket
import json

# STEP 1: RUN: ./parlai/chat_service/services/websocket/run.py --config-path parlai/chat_service/tasks/chatbot/config.yml
# STEP 2: RUN: python connections.py
# STEP 3: Interact


# import atexit

# def exit_handler():
#     print("\033[0m")
#
# atexit.register(exit_handler)
#


def prBlueBG(text):
    """
    Print given in text with a blue background.
    :param text: The text to be printed
    """
    print("\033[44m{}\033[0m".format(text), sep="")


try:
    import thread
except ImportError:
    import _thread as thread
import time


def on_message(ws, message):
    m = json.loads(message)
    print("Bot: " + m['text'])


def on_error(ws, error):
    print(error)


def on_close(ws):
    print("\033[0m")
    print("### closed ###")


def on_open(ws):
    def run(*args):
        while True:
            x = input("\033[44m Me: ")
            print("\033[0m", end="")
            data = {}
            data['text'] = x
            data['payload'] = 'none'
            json_data = json.dumps(data)
            ws.send(json_data)
            time.sleep(1)
        ws.close()
        print("thread terminating...")

    thread.start_new_thread(run, ())


if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        "ws://localhost:35496/websocket",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open
    ws.run_forever()
