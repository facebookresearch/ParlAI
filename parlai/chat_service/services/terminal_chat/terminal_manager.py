#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.chat_service.services.websocket.websocket_manager import WebsocketManager


class TerminalManager(WebsocketManager):
    """
    Chat Service manager that runs chat service tasks using terminal to send and receive
    messages.
    """

    pass
