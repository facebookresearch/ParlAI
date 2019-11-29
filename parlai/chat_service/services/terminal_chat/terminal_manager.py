#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.chat_service.services.websocket.websocket_manager import WebsocketManager
from parlai.chat_service.core.chat_service_manager import ChatServiceManager


class TerminalManager(WebsocketManager):
    class TerminalMessageSender(ChatServiceManager.ChatServiceMessageSender):
        def send_read(self, receiver_id):
            pass

        def typing_on(self, receiver_id, persona_id=None):
            pass

    def __init__(self, opt):
        super().__init__(opt)
