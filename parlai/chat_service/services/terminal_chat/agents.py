#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.chat_service.core.agents import ChatServiceAgent

class TerminalAgents(ChatServiceAgent):
    
    def __init__(self, opt, manager, receiver_id, task_id):
        super().__init__(opt, manager, receiver_id, task_id)
    
    def observe(self, act):
        raise NotImplementedError

    def put_data(self, message):
        raise NotImplementedError