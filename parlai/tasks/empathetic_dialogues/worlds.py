#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld


class SelfChatWorld(SelfChatBaseWorld):
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)

    def get_contexts(self):
        return None

    def share(self):
        shared_data = super().share()
        return shared_data