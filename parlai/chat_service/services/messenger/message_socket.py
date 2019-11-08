#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.chat_service.core.manager_utils import ChatServiceMessageSocket

# Socket handler
class MessageSocket(ChatServiceMessageSocket):
    """MessageSocket is a wrapper around websocket to forward messages from the
    remote server to the MessengerManager.
    """
    pass