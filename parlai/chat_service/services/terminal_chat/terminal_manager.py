#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.chat_service.core.chat_service_manager import ChatServiceManager


class TerminalManager(ChatServiceManager):
    class TerminalMessageSender(ChatServiceManager.ChatServiceMessageSender):
        def send_read(self, receiver_id):
            raise NotImplementedError

        def typing_on(self, receiver_id, persona_id=None):
            raise NotImplementedError

    def __init__(self, opt):
        super().__init__(opt)

    def parse_additional_args(self, opt):
        """Parse any other service specific args here."""
        raise NotImplementedError

    def _complete_setup(self):
        """
        Complete necessary setup items. Consider this as a unified method for
        setting up. Call every other functions used in setup from here.
        To be called during instantiation
        """
        raise NotImplementedError

    def _load_model(self):
        """Load model if necessary."""
        raise NotImplementedError

    def restructure_message(self):
        """Use this function to restructure the message into the provided format."""
        raise NotImplementedError

    def _handle_bot_read(self, agent_id):
        """Use this function to handle/execute events once the bot has observed the message."""
        raise NotImplementedError

    def _confirm_message_delivery(self, event):
        """A callback for when messages are marked as delivered"""
        raise NotImplementedError

    def setup_server(self):
        """Prepare the Chat Service server for handling messages."""
        raise NotImplementedError

    def setup_socket(self):
        """Set up socket to start communicating to workers."""
        raise NotImplementedError

    def observe_message(self, receiver_id, text, quick_replies=None, persona_id=None):
        """Send a message through the message manager.

        :param receiver_id:
            int identifier for agent to send message to
        :param text:
            string text to send
        :param quick_replies:
            list of quick replies
        :param persona_id:
            identifier of persona
        """
        raise NotImplementedError
