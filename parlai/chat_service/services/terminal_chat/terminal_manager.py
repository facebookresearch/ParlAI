#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.chat_service.core.chat_service_manager import ChatServiceManager
from parlai.core.agents import create_agent


class TerminalManager(ChatServiceManager):
    class TerminalMessageSender(ChatServiceManager.ChatServiceMessageSender):
        def send_read(self, receiver_id):
            pass

        def typing_on(self, receiver_id, persona_id=None):
            pass

    def __init__(self, opt):
        super().__init__(opt)
        self.port = opt.get('port')

    def parse_additional_args(self, opt):
        """Parse any other service specific args here."""
        pass

    def _complete_setup(self):
        """
        Complete necessary setup items. Consider this as a unified method for setting up.

        Call every other functions used in setup from here.
        To be called during instantiation
        """
        self.setup_server()
        self.init_new_state()
        self.setup_socket()
        self.start_new_run()
        self._load_model()

    def _load_model(self):
        """Load model if necessary."""
        if 'model_file' in self.opt or 'model' in self.opt:
            self.runner_opt['shared_bot_params'] = create_agent(self.runner_opt).share()

    def _handle_message_read(self, event):
        pass

    def restructure_message(self):
        """Use this function to restructure the message into the provided format."""
        raise NotImplementedError

    def _handle_bot_read(self, agent_id):
        """Use this function to handle/execute events once the bot has observed the message."""
        pass

    def _confirm_message_delivery(self, event):
        """A callback for when messages are marked as delivered"""
        pass

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
