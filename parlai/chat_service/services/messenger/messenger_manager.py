#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Messenger Manager Module.

Contains implementation of the MessengerManager, which helps run ParlAI via FB
Messenger.
"""
import logging
import os

from parlai.core.agents import create_agent
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.server as server_utils
from parlai.utils.io import PathManager
from parlai.chat_service.services.messenger.agents import MessengerAgent
from parlai.chat_service.core.socket import ChatServiceMessageSocket
from parlai.chat_service.services.messenger.message_sender import MessageSender
from parlai.chat_service.core.chat_service_manager import ChatServiceManager

parent_dir = os.path.dirname(os.path.abspath(__file__))


class MessengerManager(ChatServiceManager):
    """
    Manages interactions between agents on messenger as well as direct interactions
    between agents and the messenger overworld.
    """

    def __init__(self, opt):
        """
        Create an MessengerManager using the given setup options.
        """
        # Manager attributes
        super().__init__(opt)

        self._init_logs()
        # Read in Config
        self._parse_config(opt)
        self._complete_setup()

    def parse_additional_args(self, opt):
        self.service_reference_id = self.config['additional_args']['page_id']
        self.should_load_model = self.config['additional_args'].get('load_model', True)
        if self.service_reference_id == 1:
            raise RuntimeError(
                'Please configure your own page in order to run this task. '
                'See the docs (https://parl.ai/docs/tutorial_messenger.html) '
                'for more information.'
            )

    def restructure_message(self, message):
        message.update(message.get('message', {}))
        return message

    def _complete_setup(self):
        """
        Complete necessary setup items.
        """
        self.setup_server()
        self.init_new_state()
        self.setup_socket()
        self.start_new_run()
        self._load_model()

    def _load_model(self):
        """
        Load model if necessary.
        """
        if 'models' in self.opt and self.should_load_model:
            model_params = {}
            model_info = {}
            for model in self.opt['models']:
                model_opt = self.opt['models'][model]
                overrides = model_opt.get('overrides', {})
                if type(overrides) is list:
                    model_opt['overrides'] = overrides[0]
                model_params[model] = create_agent(model_opt).share()
                model_info[model] = {'overrides': overrides}
            self.runner_opt['model_info'] = model_info
            self.runner_opt['shared_bot_params'] = model_params

    def _init_logs(self):
        """
        Initialize logging settings from the opt.
        """
        log_utils.set_is_debug(self.opt['is_debug'])
        log_utils.set_log_level(self.opt['log_level'])

    def mark_removed(self, agent_id, pageid):
        """
        Mark the agent as removed from the pool.

        Can be overriden to change other metadata linked to agent removal.

        :param agent_id:
            int agent psid
        :param pageid:
            int page id
        """
        pass

    def _handle_bot_read(self, agent_id):
        self.sender.send_read(agent_id)
        self.sender.typing_on(agent_id)

    def _confirm_message_delivery(self, event):
        # By default we don't actually do anything when messages are marked as
        # being delivered, but we expose the ability for others to
        self._log_debug(
            'Messages {} marked as received.'.format(event['delivery']['mids'])
        )

    def _handle_message_read(self, event):
        # If the message was sent by another user (as in during a conversation)
        # then we need to propogate the read back to that user.
        self._log_debug('Messages {} marked as read.'.format(event['read']))
        super()._handle_message_read(event)

    def _handle_webhook_event(self, event):
        if 'message' in event:
            if ('image_url' in event and event['image_url'] is not None) or (
                'attachment_url' in event and event['attachment_url'] is not None
            ):
                event['message']['image'] = True
            self._on_new_message(event)
        elif 'delivery' in event:
            self.confirm_message_delivery(event)
        elif 'read' in event:
            self.handle_message_read(event)

    def after_agent_removed(self, agent_id):
        """
        Perform any changes to metadata on agent removal.

        override if extra bookkeeping must be done when removing agent
        """
        pass

    def _create_agent(self, task_id, agent_id):
        """
        Initialize an agent and return it.

        Called each time an agent is placed into a new task.

        :param task_id:
            string task identifier
        :param agent_id:
            int agent id
        """
        return MessengerAgent(
            self.opt, self, task_id, agent_id, self.service_reference_id
        )

    def _log_missing_agent(self, agent_id, assignment_id):
        """
        Log the occurence of a missing agent.
        """
        log_utils.print_and_log(
            logging.WARN,
            'Expected to have an agent for {}_{}, yet none was found'.format(
                agent_id, assignment_id
            ),
        )

    # Manager Lifecycle Functions #
    def setup_server(self):
        """
        Prepare the Messenger server for handling messages.
        """
        if self.bypass_server_setup:
            return

        log_utils.print_and_log(
            logging.INFO,
            '\nYou are going to allow people on Facebook to be agents in '
            'ParlAI.\nDuring this process, Internet connection is required, '
            'and you should turn off your computer\'s auto-sleep '
            'feature.\n',
            should_print=True,
        )
        input('Please press Enter to continue... ')
        log_utils.print_and_log(logging.NOTSET, '', True)

        if self.opt['local'] is True:
            log_utils.print_and_log(
                logging.INFO,
                'In order to run the server locally, you will need '
                'to have a public HTTPS endpoint (SSL signed) running on '
                'the server you are currently excecuting ParlAI on. Enter '
                'that public URL hostname when prompted and ensure that the '
                'port being used by ParlAI (usually 3000) has external '
                'traffic routed to it.',
                should_print=True,
            )
            input('Please press Enter to continue... ')

        log_utils.print_and_log(
            logging.INFO, 'Setting up Messenger webhook...', should_print=True
        )

        # Setup the server with a task name related to the current task
        task_name = '{}-{}'.format('ParlAI-Messenger', self.opt['task'])
        self.server_task_name = ''.join(
            e for e in task_name.lower() if e.isalnum() or e == '-'
        )
        self.server_url = server_utils.setup_server(
            self.server_task_name, local=self.opt['local']
        )
        log_utils.print_and_log(
            logging.INFO,
            'Webhook address: {}/webhook'.format(self.server_url),
            should_print=True,
        )

    # override if permission needed externally
    def get_app_token(self):
        """
        Find and return an app access token.
        """
        if not self.opt.get('force_page_token'):
            if not os.path.exists(os.path.expanduser('~/.parlai/')):
                PathManager.mkdirs(os.path.expanduser('~/.parlai/'))
            access_token_file_path = '~/.parlai/messenger_token'
            expanded_file_path = os.path.expanduser(access_token_file_path)
            if os.path.exists(expanded_file_path):
                with open(expanded_file_path, 'r') as access_token_file:
                    return access_token_file.read()

        token = input(
            'Enter your page\'s access token from the developer page at'
            'https://developers.facebook.com/apps/<YOUR APP ID>'
            '/messenger/settings/ to continue setup:'
        )
        access_token_file_path = '~/.parlai/messenger_token'
        expanded_file_path = os.path.expanduser(access_token_file_path)
        with open(expanded_file_path, 'w+') as access_token_file:
            access_token_file.write(token)
        return token

    def setup_socket(self):
        """
        Set up socket to start communicating to workers.
        """
        if self.bypass_server_setup:
            return

        log_utils.print_and_log(
            logging.INFO, 'Local: Setting up WebSocket...', should_print=True
        )

        self.app_token = self.get_app_token()
        self.sender = MessageSender(self.app_token)

        # Set up receive
        socket_use_url = self.server_url
        if self.opt['local']:  # skip some hops for local stuff
            socket_use_url = 'https://localhost'
        self.socket = ChatServiceMessageSocket(
            socket_use_url, self.port, self._handle_webhook_event
        )
        log_utils.print_and_log(logging.INFO, 'done with websocket', should_print=True)

    # Agent Interaction Functions #

    def observe_message(self, receiver_id, text, quick_replies=None, persona_id=None):
        """
        Send a message through the message manager.

        :param receiver_id:
            int identifier for agent to send message to
        :param text:
            string text to send
        :param quick_replies:
            list of quick replies
        :param persona_id:
            identifier of persona
        """
        return self.sender.send_fb_message(
            receiver_id, text, True, quick_replies=quick_replies, persona_id=persona_id
        )

    def observe_payload(self, receiver_id, data, quick_replies=None, persona_id=None):
        """
        Send a payload through the message manager.

        :param receiver_id:
            int identifier for agent to send message to
        :param data:
            object data to send
        :param quick_replies:
            list of quick replies
        :param persona_id:
            identifier of persona
        """
        return self.sender.send_fb_payload(
            receiver_id, data, quick_replies=quick_replies, persona_id=persona_id
        )

    def upload_attachment(self, payload):
        """
        Upload an attachment and return an attachment ID.

        :param payload:
            dict with the following format:
                {'type': <TYPE>, 'url': <URL>} or
                {'type': <TYPE>, 'filename': <FILENAME>, 'format': <FILEFORMAT>}.
                For example,
                {'type': 'image', 'filename': 'test.png', 'format': 'png'}
        """
        return self.sender.upload_fb_attachment(payload)
