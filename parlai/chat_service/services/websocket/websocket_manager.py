#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Websocket Manager Module
Contains implementation of the WebsocketManager which helps run ParlAI via
websockets
"""

import time
import logging
# TODO: Use a generalized AgentState module (Issue #2079)
from parlai.chat_service.services.messenger.messenger_manager import AgentState
import parlai.chat_service.services.messenger.shared_utils as shared_utils
from parlai.chat_service.services.websocket.sockets import (
    MessageSocketHandler,
)
from agents import WebsocketAgent
import tornado
from tornado.options import define, options

PORT = 35496


class WebsocketManager:
    """
    Manages interactions between agents on a websocket as well as direct interactions
    between agents and an overworld.
    """
    def __init__(self, opt):
        """Create a WebsocketManager using the given setup options"""
        self.subs = []
        self.port = opt.get('port', PORT)
        self.opt = opt
        self.app = None
        self.debug = opt.get('debug', True)
        # TODO: Rename when using a generalized AgentState module (Issue #2079)
        self.messenger_agent_states = {}
        self._parse_config(opt)

    def _parse_config(self, opt):
        """Parse config for task."""
        self.config = opt['config']

    def start_task(self):
        """Begin handling task.
        """
        self.running = True
        logger = logging.getLogger("MainLogger")
        self.app = self._make_app()
        self.app.listen(self.port)
        tornado.ioloop.IOLoop.current().start()

    def shutdown(self):
        """Defined to shutown the tornado application"""
        tornado.ioloop.IOLoop.current().stop()

    def _new_message(self, message, socketID):
        """Callback when a new message is received
        Args:
            message: string. Message from client
            socketID: UUID. UUID of message sender socket
        """
        logging.info("Manager got new message!")
        if socketID not in self.messenger_agent_states:
            self._on_first_message(message)
            return

    def _on_first_message(self, message, socketID):
        """Handle first message from player.

        Run when a socketID is given that is not paired with any assignment yet.
        Launch an overworld, complete onboarding, etc.

        :param message:
            string message sent from agent
        :param socketID:
            int socket ID of the message sender
        """
        task_id = 'overworld-{}-{}'.format(socketID, time.time())
        agent = self._create_agent(task_id, socketID)
        agent_state = AgentState(socketID, agent)
        self.messenger_agent_states[socketID] = agent_state

    def _create_agent(self, task_id, socketID):
        """Initialize an agent and return it.

        Called each time an agent is placed into a new task.

        :param task_id:
            string task identifier
        :param agent_id:
            int agent id
        """
        return WebsocketAgent(self.opt, self, task_id, socketID)

    def _make_app(self):
        """
        Starts the tornado application
        """
        message_callback = self._new_message

        options['log_to_stderr'] = True
        tornado.options.parse_command_line([])

        return tornado.web.Application([
            (r"/websocket", MessageSocketHandler, {'message_callback': message_callback}),
        ], debug=self.debug)
