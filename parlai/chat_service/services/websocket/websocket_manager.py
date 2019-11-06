#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Websocket Manager Module
Contains implementation of the WebsocketManager which helps run ParlAI via
websockets
"""

import asyncio
import time
import datetime
import logging
import threading
import traceback
import sys
from parlai.core.agents import create_agent

# TODO: Use generalized module (Issue #2079)
from parlai.chat_service.services.messenger.messenger_manager import AgentState

# TODO: Use a generalized World Runner module (Issue #2079)
from parlai.chat_service.services.messenger.world_runner import MessengerWorldRunner
import parlai.chat_service.services.messenger.shared_utils as shared_utils
from parlai.chat_service.services.websocket.sockets import MessageSocketHandler
from agents import WebsocketAgent
import tornado
from tornado.options import options


class WebsocketManager:
    """
    Manages interactions between agents on a websocket as well as direct interactions
    between agents and an overworld.
    """

    def __init__(self, opt):
        """Create a WebsocketManager using the given setup options"""
        self.subs = []
        self.port = opt.get('port')
        self.opt = opt
        self.app = None
        self.debug = opt.get('debug', True)
        # TODO: Rename when using a generalized AgentState module (Issue #2079)
        self.agent_pool = {}
        self.messenger_agent_states = {}
        self.agent_id_to_overworld_future = {}
        self.task_to_agent_ids = {}
        self.conversation_index = 0
        self.shutting_down = False
        self.agent_pool_change_condition = threading.Condition()
        self.active_worlds = {}
        self._parse_config(opt)
        self._complete_setup()

    def _parse_config(self, opt):
        """Parse config for task."""
        self.config = opt['config']
        self.overworld = self.config['overworld']
        self.world_path = self.config['world_path']
        self.world_module = shared_utils.get_world_module(self.world_path)
        self.max_workers = self.config['max_workers']
        self.task_configs = self.config['configs']
        self.opt['task'] = self.config['task_name']
        self.world_runner = MessengerWorldRunner(
            opt, self.world_path, self.max_workers, self, opt['is_debug']
        )
        self.max_agents_for = {
            task: cfg.agents_required for task, cfg in self.task_configs.items()
        }
        self.onboard_map = {
            task: cfg.onboarding_name for task, cfg in self.task_configs.items()
        }
        self.taskworld_map = {
            task: cfg.task_name for task, cfg in self.task_configs.items()
        }

    def check_timeout_in_pool(self, world_type, agent_pool, max_time_in_pool):
        """Check for timed-out agents in pool.

        :param world_type:
            string world type
        :param agent_pool:
            list of ``AgentState``s
        :param max_time_in_pool:
            int maximum time allowed for agent to be in pool
        """
        for agent_state in agent_pool:
            time_in_pool = agent_state.time_in_pool.get(world_type)
            if time_in_pool and time.time() - time_in_pool > max_time_in_pool:
                # remove agent from agent_pool
                self.remove_agent_from_pool(agent_state, world_type)
                # put agent back in overworld
                agent_state.set_active_agent(agent_state.get_overworld_agent())

                agent_state.stored_data['removed_after_timeout'] = True

                # reset wait message state
                agent_state.stored_data['seen_wait_message'] = False

            elif time_in_pool and time.time() - time_in_pool > 30:
                # tell agent that a match is taking longer than
                # expected
                if (
                    not agent_state.stored_data.get('seen_wait_message')
                    or not agent_state.stored_data['seen_wait_message']
                ):
                    self.observe_message(
                        agent_state.messenger_id,
                        'Pairing is taking longer than expected. '
                        'If you wish to exit, type *EXIT*.',
                    )
                    agent_state.stored_data['seen_wait_message'] = True

    def remove_agent_from_pool(self, agent, world_type='default', mark_removed=True):
        """Remove agent from the pool.

        :param agent:
            MessengerAgent object
        :param world_type:
            string, world name
        :param mark_removed:
            bool, whether to mark an agent as removed from the pool
        """
        with self.agent_pool_change_condition:
            self._log_debug('Removing agent {} from pool...'.format(agent.messenger_id))
            if world_type in self.agent_pool and agent in self.agent_pool[world_type]:
                self.agent_pool[world_type].remove(agent)
                # reset agent's time_in_pool
                if world_type in agent.time_in_pool:
                    del agent.time_in_pool[world_type]
                # maybe mark agent as removed
                if mark_removed:
                    agent.stored_data['removed_from_pool'] = True

    def add_agent_to_pool(self, agent, world_type='default'):
        """Add the agent to pool.

        :param agent:
            MessengerAgent object
        :param world_type:
            Name of world whose pool should now contain agent
        """
        with self.agent_pool_change_condition:
            self._log_debug('Adding agent {} to pool...'.format(agent.messenger_id))
            # time agent entered agent_pool
            agent.time_in_pool.setdefault(world_type, time.time())
            # add agent to pool
            self.agent_pool.setdefault(world_type, []).append(agent)

    def _expire_all_conversations(self):
        """Iterate through all sub-worlds and shut them down."""
        self.running = False
        for agent_id, overworld_fut in self.agent_id_to_overworld_future.items():
            self.observe_message(
                agent_id,
                'System: The conversation bot going to go offline soon, '
                'finish any active conversations in the next 5 minutes.',
            )
            overworld_fut.cancel()

        # 5 minute grace period for conversations to finish
        time_passed = 0
        while time_passed < 5 * 60:
            any_alive = False
            for task_fut in self.active_worlds.values():
                if task_fut is not None:
                    any_alive = any_alive or task_fut.running()
            if not any_alive:
                break
            time.sleep(1)

        # Tell the worlds that they should be shutting down at this point,
        # agents will refer to this value
        self.shutting_down = True

    def _get_unique_pool(self):
        """Return unique pool.

        Returns a filtered version of the agent pool where each agent is
        only listed a maximum of one time.

        :return:
            a dictionary mapping world_types to agent pools
        """
        valid_pools = {}
        for world_type, agent_pool in self.agent_pool.items():
            eligibility_function = shared_utils.get_eligibility_fn(
                self.world_module, world_type
            )
            if eligibility_function is not None:
                valid_pools[world_type] = [
                    w for w in agent_pool if eligibility_function(w)
                ]
            else:
                valid_pools[world_type] = self.agent_pool[world_type]
        return valid_pools

    def get_agent_state(self, agent_id):
        """Return agent state.

        :param agent_id:
            int agent identifier

        :return:
            AgentState object if agent_id is being tracked, else None
        """
        if agent_id in self.messenger_agent_states:
            return self.messenger_agent_states[agent_id]
        return None

    def _get_agent(self, agent_id, task_id):
        """Return agent object for given agent ID and task ID.

        :param agent_id:
            int agent identifier
        :param task_id:
            string task name

        :return:
            MessengerAgent object associated with given agent ID and task ID if
            possible, else None
        """
        agent_state = self.get_agent_state(agent_id)
        if agent_state is not None:
            if agent_state.has_task(task_id):
                return agent_state.get_agent_for_task(task_id)
        return None

    def _complete_setup(self):
        """Complete necessary setup items."""
        self._load_model()

    def _load_model(self):
        """Load model if necessary"""
        if 'model_file' in self.opt or 'model' in self.opt:
            self.opt['shared_bot_params'] = create_agent(self.opt).share()

    def _manager_loop_fn(self):
        """An iteration of the manager's main loop to launch worlds.
        """

        def _done_callback(fut):
            """Log and raise exception of task world, if there is one.

            Additionally, set active agent to overworld agent.
            """
            e = fut.exception()
            if e is not None:
                shared_utils.print_and_log(
                    logging.ERROR,
                    'World {} had error {}'.format(world_type, repr(e)),
                    should_print=True,
                )
                traceback.print_exc(file=sys.stdout)
                for agent in agents:
                    self.observe_message(
                        agent.id, 'Sorry, this world closed. Returning to overworld.'
                    )
            else:
                shared_utils.print_and_log(
                    logging.INFO,
                    'World {} had no error'.format(world_type),
                    should_print=True,
                )
            self.active_worlds[task_id] = None
            for agent in agents:
                agent_state = self.get_agent_state(agent.id)
                agent_state.set_active_agent(agent_state.get_overworld_agent())

        with self.agent_pool_change_condition:
            valid_pools = self._get_unique_pool()
            for world_type, agent_pool in valid_pools.items():
                # check if agent has exceeded max time in pool
                world_config = self.task_configs[world_type]
                if world_config.max_time_in_pool is not None:
                    self.check_timeout_in_pool(
                        world_type, agent_pool, world_config.max_time_in_pool
                    )

                needed_agents = self.max_agents_for[world_type]
                if len(agent_pool) >= needed_agents:
                    shared_utils.print_and_log(
                        logging.INFO, 'starting pool', should_print=True
                    )
                    # enough agents in pool to start new conversation
                    self.conversation_index += 1
                    task_id = 't_{}'.format(self.conversation_index)

                    # Add the required number of valid agents to the conv
                    agent_states = [w for w in agent_pool[:needed_agents]]
                    agents = []
                    for state in agent_states:
                        agent = self._create_agent(task_id, state.get_id())
                        agent.onboard_data = state.onboard_data
                        state.assign_agent_to_task(agent, task_id)
                        state.set_active_agent(agent)
                        agents.append(agent)
                        # reset wait message state
                        state.stored_data['seen_wait_message'] = False
                    assign_role_function = shared_utils.get_assign_roles_fn(
                        self.world_module, self.taskworld_map[world_type]
                    )
                    if assign_role_function is None:
                        assign_role_function = shared_utils.default_assign_roles_fn
                    assign_role_function(agents)
                    # Allow task creator to filter out workers and run
                    # versions of the task that require fewer agents
                    for a in agents:
                        # Remove selected workers from the agent pool
                        self.remove_agent_from_pool(
                            self.get_agent_state(a.id),
                            world_type=world_type,
                            mark_removed=False,
                        )
                    for a in agents:
                        partner_list = agents.copy()
                        partner_list.remove(a)
                        a.message_partners = partner_list
                    # launch task world.
                    future = self.world_runner.launch_task_world(
                        task_id, self.taskworld_map[world_type], agents
                    )
                    future.add_done_callback(_done_callback)
                    self.active_worlds[task_id] = future

    def start_task(self):
        """Begin handling task.
        """
        self.running = True
        self.app = self._make_app()
        self.app.listen(self.port)
        # Must use a tornado callback to run the main loop
        callback_time = shared_utils.THREAD_MEDIUM_SLEEP * 1000
        tornado.ioloop.PeriodicCallback(
            callback=self._manager_loop_fn, callback_time=callback_time
        ).start()
        tornado.ioloop.IOLoop.current().start()

    def shutdown(self):
        """Defined to shutown the tornado application"""
        tornado.ioloop.IOLoop.current().stop()

    def _on_new_message(self, message, socketID):
        """Callback when a new message is received
        Args:
            message: string. Message from client
            socketID: UUID. UUID of message sender socket
        """
        logging.info("Manager got new message!")
        if socketID not in self.messenger_agent_states:
            self._on_first_message(message, socketID)
            return

        agent_state = self.get_agent_state(socketID)
        if agent_state.get_active_agent() is None:
            # return agent to overworld
            if message.upper() == 'EXIT':
                # remove agent from agent_pool
                to_remove = []
                for world_type, _time in agent_state.time_in_pool.items():
                    to_remove.append(world_type)
                for world_type in to_remove:
                    self.remove_agent_from_pool(
                        agent_state, world_type, mark_removed=False
                    )
                agent_state.stored_data['seen_wait_message'] = False
                agent_state.set_active_agent(agent_state.get_overworld_agent())
            else:
                self.observe_message(
                    socketID,
                    'Please wait while we pair you with another person. '
                    'If you wish to exit, type *EXIT*.',
                )
        else:
            agent = agent_state.get_active_agent()
            agent.put_data(message)

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

        future = self.world_runner.launch_overworld(
            task_id, self.overworld, self.onboard_map, agent
        )

        def _done_callback(fut):
            """Log and raise exception of overworld (if there is one)."""
            e = fut.exception()
            if e is not None:
                self._log_debug('{} returned with error {}'.format(task_id, repr(e)))
                if self.debug:
                    raise e

        future.add_done_callback(_done_callback)
        self.agent_id_to_overworld_future[socketID] = future

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
        message_callback = self._on_new_message

        options['log_to_stderr'] = True
        tornado.options.parse_command_line([])

        return tornado.web.Application(
            [
                (
                    r"/websocket",
                    MessageSocketHandler,
                    {'message_callback': message_callback},
                )
            ],
            debug=self.debug,
        )

    def observe_message(self, socket_id, text):
        """Send a message through the message manager.

        :param socket_id:
            int identifier for agent socket to send message to
        :param text:
            string text to send
        """
        asyncio.set_event_loop(asyncio.new_event_loop())
        return MessageSocketHandler.subs[socket_id].write_message(text)

    def _log_debug(self, text):
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        shared_utils.print_and_log(logging.DEBUG, f'{time}: {text}', should_print=True)
