# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import logging
import os
import threading
import time

from parlai.messenger.core.agents import MessengerAgent
from parlai.messenger.core.message_socket import MessageSocket
import parlai.messenger.core.server_utils as server_utils
import parlai.messenger.core.shared_utils as shared_utils

parent_dir = os.path.dirname(os.path.abspath(__file__))


class AgentState():
    """Keeps track of a messenger connection through multiple potential
    worlds"""

    def __init__(self, messenger_id, overworld_agent):
        self.messenger_id = messenger_id
        self.overworld_agent = overworld_agent
        self.active_agent = overworld_agent
        self.task_id_to_agent = {}
        self.onboard_data = None
        self.stored_data = {}
        self.time_in_pool = {}

    def get_active_agent(self):
        return self.active_agent

    def set_active_agent(self, active_agent):
        self.active_agent = active_agent

    def get_overworld_agent(self):
        return self.overworld_agent

    def get_id(self):
        return self.messenger_id

    def has_task(self, task_id):
        return task_id in self.task_id_to_agent

    def get_agent_for_task(self, task_id):
        if self.has_task(task_id):
            return self.task_id_to_agent[task_id]
        else:
            return None

    def assign_agent_to_task(self, agent, task_id):
        self.task_id_to_agent[task_id] = agent


class MessengerManager():
    """Manages interactions between agents on messenger as well as direct
    interactions between agents and the messenger overworld
    """

    def __init__(self, opt):
        """Create an MessengerManager using the given setup options
        """
        self.opt = opt
        self.server_url = None
        self.port = 443
        self.agent_pool_change_condition = threading.Condition()
        self.overworld_func = None
        self.overworld_threads = {}
        self.active_worlds = {}
        self.message_socket = None
        self.page_id = None
        self._init_logs()
        self.running = True
        self.conversation_index = 0
        self.shutting_down = True

    # Helpers and internal manager methods #

    def _init_logs(self):
        """Initialize logging settings from the opt"""
        shared_utils.set_is_debug(self.opt['is_debug'])
        shared_utils.set_log_level(self.opt['log_level'])

    def add_agent_to_pool(self, agent, world_type='default'):
        """Add the agent to pool"""
        with self.agent_pool_change_condition:
            shared_utils.print_and_log(
                logging.DEBUG,
                "Adding agent {} to pool...".format(agent.messenger_id)
            )
            # time agent entered agent_pool
            agent.time_in_pool.setdefault(world_type, time.time())
            # add agent to pool
            self.agent_pool.setdefault(world_type, []).append(agent)

    def remove_agent_from_pool(self, agent, world_type='default', mark_removed=True):
        """Remove agent from the pool"""
        with self.agent_pool_change_condition:
            shared_utils.print_and_log(
                logging.DEBUG,
                "Removing agent {} from pool...".format(agent.messenger_id)
            )
            if world_type in self.agent_pool and agent in self.agent_pool[world_type]:
                self.agent_pool[world_type].remove(agent)
                # reset agent's time_in_pool
                if world_type in agent.time_in_pool:
                    del agent.time_in_pool[world_type]
                # maybe mark agent as removed
                if mark_removed:
                    agent.stored_data['removed_from_pool']=True

    def _expire_all_conversations(self):
        """iterate through all sub-worlds and shut them down"""
        self.running = False
        for agent_id, overworld_thread in \
                self.agent_id_to_overworld_thread.items():
            self.observe_message(
                agent_id,
                "System: The conversation bot going to go offline soon, "
                "finish any active conversations in the next 5 minutes."
            )
            overworld_thread.join()

        # 5 minute grace period for conversations to finish
        time_passed = 0
        while time_passed < 5 * 60:
            any_alive = False
            for task_thread in self.active_worlds.values():
                if task_thread is not None:
                    any_alive = any_alive or task_thread.isAlive()
            if not any_alive:
                break
            time.sleep(1)

        # Tell the worlds that they should be shutting down at this point,
        # agents will refer to this value
        self.shutting_down = True

    def _get_unique_pool(self):
        """Return a filtered version of the agent pool where each agent is
        only listed a maximum of one time.
        """
        # TODO filter by psid -> agent id mappings for multi-page setup
        return self.agent_pool

    def _on_first_message(self, message):
        """Handle a new incoming message from a psid that is not yet
        registered to any assignment.
        """
        if self.page_id is None:
            self.page_id = message['recipient']['id']
        agent_id = message['sender']['id']
        if self.opt['password'] is not None:
            if message['message']['text'] != self.opt['password']:
                self.observe_message(
                    agent_id,
                    "Sorry, this conversation bot is password-protected. If "
                    "you have the password, please send it now."
                )
                return

        def _overworld_function(opt, agent_id, task_id):
            """Wrapper function for maintaining an overworld"""
            agent_state = self._get_agent_state(agent_id)
            agent = agent_state.get_overworld_agent()
            overworld = self.overworld_func(opt, agent)

            while self.running:
                world_type = overworld.parley()
                if world_type is None:
                    continue
                self._onboard_new_worker(agent_id, world_type)
                time.sleep(5)  # wait for onboard_world to start
                while agent_state.get_active_agent() != agent:
                    time.sleep(1)
                overworld.return_overworld()

        task_id = 'overworld-{}-{}'.format(agent_id, time.time())
        agent = self._create_agent(task_id, agent_id)
        agent_state = AgentState(agent_id, agent)
        self.messenger_agent_states[agent_id] = agent_state
        # Start the onboarding thread and run it
        overworld_thread = threading.Thread(
            target=_overworld_function,
            args=(self.opt, agent_id, task_id),
            name=task_id
        )
        overworld_thread.daemon = True
        overworld_thread.start()

        self.agent_id_to_overworld_thread[agent_id] = overworld_thread
        pass

    def _on_new_message(self, message):
        """Put an incoming message onto the correct agent's message queue.
        """
        agent_id = message['sender']['id']
        if agent_id not in self.messenger_agent_states:
            self._on_first_message(message)
            return
        agent_state = self._get_agent_state(agent_id)
        if agent_state.get_active_agent() is None:
            # return agent to overworld
            if 'text' in message['message'] and message['message']['text'].upper()=='EXIT':
                # remove agent from agent_pool
                to_remove = []
                for world_type, time in agent_state.time_in_pool.items():
                    to_remove.append(world_type)
                for world_type in to_remove:
                    self.remove_agent_from_pool(agent_state, world_type, mark_removed=False)
                # put agent back in overworld
                agent_state.set_active_agent(agent_state.get_overworld_agent())
            else:
                self.observe_message(
                    agent_id,
                    "We are trying to pair you with another person, please wait. "
                    "If you wish to return to the Overworld, click *EXIT*",
                    quick_replies=['EXIT']
                )
        else:
            agent_state.get_active_agent().put_data(message)

    def _create_agent(self, task_id, agent_id):
        """Initialize an agent and return it"""
        return MessengerAgent(self.opt, self, task_id, agent_id, self.page_id)

    def _onboard_new_worker(self, agent_id, world_type):
        """Handle creating an onboarding thread and moving an agent through
        the onboarding process
        """
        def _onboard_function(opt, agent_id, world_type, task_id):
            """Onboarding wrapper to set state to onboarding properly"""
            agent_state = self._get_agent_state(agent_id)
            data = None
            if (world_type in self.onboard_functions and
                    self.onboard_functions[world_type] is not None):
                # call onboarding function
                agent = self._create_agent(task_id, agent_id)
                agent_state.set_active_agent(agent)
                agent_state.assign_agent_to_task(agent, task_id)
                data = self.onboard_functions[world_type](opt, agent, task_id)
                agent_state.onboard_data = data
                agent_state.set_active_agent(None)

            # once onboarding is done, move into a waiting world
            self.add_agent_to_pool(agent_state, world_type)

        task_id = 'onboard-{}-{}'.format(agent_id, time.time())
        # Start the onboarding thread and run it
        onboard_thread = threading.Thread(
            target=_onboard_function,
            args=(self.opt, agent_id, world_type, task_id),
            name=task_id
        )
        onboard_thread.daemon = True
        onboard_thread.start()

        self.agent_id_to_onboard_thread[agent_id] = onboard_thread

    def _get_agent_state(self, agent_id):
        """A safe way to get a worker by agent_id"""
        if agent_id in self.messenger_agent_states:
            return self.messenger_agent_states[agent_id]
        return None

    def _get_agent(self, agent_id, task_id):
        """A safe way to get an agent by agent_id and task_id"""
        agent_state = self._get_agent_state(agent_id)
        if agent_state is not None:
            if agent_state.has_task(task_id):
                return agent_state.get_agent_for_task(task_id)
        return None

    def _log_missing_agent(self, agent_id, assignment_id):
        """Logs when an agent was expected to exist, yet for some reason it
        didn't. If these happen often there is a problem"""
        shared_utils.print_and_log(
            logging.WARN,
            'Expected to have an agent for {}_{}, yet none was found'.format(
                agent_id,
                assignment_id
            )
        )

    def get_agent_state(self, agent_id):
        """Get a worker by agent_id"""
        if agent_id in self.messenger_agent_states:
            return self.messenger_agent_states[agent_id]
        return None

    # Manager Lifecycle Functions #

    def setup_server(self):
        """Prepare the Messenger server for handling messages"""
        shared_utils.print_and_log(
            logging.INFO,
            '\nYou are going to allow people on Facebook to be agents in '
            'ParlAI.\nDuring this process, Internet connection is required, '
            'and you should turn off your computer\'s auto-sleep '
            'feature.\n',
            should_print=True
        )
        input('Please press Enter to continue... ')
        shared_utils.print_and_log(logging.NOTSET, '', True)

        shared_utils.print_and_log(logging.INFO,
                                   'Setting up Messenger webhook...',
                                   should_print=True)

        # Setup the server with a task name related to the current task
        task_name = '{}-{}'.format('ParlAI-Messenger', self.opt['task'])
        self.server_task_name = \
            ''.join(e for e in task_name.lower() if e.isalnum() or e == '-')
        self.server_url = server_utils.setup_server(self.server_task_name)
        shared_utils.print_and_log(
            logging.INFO,
            'Webhook address: {}/webhook'.format(self.server_url),
            should_print=True
        )

    def setup_socket(self):
        """Set up socket to start communicating to workers"""
        shared_utils.print_and_log(logging.INFO,
                                   'Local: Setting up WebSocket...',
                                   should_print=True)
        self.app_token = None
        if self.opt.get('force_page_token'):
            pass
        else:
            if not os.path.exists(os.path.expanduser('~/.parlai/')):
                os.makedirs(os.path.expanduser('~/.parlai/'))
            access_token_file_path = '~/.parlai/messenger_token'
            expanded_file_path = os.path.expanduser(access_token_file_path)
            if os.path.exists(expanded_file_path):
                with open(expanded_file_path, 'r') as access_token_file:
                    self.app_token = access_token_file.read()

        # cache the app token
        if self.app_token is None:
            self.app_token = input(
                'Enter your page\'s access token from the developer page at'
                'https://developers.facebook.com/apps/<YOUR APP ID>'
                '/messenger/settings/ to continue setup:'
            )
            access_token_file_path = '~/.parlai/messenger_token'
            expanded_file_path = os.path.expanduser(access_token_file_path)
            with open(expanded_file_path, 'w+') as access_token_file:
                access_token_file.write(self.app_token)
        self.message_socket = MessageSocket(self.server_url, self.port,
                                            self.app_token,
                                            self._on_new_message)

    def init_new_state(self):
        """Initialize everything in the agent, task, and thread states
        to prepare for a new run
        """
        self.agent_pool = {}
        self.messenger_agent_states = {}
        self.task_to_agent_ids = {}
        self.agent_id_to_onboard_thread = {}
        self.agent_id_to_overworld_thread = {}

    def start_new_run(self):
        self.run_id = str(int(time.time()))
        self.task_group_id = '{}_{}'.format(self.opt['task'], self.run_id)

    def set_onboard_functions(self, onboard_functions):
        self.onboard_functions = onboard_functions

    def set_overworld_func(self, overworld_func):
        self.overworld_func = overworld_func

    def set_agents_required(self, max_agents_for):
        self.max_agents_for = max_agents_for

    def start_task(self, assign_role_functions, task_functions, max_time_in_pool=None):
        """Handle running a task by checking to see when enough agents are
        in the pool to start an instance of the task. Continue doing this
        until the desired number of conversations is had.
        """

        def _task_function(opt, agents, conversation_id, world_type):
            """Run the task function for the given agents"""
            shared_utils.print_and_log(
                logging.INFO,
                'Starting task {}...'.format(conversation_id)
            )
            try:
                task_functions[world_type](self, opt, agents, conversation_id)
            except Exception as e:
                shared_utils.print_and_log(
                    logging.ERROR,
                    'Starting world {} had error {}'.format(world_type, e),
                )
                for agent in agents:
                    self.observe_message(
                        agent.id,
                        "Sorry, this world closed. Returning to overworld."
                    )
            self.active_worlds[task_id] = None
            for agent in agents:
                agent_state = self._get_agent_state(agent.id)
                agent_state.set_active_agent(agent_state.get_overworld_agent())

        self.running = True
        while self.running:
            # Loop forever until the server is shut down
            with self.agent_pool_change_condition:
                valid_pools = self._get_unique_pool()
                for world_type, agent_pool in valid_pools.items():
                    # check if agent has exceeded max time in pool
                    if max_time_in_pool is not None and max_time_in_pool[world_type] is not None:
                        for agent_state in agent_pool:
                            if agent_state.time_in_pool.get(world_type):
                                if time.time() - agent_state.time_in_pool[world_type] > max_time_in_pool[world_type]:
                                    # remove agent from agent_pool
                                    self.remove_agent_from_pool(agent_state, world_type)
                                    # put agent back in overworld
                                    agent_state.set_active_agent(agent_state.get_overworld_agent())

                    needed_agents = self.max_agents_for[world_type]
                    if len(agent_pool) >= needed_agents:
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
                        assign_role_functions[world_type](agents)
                        # Allow task creator to filter out workers and run
                        # versions of the task that require fewer agents
                        agents = [a for a in agents if a.disp_id is not None]
                        for a in agents:
                            # Remove selected workers from the pool
                            agent_pool.remove(self._get_agent_state(a.id))

                        # Start a new thread for this task world
                        task_thread = threading.Thread(
                            target=_task_function,
                            args=(self.opt, agents, task_id, world_type),
                            name='task-{}'.format(task_id)
                        )
                        task_thread.daemon = True
                        task_thread.start()
                        self.active_worlds[task_id] = task_thread

            time.sleep(shared_utils.THREAD_MEDIUM_SLEEP)

    def shutdown(self):
        """Handle any client shutdown cleanup."""
        # Ensure all threads are cleaned and conversations are handled
        try:
            self.is_running = False
            self.message_socket.keep_running = False
            self._expire_all_conversations()
        except BaseException:
            pass
        finally:
            server_utils.delete_server(self.server_task_name)

    # Agent Interaction Functions #

    def observe_message(self, receiver_id, text, quick_replies=None):
        """Send a message through the message manager"""
        return self.message_socket.send_fb_message(receiver_id, text, True,
                                                   quick_replies=quick_replies)

    def observe_payload(self, receiver_id, data):
        """Send a payload through the message manager"""
        return self.message_socket.send_fb_payload(receiver_id, data)
