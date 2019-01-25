#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import threading
import time
import traceback

from parlai.messenger.core.agents import MessengerAgent
from parlai.messenger.core.message_socket import MessageSocket
from parlai.messenger.core.message_sender import MessageSender
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
        self.message_sender = None
        self.page_id = None
        self._init_logs()
        self.running = True
        self.conversation_index = 0
        self.shutting_down = True
        self.bypass_server_setup = self.opt.get('bypass_server_setup')

        # Messaging interaction functions that determine what to do when
        # messages are confirmed as delivered, marked as read by a user, and
        # noted as read by the bot.
        self.confirm_message_delivery = self._confirm_message_delivery
        self.handle_message_read = self._handle_message_read
        self.handle_bot_read = self._handle_bot_read

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

    def mark_removed(self, agent_id, pageid):
        """Mark the agent as removed from the pool. Can be overriden to change
        other metadata linked to agent removal."""
        pass

    def remove_agent_from_pool(self, agent, world_type='default',
                               mark_removed=True):
        """Remove agent from the pool."""
        with self.agent_pool_change_condition:
            shared_utils.print_and_log(
                logging.DEBUG,
                "Removing agent {} from pool...".format(agent.messenger_id)
            )
            if world_type in self.agent_pool and \
                    agent in self.agent_pool[world_type]:
                self.agent_pool[world_type].remove(agent)
                # reset agent's time_in_pool
                if world_type in agent.time_in_pool:
                    del agent.time_in_pool[world_type]
                # maybe mark agent as removed
                if mark_removed:
                    agent.stored_data['removed_from_pool'] = True
                    if self.page_id is not None:
                        self.mark_removed(
                            int(agent.messenger_id), int(self.page_id))

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

    def _get_unique_pool(self, eligibility_functions):
        """Return a filtered version of the agent pool where each agent is
        only listed a maximum of one time.
        """
        # TODO filter by psid -> agent id mappings for multi-page setup
        if eligibility_functions is None:
            return self.agent_pool

        valid_pools = {}
        for world_type, agent_pool in self.agent_pool.items():
            if world_type in eligibility_functions and \
                    eligibility_functions[world_type] is not None:
                valid_pools[world_type] = [
                    w for w in agent_pool
                    if eligibility_functions[world_type](w)
                ]
            else:
                valid_pools[world_type] = self.agent_pool[world_type]
        return valid_pools

    def _handle_bot_read(self, agent_id):
        self.message_sender.send_read(agent_id)
        self.message_sender.typing_on(agent_id)

    def _confirm_message_delivery(self, event):
        # By default we don't actually do anything when messages are marked as
        # being delivered, but we expose the ability for others to
        shared_utils.print_and_log(
            logging.DEBUG,
            "Messages {} marked as received.".format(event['delivery']['mids'])
        )

    def _handle_message_read(self, event):
        # If the message was sent by another user (as in during a conversation)
        # then we need to propogate the read back to that user.
        shared_utils.print_and_log(
            logging.DEBUG,
            "Messages {} marked as read.".format(event['read'])
        )
        reader = event['sender']['id']
        agent_state = self.get_agent_state(reader)
        if agent_state is None:
            return
        agent = agent_state.get_active_agent()
        if agent is not None:
            for partner in agent.message_partners:
                # We don't know who sent the message that was seen, but we can
                # send a message observed event to everyone else in the chat
                self.message_sender.send_read(partner.id)

    def _handle_webhook_event(self, event):
        if 'message' in event:
            if (('image_url' in event and event['image_url'] is not None) or
                    ('attachment_url' in event and event['attachment_url'] is
                        not None)):
                event['message']['image'] = True
            self._on_new_message(event)
        elif 'delivery' in event:
            self.confirm_message_delivery(event)
        elif 'read' in event:
            self.handle_message_read(event)

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
            agent_state = self.get_agent_state(agent_id)
            if self.opt['password'] is None:
                agent_state.stored_data['first_message'] = message
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

    def after_agent_removed(self, agent_id):
        """Perform any changes to metadata on agent removal
        override if extra bookkeeping must be done when removing agent"""
        pass

    def _on_new_message(self, message):
        """Put an incoming message onto the correct agent's message queue.
        """
        agent_id = message['sender']['id']
        if agent_id not in self.messenger_agent_states:
            self._on_first_message(message)
            return
        agent_state = self.get_agent_state(agent_id)
        if agent_state.get_active_agent() is None:
            # return agent to overworld
            if 'text' in message['message'] and \
                    message['message']['text'].upper() == 'EXIT':
                # remove agent from agent_pool
                to_remove = []
                for world_type, _time in agent_state.time_in_pool.items():
                    to_remove.append(world_type)
                for world_type in to_remove:
                    self.remove_agent_from_pool(agent_state, world_type,
                                                mark_removed=False)
                self.after_agent_removed(agent_state.get_id())
                agent_state.set_active_agent(agent_state.get_overworld_agent())
            else:
                self.observe_message(
                    agent_id,
                    "Please wait while we pair you with another person. "
                    "If you wish to exit, type *EXIT*."
                )
                self.message_sender.typing_on(agent_id)
        else:
            # If an agent is in a solo world, we can put a typing indicator
            # and mark the message as read
            agent = agent_state.get_active_agent()
            if len(agent.message_partners) == 0:
                self.handle_bot_read(agent.id)
            agent.put_data(message)

    def _create_agent(self, task_id, agent_id):
        """Initialize an agent and return it"""
        return MessengerAgent(self.opt, self, task_id, agent_id, self.page_id)

    def _onboard_new_worker(self, agent_id, world_type):
        """Handle creating an onboarding thread and moving an agent through
        the onboarding process
        """
        def _onboard_function(opt, agent_id, world_type, task_id):
            """Onboarding wrapper to set state to onboarding properly"""
            agent_state = self.get_agent_state(agent_id)
            data = None
            if (world_type in self.onboard_functions and
                    self.onboard_functions[world_type] is not None):
                # call onboarding function
                agent = self._create_agent(task_id, agent_id)
                agent_state.set_active_agent(agent)
                agent_state.assign_agent_to_task(agent, task_id)
                try:
                    data = \
                        self.onboard_functions[world_type](opt, agent, task_id)
                    agent_state.onboard_data = data
                    agent_state.set_active_agent(None)
                except Exception as e:
                    shared_utils.print_and_log(
                        logging.ERROR,
                        'Onboard {} had error {}'.format(world_type, repr(e)),
                        should_print=True
                    )
                    traceback.print_exc(file=sys.stderr)
                    self.observe_message(
                        agent.id,
                        "Sorry, this world closed. Returning to overworld."
                    )
                    agent_state.set_active_agent(
                        agent_state.get_overworld_agent())
                    return

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

    def get_agent_state(self, agent_id):
        """A safe way to get a worker by agent_id"""
        if agent_id in self.messenger_agent_states:
            return self.messenger_agent_states[agent_id]
        return None

    def _get_agent(self, agent_id, task_id):
        """A safe way to get an agent by agent_id and task_id"""
        agent_state = self.get_agent_state(agent_id)
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

    # Manager Lifecycle Functions #
    def setup_server(self):
        """Prepare the Messenger server for handling messages"""
        if self.bypass_server_setup:
            return

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

        if self.opt['local'] is True:
            shared_utils.print_and_log(
                logging.INFO,
                "In order to run the server locally, you will need "
                "to have a public HTTPS endpoint (SSL signed) running on "
                "the server you are currently excecuting ParlAI on. Enter "
                "that public URL hostname when prompted and ensure that the "
                "port being used by ParlAI (usually 3000) has external "
                "traffic routed to it.",
                should_print=True,
            )
            input('Please press Enter to continue... ')

        shared_utils.print_and_log(logging.INFO,
                                   'Setting up Messenger webhook...',
                                   should_print=True)

        # Setup the server with a task name related to the current task
        task_name = '{}-{}'.format('ParlAI-Messenger', self.opt['task'])
        self.server_task_name = \
            ''.join(e for e in task_name.lower() if e.isalnum() or e == '-')
        self.server_url = server_utils.setup_server(
            self.server_task_name, local=self.opt['local'])
        shared_utils.print_and_log(
            logging.INFO,
            'Webhook address: {}/webhook'.format(self.server_url),
            should_print=True
        )

    # override if permission needed externally
    def get_app_token(self):
        """Find and return an app access token"""
        if not self.opt.get('force_page_token'):
            if not os.path.exists(os.path.expanduser('~/.parlai/')):
                os.makedirs(os.path.expanduser('~/.parlai/'))
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
        """Set up socket to start communicating to workers"""
        if not self.bypass_server_setup:
            shared_utils.print_and_log(logging.INFO,
                                       'Local: Setting up WebSocket...',
                                       should_print=True)

        self.app_token = self.get_app_token()
        self.message_sender = MessageSender(self.app_token)

        # Set up receive
        if not self.bypass_server_setup:
            socket_use_url = self.server_url
            if (self.opt['local']):  # skip some hops for local stuff
                socket_use_url = "https://localhost"
            self.message_socket = MessageSocket(socket_use_url, self.port,
                                                self._handle_webhook_event)

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

    def check_timeout_in_pool(self, world_type, agent_pool, max_time_in_pool):
        for agent_state in agent_pool:
            time_in_pool = agent_state.time_in_pool.get(world_type)
            if time_in_pool and time.time() - time_in_pool \
                    > max_time_in_pool[world_type]:
                # remove agent from agent_pool
                self.remove_agent_from_pool(
                    agent_state, world_type)
                # put agent back in overworld
                agent_state.set_active_agent(
                    agent_state.get_overworld_agent())

                agent_state.stored_data['removed_after_timeout'] = True
                self.after_agent_removed(agent_state.messenger_id)

                # reset wait message state
                agent_state.stored_data['seen_wait_message'] = False

            elif time_in_pool and time.time() - time_in_pool > 30:
                # tell agent that a match is taking longer than
                # expected
                if not agent_state.stored_data.get('seen_wait_message') \
                        or not agent_state.stored_data['seen_wait_message']:
                    self.observe_message(
                        agent_state.messenger_id,
                        "Pairing is taking longer than expected. "
                        "If you wish to exit, type *EXIT*.",
                    )
                    self.message_sender.typing_on(agent_state.messenger_id)
                    agent_state.stored_data['seen_wait_message'] = True

    def start_task(self, assign_role_functions, task_functions,
                   max_time_in_pool=None, eligibility_functions=None):
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
                    'World {} had error {}'.format(world_type, repr(e)),
                    should_print=True,
                )
                print("Exception in user code:")
                traceback.print_exc(file=sys.stdout)
                for agent in agents:
                    self.observe_message(
                        agent.id,
                        "Sorry, this world closed. Returning to overworld."
                    )
            self.active_worlds[task_id] = None
            for agent in agents:
                self.after_agent_removed(agent.id)
                agent_state = self.get_agent_state(agent.id)
                agent_state.set_active_agent(agent_state.get_overworld_agent())

        self.running = True
        while self.running:
            # Loop forever until the server is shut down
            with self.agent_pool_change_condition:
                valid_pools = self._get_unique_pool(eligibility_functions)
                for world_type, agent_pool in valid_pools.items():
                    # check if agent has exceeded max time in pool
                    if max_time_in_pool is not None and \
                            max_time_in_pool[world_type] is not None:
                        self.check_timeout_in_pool(world_type, agent_pool,
                                                   max_time_in_pool)

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
                            # reset wait message state
                            state.stored_data['seen_wait_message'] = False
                        assign_role_functions[world_type](agents)
                        # Allow task creator to filter out workers and run
                        # versions of the task that require fewer agents
                        agents = [a for a in agents if a.disp_id is not None]
                        for a in agents:
                            # Remove selected workers from the agent pool
                            self.remove_agent_from_pool(
                                self.get_agent_state(a.id),
                                world_type=world_type,
                                mark_removed=False
                            )
                        for a in agents:
                            partner_list = agents.copy()
                            partner_list.remove(a)
                            a.message_partners = partner_list
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
            if not self.bypass_server_setup:
                self.message_socket.keep_running = False
            self._expire_all_conversations()
        except BaseException:
            pass
        finally:
            if not self.bypass_server_setup:
                server_utils.delete_server(self.server_task_name,
                                           self.opt['local'])

    # Agent Interaction Functions #

    def observe_message(self, receiver_id, text, quick_replies=None,
                        persona_id=None):
        """Send a message through the message manager"""
        return self.message_sender.send_fb_message(receiver_id, text, True,
                                                   quick_replies=quick_replies,
                                                   persona_id=persona_id)

    def observe_payload(self, receiver_id, data, quick_replies=None,
                        persona_id=None):
        """Send a payload through the message manager"""
        return self.message_sender.send_fb_payload(receiver_id, data,
                                                   quick_replies=quick_replies,
                                                   persona_id=persona_id)

    def upload_attachment(self, payload):
        """Uploads an attachment and returns an attachment ID
        `payload` should be a dict of the format
        {'type': <TYPE>, 'url': <URL>} or
        {'type': <TYPE>, 'filename': <FILENAME>, 'format': <FILEFORMAT>}.
        For example,
        {'type': 'image', 'filename': 'test.png', 'format': 'png'}
        """
        return self.message_sender.upload_fb_attachment(payload)
