#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Messenger Manager Module.

Contains implementation of the MessengerManager, which helps run
ParlAI via FB Messenger.
"""

import logging
import os
import sys
import threading
import time
import traceback
import datetime

from parlai.core.agents import create_agent
from parlai.chat_service.services.messenger.agents import MessengerAgent
from parlai.chat_service.services.messenger.message_socket import MessageSocket
from parlai.chat_service.services.messenger.message_sender import MessageSender
import parlai.chat_service.services.messenger.server_utils as server_utils
import parlai.chat_service.services.messenger.shared_utils as shared_utils
from parlai.chat_service.services.messenger.world_runner import MessengerWorldRunner

parent_dir = os.path.dirname(os.path.abspath(__file__))


class AgentState:
    """Keep track of Agent State.

    State includes which is the "active" agent - i.e., which agent in which
    world do we message, etc.
    """

    def __init__(self, messenger_id, overworld_agent):
        self.messenger_id = messenger_id
        self.overworld_agent = overworld_agent
        self.active_agent = overworld_agent
        self.task_id_to_agent = {}
        self.onboard_data = None
        self.stored_data = {}
        self.time_in_pool = {}

    def get_active_agent(self):
        """Return active messenger agent.

        :return:
            a MessengerAgent, which corresponds to the active agent for this
            agent state.
        """
        return self.active_agent

    def set_active_agent(self, active_agent):
        """Set active agent for this agent.

        :param active_agent:
            A MessengerAgent, the new active agent for this given agent state

        """
        self.active_agent = active_agent

    def get_overworld_agent(self):
        """Return overworld messenger agent.

        :return:
            a MessengerAgent, which corresponds agent object in the overworld
        """
        return self.overworld_agent

    def get_id(self):
        """Return agent's ID.

        :return:
            int agent ID
        """
        return self.messenger_id

    def has_task(self, task_id):
        """Determine if an agent is in a task.

        :param task_id:
            string task id

        :return:
            if agent is in that task.
        """
        return task_id in self.task_id_to_agent

    def get_agent_for_task(self, task_id):
        """Return MessengerAgent for given task id.

        For each "player", a separate agent is created for each task. This
        returns the appropriate MessengerAgent given the task id

        :param task_id:
            string, task id

        :return:
            messenger agent object associated with the given task
        """
        if self.has_task(task_id):
            return self.task_id_to_agent[task_id]
        else:
            return None

    def assign_agent_to_task(self, agent, task_id):
        """Mark agent in task.

        :param agent:
            MessengerAgent object to mark in task
        :param task_id:
            string task name
        """
        self.task_id_to_agent[task_id] = agent


class MessengerManager:
    """Manages interactions between agents on messenger as well as direct
    interactions between agents and the messenger overworld
    """

    def __init__(self, opt):
        """Create an MessengerManager using the given setup options
        """
        # Manager attributes
        self.opt = opt
        self.server_url = None
        self.port = 443
        self.agent_pool_change_condition = threading.Condition()
        self.active_worlds = {}
        self.message_socket = None
        self.message_sender = None
        self._init_logs()
        self.running = True
        self.conversation_index = 0
        self.shutting_down = False
        self.bypass_server_setup = self.opt.get('bypass_server_setup')

        # Messaging interaction functions that determine what to do when
        # messages are confirmed as delivered, marked as read by a user, and
        # noted as read by the bot.
        self.confirm_message_delivery = self._confirm_message_delivery
        self.handle_message_read = self._handle_message_read
        self.handle_bot_read = self._handle_bot_read

        # Read in Config
        self._parse_config(opt)
        self._complete_setup()

    # Helpers and internal manager methods #

    def _log_debug(self, text):
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        shared_utils.print_and_log(logging.DEBUG, f'{time}: {text}', should_print=True)

    def _parse_config(self, opt):
        """Parse config for task."""
        self.config = opt['config']
        self.overworld = self.config['overworld']
        self.world_path = self.config['world_path']
        self.world_module = shared_utils.get_world_module(self.world_path)
        self.page_id = self.config['page_id']
        if self.page_id == 1:
            raise RuntimeError(
                'Please configure your own page in order to run this task. '
                'See the docs (https://parl.ai/docs/tutorial_messenger.html) '
                'for more information.'
            )
        self.task_configs = self.config['configs']
        self.max_workers = self.config['max_workers']
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

    def _complete_setup(self):
        """Complete necessary setup items."""
        self.setup_server()
        self.init_new_state()
        self.setup_socket()
        self.start_new_run()
        self._load_model()

    def _load_model(self):
        """Load model if necessary."""
        if 'model_file' in self.opt or 'model' in self.opt:
            self.opt['shared_bot_params'] = create_agent(self.opt).share()

    def _init_logs(self):
        """Initialize logging settings from the opt."""
        shared_utils.set_is_debug(self.opt['is_debug'])
        shared_utils.set_log_level(self.opt['log_level'])

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

    def mark_removed(self, agent_id, pageid):
        """Mark the agent as removed from the pool.

        Can be overriden to change other metadata linked to agent removal.

        :param agent_id:
            int agent psid
        :param pageid:
            int page id
        """
        pass

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
                    if self.page_id is not None:
                        self.mark_removed(int(agent.messenger_id), int(self.page_id))

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

    def _handle_bot_read(self, agent_id):
        self.message_sender.send_read(agent_id)
        self.message_sender.typing_on(agent_id)

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
            if ('image_url' in event and event['image_url'] is not None) or (
                'attachment_url' in event and event['attachment_url'] is not None
            ):
                event['message']['image'] = True
            self._on_new_message(event)
        elif 'delivery' in event:
            self.confirm_message_delivery(event)
        elif 'read' in event:
            self.handle_message_read(event)

    def _on_first_message(self, message):
        """Handle first message from player.

        Run when a psid is given that is not paired with any assignment yet.
        Launch an overworld, complete onboarding, etc.

        :param message:
            message sent from agent
        """
        if self.page_id is None:
            self.page_id = message['recipient']['id']

        agent_id = message['sender']['id']
        if self.opt['password'] is not None:
            if message['message']['text'] != self.opt['password']:
                self.observe_message(
                    agent_id,
                    'Sorry, this conversation bot is password-protected. If '
                    'you have the password, please send it now.',
                )
                return

        task_id = 'overworld-{}-{}'.format(agent_id, time.time())
        agent = self._create_agent(task_id, agent_id)
        agent_state = AgentState(agent_id, agent)
        if self.opt['password'] is not None:
            agent_state.stored_data['first_message'] = message
        self.messenger_agent_states[agent_id] = agent_state

        # launch overworld
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
        self.agent_id_to_overworld_future[agent_id] = future

    def after_agent_removed(self, agent_id):
        """Perform any changes to metadata on agent removal.

        override if extra bookkeeping must be done when removing agent
        """
        pass

    def _on_new_message(self, message):
        """Put an incoming message onto the correct agent's message queue.

        :param message:
            message to put on queue
        """
        agent_id = message['sender']['id']
        if agent_id not in self.messenger_agent_states:
            self._on_first_message(message)
            return

        agent_state = self.get_agent_state(agent_id)
        if agent_state.get_active_agent() is None:
            # return agent to overworld
            if (
                'text' in message['message']
                and message['message']['text'].upper() == 'EXIT'
            ):
                # remove agent from agent_pool
                to_remove = []
                for world_type, _time in agent_state.time_in_pool.items():
                    to_remove.append(world_type)
                for world_type in to_remove:
                    self.remove_agent_from_pool(
                        agent_state, world_type, mark_removed=False
                    )
                self.after_agent_removed(agent_state.get_id())
                agent_state.set_active_agent(agent_state.get_overworld_agent())
            else:
                self.observe_message(
                    agent_id,
                    'Please wait while we pair you with another person. '
                    'If you wish to exit, type *EXIT*.',
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
        """Initialize an agent and return it.

        Called each time an agent is placed into a new task.

        :param task_id:
            string task identifier
        :param agent_id:
            int agent id
        """
        return MessengerAgent(self.opt, self, task_id, agent_id, self.page_id)

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

    def _log_missing_agent(self, agent_id, assignment_id):
        """Log the occurence of a missing agent."""
        shared_utils.print_and_log(
            logging.WARN,
            'Expected to have an agent for {}_{}, yet none was found'.format(
                agent_id, assignment_id
            ),
        )

    # Manager Lifecycle Functions #
    def setup_server(self):
        """Prepare the Messenger server for handling messages."""
        if self.bypass_server_setup:
            return

        shared_utils.print_and_log(
            logging.INFO,
            '\nYou are going to allow people on Facebook to be agents in '
            'ParlAI.\nDuring this process, Internet connection is required, '
            'and you should turn off your computer\'s auto-sleep '
            'feature.\n',
            should_print=True,
        )
        input('Please press Enter to continue... ')
        shared_utils.print_and_log(logging.NOTSET, '', True)

        if self.opt['local'] is True:
            shared_utils.print_and_log(
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

        shared_utils.print_and_log(
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
        shared_utils.print_and_log(
            logging.INFO,
            'Webhook address: {}/webhook'.format(self.server_url),
            should_print=True,
        )

    # override if permission needed externally
    def get_app_token(self):
        """Find and return an app access token."""
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
        """Set up socket to start communicating to workers."""
        if not self.bypass_server_setup:
            shared_utils.print_and_log(
                logging.INFO, 'Local: Setting up WebSocket...', should_print=True
            )

        self.app_token = self.get_app_token()
        self.message_sender = MessageSender(self.app_token)

        # Set up receive
        if not self.bypass_server_setup:
            socket_use_url = self.server_url
            if self.opt['local']:  # skip some hops for local stuff
                socket_use_url = 'https://localhost'
            self.message_socket = MessageSocket(
                socket_use_url, self.port, self._handle_webhook_event
            )
        shared_utils.print_and_log(
            logging.INFO, 'done with websocket', should_print=True
        )

    def init_new_state(self):
        """Prepare for new run.

        Initialize everything in the agent, task, and thread states
        """
        self.agent_pool = {}
        self.messenger_agent_states = {}
        self.task_to_agent_ids = {}
        self.agent_id_to_overworld_future = {}

    def start_new_run(self):
        """Begin new run."""
        self.run_id = str(int(time.time()))
        self.task_group_id = '{}_{}'.format(self.opt['task'], self.run_id)

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
                self.after_agent_removed(agent_state.messenger_id)

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
                    self.message_sender.typing_on(agent_state.messenger_id)
                    agent_state.stored_data['seen_wait_message'] = True

    def start_task(self):
        """Begin handling task.

        Periodically check to see when enough agents are in the agent pool
        to start an instance of the task. Continue doing this until the desired
        number of conversations is had.
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
                self.after_agent_removed(agent.id)
                agent_state = self.get_agent_state(agent.id)
                agent_state.set_active_agent(agent_state.get_overworld_agent())

        self.running = True
        while self.running:
            # Loop forever until the server is shut down
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
                        agents = [a for a in agents if a.disp_id is not None]
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

            time.sleep(shared_utils.THREAD_MEDIUM_SLEEP)

    def shutdown(self):
        """Handle any client shutdown cleanup."""
        # Ensure all threads are cleaned and conversations are handled
        try:
            self.is_running = False
            self.world_runner.shutdown()
            if not self.bypass_server_setup:
                self.message_socket.keep_running = False
            self._expire_all_conversations()
        except BaseException as e:
            shared_utils.print_and_log(logging.ERROR, f'world ended in error: {e}')

        finally:
            if not self.bypass_server_setup:
                server_utils.delete_server(self.server_task_name, self.opt['local'])

    # Agent Interaction Functions #

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
        return self.message_sender.send_fb_message(
            receiver_id, text, True, quick_replies=quick_replies, persona_id=persona_id
        )

    def observe_payload(self, receiver_id, data, quick_replies=None, persona_id=None):
        """Send a payload through the message manager.

        :param receiver_id:
            int identifier for agent to send message to
        :param data:
            object data to send
        :param quick_replies:
            list of quick replies
        :param persona_id:
            identifier of persona
        """
        return self.message_sender.send_fb_payload(
            receiver_id, data, quick_replies=quick_replies, persona_id=persona_id
        )

    def upload_attachment(self, payload):
        """Upload an attachment and return an attachment ID.

        :param payload:
            dict with the following format:
                {'type': <TYPE>, 'url': <URL>} or
                {'type': <TYPE>, 'filename': <FILENAME>, 'format': <FILEFORMAT>}.
                For example,
                {'type': 'image', 'filename': 'test.png', 'format': 'png'}
        """
        return self.message_sender.upload_fb_attachment(payload)
