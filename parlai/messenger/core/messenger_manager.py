# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import logging
import math
import os
import pickle
import threading
import time
import uuid

from botocore.exceptions import ClientError

from parlai.messenger.core.agents import MessengerAgent
from parlai.messenger.core.socket_manager import Packet, SocketManager
from parlai.messenger.core.worker_state import WorkerState, AssignState
import parlai.messenger.core.data_model as data_model
import parlai.messenger.core.messenger_utils as messenger_utils
import parlai.messenger.core.server_utils as server_utils
import parlai.messenger.core.shared_utils as shared_utils

parent_dir = os.path.dirname(os.path.abspath(__file__))


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
        self.run_id = None
        self.agent_pool_change_condition = threading.Condition()
        self.overworld = None
        self.world_options = {}
        self.active_worlds = {}
        self.socket_manager = None
        self._init_logs()

    # Helpers and internal manager methods #

    def _init_state(self):
        """Initialize everything in the agent, task, and thread states"""
        self.agent_pool = []
        self.messenger_agent_states = {}
        self.conv_to_agent = {}
        self.assignment_to_agent_ids = {}

    def _init_logs(self):
        """Initialize logging settings from the opt"""
        shared_utils.set_is_debug(self.opt['is_debug'])
        shared_utils.set_log_level(self.opt['log_level'])

    def _get_agent_from_pkt(self, pkt):
        """Get sender and assignment from a packet"""
        agent_id = pkt.sender_id
        assignment_id = pkt.assignment_id
        agent = self._get_agent(agent_id, assignment_id)
        if agent is None:
            self._log_missing_agent(agent_id, assignment_id)
        return agent

    def add_agent_to_pool(self, agent, world_type='default'):
        """Add the agent to pool"""
        with self.agent_pool_change_condition:
            shared_utils.print_and_log(
                logging.DEBUG,
                "Adding agent {} to pool...".format(agent.id)
            )
            if world_type not in self.agent_pool:
                self.agent_pool[world_type] = []
            self.agent_pool[world_type].append(agent)

    def _expire_all_conversations(self):
        """iterate through all sub-worlds and shut them down"""
        # TODO implement this
        pass

    def _get_unique_pool(self):
        """Return a filtered version of the agent pool where each agent is
        only listed a maximum of one time.
        """
        # TODO filter by psid -> agent id mappings for multi-page setup
        return self.agent_pool

    def _setup_socket(self):
        """Set up a socket_manager with defined callbacks"""
        # TODO update this
        self.socket_manager = SocketManager(
            self.server_url,
            self.port,
            self._on_alive,
            self._on_new_message,
            self._on_socket_dead,
            self.task_group_id
        )

    def _on_first_message(self, pkt):
        """Handle a new incoming message from a psid that is not yet
        registered to any assignment.
        """
        # TODO this should register the user and enter them into the overworld,
        # spawning a thread for that world which will put them in a task
        # world when it is completed
        #
        # world_type = self.prepare_new_worker........
        pass

    def _on_new_message(self, pkt):
        """Put an incoming message onto the correct agent's message queue.
        """
        # TODO find the correct agent to put this message into based on the
        # messenger_id to agent thing, then put it into that agent's message
        # queue on the spot. If no such agent, use the _on_first_message func
        pass

    def _create_agent(self, assignment_id, agent_id):
        """Initialize an agent and return it"""
        return MessengerAgent(self.opt, self, assignment_id, agent_id)

    def _onboard_new_worker(self, messenger_agent, world_type):
        """Handle creating an onboarding thread and moving an agent through
        the onboarding process
        """
        agent_id = messenger_agent.id
        assignment_id = messenger_agent.assignment_id

        def _onboard_function(messenger_agent, world_type):
            """Onboarding wrapper to set state to onboarding properly"""
            if (world_type in self.onboard_functions and
                    self.onboard_functions[world_type] is not None):
                # call onboarding function
                self.onboard_functions[world_type](messenger_agent)

            # once onboarding is done, move into a waiting world
            self.add_agent_to_pool(messenger_agent, world_type)

        if assignment_id not in self.assignment_to_onboard_thread:
            # Start the onboarding thread and run it
            onboard_thread = threading.Thread(
                target=_onboard_function,
                args=(messenger_agent, world_type,),
                name='onboard-{}-{}'.format(agent_id, assignment_id)
            )
            onboard_thread.daemon = True
            onboard_thread.start()

            self.assignment_to_onboard_thread[assignment_id] = onboard_thread
        pass

    def _get_agent_state(self, agent_id):
        """A safe way to get a worker by agent_id"""
        if agent_id in self.messenger_agent_states:
            return self.messenger_agent_states[agent_id]
        return None

    def _get_agent(self, agent_id, assignment_id):
        """A safe way to get an agent by agent_id and assignment_id"""
        agent_state = self._get_agent_state(agent_id)
        if agent_state is not None:
            if agent_state.has_assignment(assignment_id):
                return agent_state.get_agent_for_assignment(assignment_id)
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

    def setup_server(self, task_directory_path=None):
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

        # Setup the server with a likely-unique app-name
        task_name = '{}-{}'.format(str(uuid.uuid4())[:8], self.opt['task'])
        self.server_task_name = \
            ''.join(e for e in task_name.lower() if e.isalnum() or e == '-')
        self.server_url = server_utils.setup_server(self.server_task_name)
        shared_utils.print_and_log(logging.INFO,
                                   'Webhook address: {}/TODO/'.format(
                                       self.server_url),
                                   should_print=True)

    def ready_to_accept_workers(self):
        """Set up socket to start communicating to workers"""
        shared_utils.print_and_log(logging.INFO,
                                   'Local: Setting up SocketIO...',
                                   not self.is_test)
        self._setup_socket()

    def start_new_run(self):
        """Clear state to prepare for a new run"""
        self.run_id = str(int(time.time()))
        self.task_group_id = '{}_{}'.format(self.opt['task'], self.run_id)
        self._init_state()

    def set_onboard_functions(self, onboard_functions):
        self.onboard_functions = onboard_functions

    def start_task(self, eligibility_function, assign_role_functions,
                   task_function):
        """Handle running a task by checking to see when enough agents are
        in the pool to start an instance of the task. Continue doing this
        until the desired number of conversations is had.
        """

        def _task_function(opt, workers, conversation_id):
            """Run the task function for the given workers"""
            shared_utils.print_and_log(
                logging.INFO,
                'Starting task {}...'.format(conversation_id)
            )
            task_function(mturk_manager=self, opt=opt, workers=workers)
            # Delete extra state data that is now unneeded
            for worker in workers:
                worker.state.clear_messages()

        self.is_running = True
        while self.is_running:
            # Loop forever until the server is shut down
            with self.worker_pool_change_condition:
                valid_pools = self._get_unique_pool()
                for world_type, agent_pool in valid_pools.items():
                    needed_agents = len(self.opt['agent_names'][world_type])
                    if len(agent_pool) >= needed_agents:
                        # enough agents in pool to start new conversation
                        self.conversation_index += 1
                        new_conversation_id = \
                            't_{}'.format(self.conversation_index)

                        # Add the required number of valid agents to the conv
                        agents = [w for w in agent_pool[:needed_agents]]
                        assign_role_functions[world_type](agents)
                        # Allow task creator to filter out workers and run
                        # versions of the task that require fewer agents
                        agents = [a for a in agents if a.disp_id is not None]
                        for a in agents:
                            # Remove selected workers from the pool
                            self.agent_pool.remove(a)

                        # Start a new thread for this task world
                        task_thread = threading.Thread(
                            target=_task_function,
                            args=(self.opt, agents, new_conversation_id),
                            name='task-{}'.format(new_conversation_id)
                        )
                        task_thread.daemon = True
                        task_thread.start()
                        self.task_threads.append(task_thread)

            time.sleep(shared_utils.THREAD_MEDIUM_SLEEP)

    def shutdown(self):
        """Handle any client shutdown cleanup."""
        # Ensure all threads are cleaned and conversations are handled
        try:
            self.is_running = False
            self._expire_onboarding_pool()
            self._expire_worker_pool()
            for assignment_id in self.assignment_to_onboard_thread:
                self.assignment_to_onboard_thread[assignment_id].join()
            self.socket_manager.close_all_channels()
        except BaseException:
            pass
        finally:
            server_utils.delete_server(self.server_task_name)
            self._save_disconnects()

    # Agent Interaction Functions #

    def send_message(self, receiver_id, assignment_id, data,
                     blocking=True, ack_func=None):
        """Send a message through the socket manager,
        update conversation state
        """
        data['type'] = data_model.MESSAGE_TYPE_MESSAGE
        # Force messages to have a unique ID
        if 'message_id' not in data:
            data['message_id'] = str(uuid.uuid4())
        event_id = shared_utils.generate_event_id(receiver_id)
        packet = Packet(
            event_id,
            Packet.TYPE_MESSAGE,
            self.socket_manager.get_my_sender_id(),
            receiver_id,
            assignment_id,
            data,
            blocking=blocking,
            ack_func=ack_func
        )

        shared_utils.print_and_log(
            logging.INFO,
            'Manager sending: {}'.format(packet),
            should_print=self.opt['verbose']
        )
        # Push outgoing message to the message thread to be able to resend
        # on a reconnect event
        agent = self._get_agent(receiver_id, assignment_id)
        if agent is not None:
            agent.state.messages.append(packet.data)
        self.socket_manager.queue_packet(packet)
