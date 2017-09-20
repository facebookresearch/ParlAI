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

from parlai.mturk.core.agents import MTurkAgent
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.worker_state import WorkerState, AssignState
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.mturk_utils as mturk_utils
import parlai.mturk.core.server_utils as server_utils
import parlai.mturk.core.shared_utils as shared_utils

# Timeout before cancelling a world start
WORLD_START_TIMEOUT = 11
HEARTBEAT_DELAY_TIME = WORLD_START_TIMEOUT - SocketManager.DEF_SOCKET_TIMEOUT

# Multiplier to apply when creating hits to ensure worker availibility
HIT_MULT = 1.5

# Max number of conversation disconnects before a turker should be blocked
MAX_DISCONNECTS = 5

# Time to persist a disconnect before forgetting about it. Combined with the
# above this will block workers that disconnect at least 25 times in a week
DISCONNECT_PERSIST_LENGTH = 60 * 24 * 7

DISCONNECT_FILE_NAME = 'disconnects.pickle'

parent_dir = os.path.dirname(os.path.abspath(__file__))

class MTurkManager():
    """Manages interactions between MTurk agents as well as direct interactions
    between a world and the MTurk server.
    """

    def __init__(self, opt, mturk_agent_ids, is_test=False):
        """Create an MTurkManager using the given setup opts and a list of
        agent_ids that will participate in each conversation
        """
        self.opt = opt
        self.server_url = None
        self.port = 443
        self.task_group_id = None
        self.run_id = None
        self.mturk_agent_ids = mturk_agent_ids
        self.task_files_to_copy = None
        self.is_sandbox = opt['is_sandbox']
        self.worker_pool_change_condition = threading.Condition()
        self.onboard_function = None
        self.num_conversations = opt['num_conversations']
        self.required_hits = math.ceil(
            self.num_conversations * len(self.mturk_agent_ids) * HIT_MULT
        )
        self.socket_manager = None
        self.is_test = is_test
        self._init_logs()


    ### Helpers and internal manager methods ###

    def _init_state(self):
        """Initialize everything in the worker, task, and thread states"""
        self.hit_id_list = []
        self.worker_pool = []
        self.assignment_to_onboard_thread = {}
        self.task_threads = []
        self.conversation_index = 0
        self.started_conversations = 0
        self.completed_conversations = 0
        self.mturk_workers = {}
        self.conv_to_agent = {}
        self.accepting_workers = True
        self._load_disconnects()

    def _init_logs(self):
        """Initialize logging settings from the opt"""
        shared_utils.set_is_debug(self.opt['is_debug'])
        shared_utils.set_log_level(self.opt['log_level'])

    def _load_disconnects(self):
        """Load disconnects from file, populate the disconnects field for any
        worker_id that has disconnects in the list. Any disconnect that
        occurred longer ago than the disconnect persist length is ignored
        """
        self.disconnects = []
        # Load disconnects from file
        file_path = os.path.join(parent_dir, DISCONNECT_FILE_NAME)
        compare_time = time.time()
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                old_disconnects = pickle.load(f)
                self.disconnects = [
                    d
                    for d in old_disconnects
                    if (compare_time - d['time']) < DISCONNECT_PERSIST_LENGTH
                ]
        # Initialize worker states with proper number of disconnects
        for disconnect in self.disconnects:
            worker_id = disconnect['id']
            if not worker_id in self.mturk_workers:
                # add this worker to the worker state
                self.mturk_workers[worker_id] = WorkerState(worker_id)
            self.mturk_workers[worker_id].disconnects += 1

    def _save_disconnects(self):
        """Saves the local list of disconnects to file"""
        file_path = os.path.join(parent_dir, DISCONNECT_FILE_NAME)
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self.disconnects, f, pickle.HIGHEST_PROTOCOL)

    def _handle_bad_disconnect(self, worker_id):
        """Update the number of bad disconnects for the given worker, block
        them if they've exceeded the disconnect limit
        """
        if not self.is_sandbox:
            self.mturk_workers[worker_id].disconnects += 1
            self.disconnects.append({'time': time.time(), 'id': worker_id})
            if self.mturk_workers[worker_id].disconnects > MAX_DISCONNECTS:
                text = (
                    'This worker has repeatedly disconnected from these tasks,'
                    ' which require constant connection to complete properly '
                    'as they involve interaction with other Turkers. They have'
                    ' been blocked to ensure a better experience for other '
                    'workers who don\'t disconnect.'
                )
                self.block_worker(worker_id, text)
                shared_utils.print_and_log(
                    logging.INFO,
                    'Worker {} was blocked - too many disconnects'.format(
                        worker_id
                    ),
                    True
                )

    def _get_agent_from_pkt(self, pkt):
        """Get sender, assignment, and conv ids from a packet"""
        worker_id = pkt.sender_id
        assignment_id = pkt.assignment_id
        agent = self._get_agent(worker_id, assignment_id)
        if agent == None:
            self._log_missing_agent(worker_id, assignment_id)
        return agent

    def _change_worker_to_conv(self, pkt):
        """Update a worker to a new conversation given a packet from the
        conversation to be switched to
        """
        agent = self._get_agent_from_pkt(pkt)
        if agent is not None:
            self._assign_agent_to_conversation(agent, agent.conversation_id)

    def _set_worker_status_to_onboard(self, pkt):
        """Changes assignment status to onboarding based on the packet"""
        agent = self._get_agent_from_pkt(pkt)
        if agent is not None:
            agent.state.status = AssignState.STATUS_ONBOARDING

    def _set_worker_status_to_waiting(self, pkt):
        """Changes assignment status to waiting based on the packet"""
        agent = self._get_agent_from_pkt(pkt)
        if agent is not None:
            agent.state.status = AssignState.STATUS_WAITING

            # Add the worker to pool
            with self.worker_pool_change_condition:
                shared_utils.print_and_log(
                    logging.DEBUG,
                    "Adding worker {} to pool...".format(agent.worker_id)
                )
                self.worker_pool.append(agent)

    def _move_workers_to_waiting(self, workers):
        """Put all workers into waiting worlds, expire them if no longer
        accepting workers. If the worker is already final, delete it
        """
        for worker in workers:
            worker_id = worker.worker_id
            assignment_id = worker.assignment_id
            if worker.state.is_final():
                worker.reduce_state()
                self.socket_manager.close_channel(worker.get_connection_id())
                continue

            conversation_id = 'w_{}'.format(uuid.uuid4())
            if self.accepting_workers:
                # Move the worker into a waiting world
                worker.change_conversation(
                    conversation_id=conversation_id,
                    agent_id='waiting',
                    change_callback=self._set_worker_status_to_waiting
                )
            else:
                self.force_expire_hit(worker_id, assignment_id)

    def _expire_onboarding_pool(self):
        """Expire any worker that is in an onboarding thread"""
        for worker_id in self.mturk_workers:
            for assign_id in self.mturk_workers[worker_id].agents:
                agent = self.mturk_workers[worker_id].agents[assign_id]
                if (agent.state.status == AssignState.STATUS_ONBOARDING):
                    self.force_expire_hit(worker_id, assign_id)

    def _expire_worker_pool(self):
        """Expire all workers in the worker pool"""
        for agent in self.worker_pool:
            self.force_expire_hit(agent.worker_id, agent.assignment_id)

    def _get_unique_pool(self, eligibility_function):
        """Return a filtered version of the worker pool where each worker is
        only listed a maximum of one time. In sandbox this is overridden for
        testing purposes, and the same worker can be returned more than once
        """
        workers = [w for w in self.worker_pool
                   if not w.hit_is_returned and eligibility_function(w)]
        unique_workers = []
        unique_worker_ids = []
        for w in workers:
            if (self.is_sandbox) or (w.worker_id not in unique_worker_ids):
                unique_workers.append(w)
                unique_worker_ids.append(w.worker_id)
        return unique_workers

    def _handle_worker_disconnect(self, worker_id, assignment_id):
        """Mark a worker as disconnected and send a message to all agents in
        his conversation that a partner has disconnected.
        """
        agent = self._get_agent(worker_id, assignment_id)
        if agent is None:
            self._log_missing_agent(worker_id, assignment_id)
        else:
            # Disconnect in conversation is not workable
            agent.state.status = AssignState.STATUS_DISCONNECT
            # in conversation, inform others about disconnect
            conversation_id = agent.conversation_id
            if agent in self.conv_to_agent[conversation_id]:
                for other_agent in self.conv_to_agent[conversation_id]:
                    if agent.assignment_id != other_agent.assignment_id:
                        self._handle_partner_disconnect(
                            other_agent.worker_id,
                            other_agent.assignment_id
                        )
            if len(self.mturk_agent_ids) > 1:
                # The user disconnected from inside a conversation with
                # another turker, record this as bad behavoir
                self._handle_bad_disconnect(worker_id)

    def _handle_partner_disconnect(self, worker_id, assignment_id):
        """Send a message to a worker notifying them that a partner has
        disconnected and we marked the HIT as complete for them
        """
        agent = self._get_agent(worker_id, assignment_id)
        if agent is None:
            self._log_missing_agent(worker_id, assignment_id)
        elif not agent.state.is_final():
            # Update the assignment state
            agent.some_agent_disconnected = True
            agent.state.status = AssignState.STATUS_PARTNER_DISCONNECT

            # Create and send the command
            data = agent.get_inactive_command_data()
            self.send_command(worker_id, assignment_id, data)

    def _restore_worker_state(self, worker_id, assignment_id):
        """Send a command to restore the state of an agent who reconnected"""
        agent = self._get_agent(worker_id, assignment_id)
        if agent is None:
            self._log_missing_agent(worker_id, assignment_id)
        else:
            def _push_worker_state(msg):
                if len(agent.state.messages) != 0:
                    data = {
                        'text': data_model.COMMAND_RESTORE_STATE,
                        'messages': agent.state.messages,
                        'last_command': agent.state.last_command
                    }
                    self.send_command(worker_id, assignment_id, data)

            agent.change_conversation(
                conversation_id=agent.conversation_id,
                agent_id=agent.id,
                change_callback=_push_worker_state
            )

    def _setup_socket(self):
        """Set up a socket_manager with defined callbacks"""
        self.socket_manager = SocketManager(
            self.server_url,
            self.port,
            self._on_alive,
            self._on_new_message,
            self._on_socket_dead,
            self.task_group_id
        )

    def _on_alive(self, pkt):
        """Update MTurkManager's state when a worker sends an
        alive packet. This asks the socket manager to open a new channel and
        then handles ensuring the worker state is consistent
        """
        shared_utils.print_and_log(
            logging.DEBUG,
            'on_agent_alive: {}'.format(pkt)
        )
        worker_id = pkt.data['worker_id']
        hit_id = pkt.data['hit_id']
        assign_id = pkt.data['assignment_id']
        conversation_id = pkt.data['conversation_id']
        # Open a channel if it doesn't already exist
        self.socket_manager.open_channel(worker_id, assign_id)

        if not worker_id in self.mturk_workers:
            # First time this worker has connected, start tracking
            self.mturk_workers[worker_id] = WorkerState(worker_id)

        # Update state of worker based on this connect
        curr_worker_state = self._get_worker(worker_id)

        if not assign_id:
            # invalid assignment_id is an auto-fail
            shared_utils.print_and_log(
                logging.WARN,
                'Agent ({}) with no assign_id called alive'.format(worker_id)
            )
        elif not assign_id in curr_worker_state.agents:
            # First time this worker has connected under this assignment, init
            # new agent if we are still accepting workers
            if self.accepting_workers:
                convs = curr_worker_state.active_conversation_count()
                allowed_convs = self.opt['allowed_conversations']
                if allowed_convs == 0 or convs < allowed_convs:
                    agent = self._create_agent(hit_id, assign_id, worker_id)
                    curr_worker_state.add_agent(assign_id, agent)
                    self._onboard_new_worker(agent)
                else:
                    text = ('You can participate in only {} of these HITs at '
                           'once. Please return this HIT and finish your '
                           'existing HITs before accepting more.'.format(
                                allowed_convs
                           ))
                    self.force_expire_hit(worker_id, assign_id, text)
            else:
                self.force_expire_hit(worker_id, assign_id)
        else:
            agent = curr_worker_state.agents[assign_id]
            agent.log_reconnect()
            if agent.state.status == AssignState.STATUS_NONE:
                # Reconnecting before even being given a world. The retries
                # for switching to the onboarding world should catch this
                return
            elif (agent.state.status == AssignState.STATUS_ONBOARDING or
                  agent.state.status == AssignState.STATUS_WAITING):
                # Reconnecting to the onboarding world or to a waiting world
                # should either restore state or expire (if workers are no
                # longer being accepted for this task)
                if not self.accepting_workers:
                    self.force_expire_hit(worker_id, assign_id)
                elif not conversation_id:
                    self._restore_worker_state(worker_id, assign_id)
            elif agent.state.status == AssignState.STATUS_IN_TASK:
                # Reconnecting to the onboarding world or to a task world
                # should resend the messages already in the conversation
                if not conversation_id:
                    self._restore_worker_state(worker_id, assign_id)
            elif agent.state.status == AssignState.STATUS_ASSIGNED:
                # Connect after a switch to a task world, mark the switch
                agent.state.status = AssignState.STATUS_IN_TASK
                agent.state.last_command = None
                agent.state.messages = []
            elif (agent.state.status == AssignState.STATUS_DISCONNECT or
                  agent.state.status == AssignState.STATUS_DONE or
                  agent.state.status == AssignState.STATUS_EXPIRED or
                  agent.state.status == AssignState.STATUS_RETURNED or
                  agent.state.status == AssignState.STATUS_PARTNER_DISCONNECT):
                # inform the connecting user in all of these cases that the
                # task is no longer workable, use appropriate message
                data = agent.get_inactive_command_data()
                self.send_command(worker_id, assign_id, data)

    def _on_new_message(self, pkt):
        """Put an incoming message onto the correct agent's message queue and
        add it to the proper message thread as long as the agent is active
        """
        worker_id = pkt.sender_id
        assignment_id = pkt.assignment_id
        agent = self._get_agent(worker_id, assignment_id)
        if agent is None:
            self._log_missing_agent(worker_id, assignment_id)
        elif not agent.state.is_final():
            shared_utils.print_and_log(
                logging.INFO,
                'Manager received: {}'.format(pkt),
                should_print=self.opt['verbose']
            )
            # Push the message to the message thread to send on a reconnect
            agent.state.messages.append(pkt.data)

            # Clear the send message command, as a message was recieved
            agent.state.last_command = None
            # TODO ensure you can't duplicate a message push here
            agent.msg_queue.put(pkt.data)

    def _on_socket_dead(self, worker_id, assignment_id):
        """Handle a disconnect event, update state as required and notifying
        other agents if the disconnected agent was in conversation with them

        returns False if the socket death should be ignored and the socket
        should stay open and not be considered disconnected
        """
        agent = self._get_agent(worker_id, assignment_id)
        if agent is None:
            # This worker never registered, so we don't do anything
            return

        shared_utils.print_and_log(
            logging.DEBUG,
            'Worker {} disconnected from {} in status {}'.format(
                worker_id,
                assignment_id,
                agent.state.status
            )
        )

        if agent.state.status == AssignState.STATUS_NONE:
            # Agent never made it to onboarding, delete
            agent.state.status = AssignState.STATUS_DISCONNECT
            agent.reduce_state()
        elif agent.state.status == AssignState.STATUS_ONBOARDING:
            # Agent never made it to task pool, the onboarding thread will die
            # and delete the agent if we mark it as a disconnect
            agent.state.status = AssignState.STATUS_DISCONNECT
            agent.disconnected = True
        elif agent.state.status == AssignState.STATUS_WAITING:
            # agent is in pool, remove from pool and delete
            if agent in self.worker_pool:
                with self.worker_pool_change_condition:
                    self.worker_pool.remove(agent)
            agent.state.status = AssignState.STATUS_DISCONNECT
            agent.reduce_state()
        elif agent.state.status == AssignState.STATUS_IN_TASK:
            self._handle_worker_disconnect(worker_id, assignment_id)
            agent.disconnected = True
        elif agent.state.status == AssignState.STATUS_DONE:
            # It's okay if a complete assignment socket dies, but wait for the
            # world to clean up the resource
            return
        elif agent.state.status == AssignState.STATUS_ASSIGNED:
            # mark the agent in the assigned state as disconnected, the task
            # spawn thread is responsible for cleanup
            agent.state.status = AssignState.STATUS_DISCONNECT
            agent.disconnected = True

        self.socket_manager.close_channel(agent.get_connection_id())

    def _create_agent(self, hit_id, assignment_id, worker_id):
        """Initialize an agent and return it"""
        return MTurkAgent(self.opt, self, hit_id, assignment_id, worker_id)


    def _onboard_new_worker(self, mturk_agent):
        """Handle creating an onboarding thread and moving an agent through
        the onboarding process, updating the state properly along the way
        """
        # get state variable in question
        worker_id = mturk_agent.worker_id
        assignment_id = mturk_agent.assignment_id

        def _onboard_function(mturk_agent):
            """Onboarding wrapper to set state to onboarding properly"""
            if self.onboard_function:
                conversation_id = 'o_'+str(uuid.uuid4())
                mturk_agent.change_conversation(
                    conversation_id=conversation_id,
                    agent_id='onboarding',
                    change_callback=self._set_worker_status_to_onboard
                )
                # Wait for turker to be in onboarding status
                mturk_agent.wait_for_status(AssignState.STATUS_ONBOARDING)
                # call onboarding function
                self.onboard_function(mturk_agent)

            # once onboarding is done, move into a waiting world
            self._move_workers_to_waiting([mturk_agent])

        if not assignment_id in self.assignment_to_onboard_thread:
            # Start the onboarding thread and run it
            onboard_thread = threading.Thread(
                target=_onboard_function,
                args=(mturk_agent,),
                name='onboard-{}-{}'.format(worker_id, assignment_id)
            )
            onboard_thread.daemon = True
            onboard_thread.start()

            self.assignment_to_onboard_thread[assignment_id] = onboard_thread

    def _assign_agent_to_conversation(self, agent, conv_id):
        """Register an agent object with a conversation id, update status"""
        if agent.state.status != AssignState.STATUS_IN_TASK:
            # Avoid on a second ack if alive already came through
            agent.state.status = AssignState.STATUS_ASSIGNED
            self.socket_manager.delay_heartbeat_until(
                agent.get_connection_id(),
                time.time() + HEARTBEAT_DELAY_TIME
            )

        agent.conversation_id = conv_id
        if not conv_id in self.conv_to_agent:
            self.conv_to_agent[conv_id] = []
        self.conv_to_agent[conv_id].append(agent)

    def _no_workers_incomplete(self, workers):
        """Return True if all the given workers completed their task"""
        for w in workers:
            if w.state.is_final() and w.state.status != \
                    AssignState.STATUS_DONE:
                return False
        return True

    def _get_worker(self, worker_id):
        """A safe way to get a worker by worker_id"""
        if worker_id in self.mturk_workers:
            return self.mturk_workers[worker_id]
        return None

    def _get_agent(self, worker_id, assignment_id):
        """A safe way to get an agent by worker and assignment_id"""
        worker = self._get_worker(worker_id)
        if worker is not None:
            if assignment_id in worker.agents:
                return worker.agents[assignment_id]
        return None

    def _log_missing_agent(self, worker_id, assignment_id):
        """Logs when an agent was expected to exist, yet for some reason it
        didn't. If these happen often there is a problem"""
        shared_utils.print_and_log(
            logging.WARN,
            'Expected to have an agent for {}_{}, yet none was found'.format(
                worker_id,
                assignment_id
            )
        )

    ### Manager Lifecycle Functions ###

    def setup_server(self, task_directory_path=None):
        """Prepare the MTurk server for the new HIT we would like to submit"""
        fin_word = 'start'
        if self.opt['count_complete']:
            fin_word = 'finish'
        shared_utils.print_and_log(
            logging.INFO,
            '\nYou are going to allow workers from Amazon Mechanical Turk to '
            'be an agent in ParlAI.\nDuring this process, Internet connection '
            'is required, and you should turn off your computer\'s auto-sleep '
            'feature.\nEnough HITs will be created to fulfill {} times the '
            'number of conversations requested, extra HITs will be expired '
            'once the desired conversations {}.'.format(HIT_MULT, fin_word),
            should_print=True
        )
        key_input = input('Please press Enter to continue... ')
        shared_utils.print_and_log(logging.NOTSET, '', True)

        mturk_utils.setup_aws_credentials()

        # See if there's enough money in the account to fund the HITs requested
        num_assignments = self.required_hits
        payment_opt = {
            'type': 'reward',
            'num_total_assignments': num_assignments,
            'reward': self.opt['reward'],  # in dollars
            'unique': self.opt['unique_worker']
        }
        total_cost = mturk_utils.calculate_mturk_cost(payment_opt=payment_opt)
        if not mturk_utils.check_mturk_balance(
                balance_needed=total_cost,
                is_sandbox=self.opt['is_sandbox']):
            raise SystemExit('Insufficient funds')

        if ((not self.opt['is_sandbox']) and
                (total_cost > 100 or self.opt['reward'] > 1)):
            confirm_string = '$%.2f' % total_cost
            expected_cost = total_cost / HIT_MULT
            expected_string = '$%.2f' % expected_cost
            shared_utils.print_and_log(
                logging.INFO,
                'You are going to create {} HITs at {} per assignment, for a '
                'total cost up to {} after MTurk fees. Please enter "{}" to '
                'confirm and continue, and anything else to cancel.\nNote that'
                ' of the {}, the target amount to spend is {}.'.format(
                    self.required_hits,
                    '$%.2f' % self.opt['reward'],
                    confirm_string,
                    confirm_string,
                    confirm_string,
                    expected_string
                ),
                should_print=True
            )
            check = input('Enter here: ')
            if (check != confirm_string and ('$' + check) != confirm_string):
                raise SystemExit('Cancelling')

        shared_utils.print_and_log(logging.INFO, 'Setting up MTurk server...',
                                   should_print=True)
        mturk_utils.create_hit_config(
            task_description=self.opt['task_description'],
            unique_worker=self.opt['unique_worker'],
            is_sandbox=self.opt['is_sandbox']
        )
        # Poplulate files to copy over to the server
        if not self.task_files_to_copy:
            self.task_files_to_copy = []
        if not task_directory_path:
            task_directory_path = os.path.join(
                self.opt['parlai_home'],
                'parlai',
                'mturk',
                'tasks',
                self.opt['task']
            )
        self.task_files_to_copy.append(
            os.path.join(task_directory_path, 'html', 'cover_page.html'))
        for mturk_agent_id in self.mturk_agent_ids + ['onboarding']:
            self.task_files_to_copy.append(os.path.join(
                task_directory_path,
                'html',
                '{}_index.html'.format(mturk_agent_id)
            ))

        # Setup the server with a likely-unique app-name
        task_name = '{}-{}'.format(str(uuid.uuid4())[:8], self.opt['task'])
        self.server_task_name = \
            ''.join(e for e in task_name if e.isalnum() or e == '-')
        self.server_url = server_utils.setup_server(self.server_task_name,
                                                    self.task_files_to_copy)
        shared_utils.print_and_log(logging.INFO, self.server_url)

        shared_utils.print_and_log(logging.INFO, "MTurk server setup done.\n",
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

    def set_onboard_function(self, onboard_function):
        self.onboard_function = onboard_function

    def start_task(self, eligibility_function, assign_role_function,
                   task_function):
        """Handle running a task by checking to see when enough agents are
        in the pool to start an instance of the task. Continue doing this
        until the desired number of conversations is had.
        """

        def _task_function(opt, workers, conversation_id):
            """Wait for all workers to join world before running the task"""
            shared_utils.print_and_log(
                logging.INFO,
                'Starting task {}...'.format(conversation_id)
            )
            shared_utils.print_and_log(
                logging.DEBUG,
                'Waiting for all workers to join the conversation...'
            )
            start_time = time.time()
            while True:
                all_joined = True
                for worker in workers:
                    # check the status of an individual worker assignment
                    if worker.state.status != AssignState.STATUS_IN_TASK:
                        all_joined = False
                if all_joined:
                    break
                if time.time() - start_time > WORLD_START_TIMEOUT:
                    # We waited but not all workers rejoined, throw workers
                    # back into the waiting pool. Stragglers will disconnect
                    # from there
                    shared_utils.print_and_log(
                        logging.INFO,
                        'Timeout waiting for {}, move back to waiting'.format(
                            conversation_id
                        )
                    )
                    self._move_workers_to_waiting(workers)
                    return
                time.sleep(shared_utils.THREAD_SHORT_SLEEP)

            shared_utils.print_and_log(
                logging.INFO,
                'All workers joined the conversation {}!'.format(
                    conversation_id
                )
            )
            self.started_conversations += 1
            task_function(mturk_manager=self, opt=opt, workers=workers)
            # Delete extra state data that is now unneeded
            for worker in workers:
                worker.state.clear_messages()

            # Count if it's a completed conversation
            if self._no_workers_incomplete(workers):
                self.completed_conversations += 1

        while True:
            # Loop forever starting task worlds until desired convos are had
            with self.worker_pool_change_condition:
                valid_workers = self._get_unique_pool(eligibility_function)
                needed_workers = len(self.mturk_agent_ids)
                if len(valid_workers) >= needed_workers:
                    # enough workers in pool to start new conversation
                    self.conversation_index += 1
                    new_conversation_id = \
                        't_{}'.format(self.conversation_index)

                    # Add the required number of valid workers to the conv
                    workers = [w for w in valid_workers[:needed_workers]]
                    assign_role_function(workers)
                    for w in workers:
                        w.change_conversation(
                            conversation_id=new_conversation_id,
                            agent_id=w.id,
                            change_callback=self._change_worker_to_conv
                        )
                        # Remove selected workers from the pool
                        self.worker_pool.remove(w)

                    # Start a new thread for this task world
                    task_thread = threading.Thread(
                        target=_task_function,
                        args=(self.opt, workers, new_conversation_id),
                        name='task-{}'.format(new_conversation_id)
                    )
                    task_thread.daemon = True
                    task_thread.start()
                    self.task_threads.append(task_thread)

            # Once we've had enough conversations, finish and break
            compare_count = self.started_conversations
            if (self.opt['count_complete']):
                compare_count = self.completed_conversations
            if compare_count == self.num_conversations:
                self.accepting_workers = False
                self.expire_all_unassigned_hits()
                self._expire_onboarding_pool()
                self._expire_worker_pool()
                # Wait for all conversations to finish, then break from
                # the while loop
                for thread in self.task_threads:
                    thread.join()
                break
            time.sleep(shared_utils.THREAD_MEDIUM_SLEEP)

    def shutdown(self):
        """Handle any mturk client shutdown cleanup."""
        # Ensure all threads are cleaned and state and HITs are handled
        self.expire_all_unassigned_hits()
        self._expire_onboarding_pool()
        self._expire_worker_pool()
        self.socket_manager.close_all_channels()
        for assignment_id in self.assignment_to_onboard_thread:
            self.assignment_to_onboard_thread[assignment_id].join()
        self._save_disconnects()
        server_utils.delete_server(self.server_task_name)

    ### MTurk Agent Interaction Functions ###

    def force_expire_hit(self, worker_id, assign_id, text=None, ack_func=None):
        """Send a command to expire a hit to the provided agent, update State
        to reflect that the HIT is now expired
        """
        # Expire in the state
        is_final = True
        agent = self._get_agent(worker_id, assign_id)
        if agent is not None:
            if not agent.state.is_final():
                is_final = False
                agent.state.status = AssignState.STATUS_EXPIRED
                agent.hit_is_expired = True

        # Send the expiration command
        if text == None:
            text = ('This HIT is expired, please return and take a new '
                    'one if you\'d want to work on this task.')
        data = {'text': data_model.COMMAND_EXPIRE_HIT, 'inactive_text': text}
        self.send_command(worker_id, assign_id, data, ack_func=ack_func)

    def handle_turker_timeout(self, worker_id, assign_id):
        """To be used by the MTurk agent when the worker doesn't send a message
        within the expected window.
        """
        # Expire the hit for the disconnected user
        text = ('You haven\'t entered a message in too long, leaving the other'
                ' participant unable to complete the HIT. Thus this hit has '
                'been expired and you have been considered disconnected. '
                'Disconnect too frequently and you will be blocked from '
                'working on these HITs in the future.')
        self.force_expire_hit(worker_id, assign_id, text)

        # Send the disconnect event to all workers in the convo
        self._handle_worker_disconnect(worker_id, assign_id)

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

    def send_command(self, receiver_id, assignment_id, data, blocking=True,
                     ack_func=None):
        """Sends a command through the socket manager,
        update conversation state
        """
        data['type'] = data_model.MESSAGE_TYPE_COMMAND
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

        agent = self._get_agent(receiver_id, assignment_id)
        if (data['text'] != data_model.COMMAND_CHANGE_CONVERSATION and
            data['text'] != data_model.COMMAND_RESTORE_STATE and
            agent is not None):
            # Append last command, as it might be necessary to restore state
            agent.state.last_command = packet.data

        self.socket_manager.queue_packet(packet)

    def mark_workers_done(self, workers):
        """Mark a group of workers as done to keep state consistent"""
        for worker in workers:
            if not worker.state.is_final():
                worker.state.status = AssignState.STATUS_DONE

    def free_workers(self, workers):
        """End completed worker threads"""
        for worker in workers:
            self.socket_manager.close_channel(worker.get_connection_id())


    ### Amazon MTurk Server Functions ###

    def get_agent_work_status(self, assignment_id):
        """Get the current status of an assignment's work"""
        client = mturk_utils.get_mturk_client(self.is_sandbox)
        try:
            response = client.get_assignment(AssignmentId=assignment_id)
            return response['Assignment']['AssignmentStatus']
        except ClientError as e:
            # If the assignment isn't done, asking for the assignment will fail
            not_done_message = ('This operation can be called with a status '
                                'of: Reviewable,Approved,Rejected')
            if not_done_message in e.response['Error']['Message']:
                return MTurkAgent.ASSIGNMENT_NOT_DONE

    def create_additional_hits(self, num_hits):
        """Handle creation for a specific number of hits/assignments
        Put created HIT ids into the hit_id_list
        """
        shared_utils.print_and_log(logging.INFO,
                                   'Creating {} hits...'.format(num_hits))
        hit_type_id = mturk_utils.create_hit_type(
            hit_title=self.opt['hit_title'],
            hit_description='{} (ID: {})'.format(self.opt['hit_description'],
                                                 self.task_group_id),
            hit_keywords=self.opt['hit_keywords'],
            hit_reward=self.opt['reward'],
            assignment_duration_in_seconds= # Set to 30 minutes by default
                self.opt.get('assignment_duration_in_seconds', 30 * 60),
            is_sandbox=self.opt['is_sandbox']
        )
        mturk_chat_url = '{}/chat_index?task_group_id={}'.format(
            self.server_url,
            self.task_group_id
        )
        shared_utils.print_and_log(logging.INFO, mturk_chat_url)
        mturk_page_url = None

        if self.opt['unique_worker'] == True:
            # Use a single hit with many assignments to allow
            # workers to only work on the task once
            mturk_page_url, hit_id = mturk_utils.create_hit_with_hit_type(
                page_url=mturk_chat_url,
                hit_type_id=hit_type_id,
                num_assignments=num_hits,
                is_sandbox=self.is_sandbox
            )
            self.hit_id_list.append(hit_id)
        else:
            # Create unique hits, allowing one worker to be able to handle many
            # tasks without needing to be unique
            for i in range(num_hits):
                mturk_page_url, hit_id = mturk_utils.create_hit_with_hit_type(
                    page_url=mturk_chat_url,
                    hit_type_id=hit_type_id,
                    num_assignments=1,
                    is_sandbox=self.is_sandbox
                )
                self.hit_id_list.append(hit_id)
        return mturk_page_url

    def create_hits(self):
        """Create hits based on the managers current config, return hit url"""
        shared_utils.print_and_log(logging.INFO, 'Creating HITs...', True)

        mturk_page_url = self.create_additional_hits(
            num_hits=self.required_hits
        )

        shared_utils.print_and_log(logging.INFO,
                                   'Link to HIT: {}\n'.format(mturk_page_url),
                                   should_print=True)
        shared_utils.print_and_log(
            logging.INFO,
            'Waiting for Turkers to respond... (Please don\'t close'
            ' your laptop or put your computer into sleep or standby mode.)\n',
            should_print=True
        )
        return mturk_page_url

    def get_hit(self, hit_id):
        """Get hit from mturk by hit_id"""
        client = mturk_utils.get_mturk_client(self.is_sandbox)
        return client.get_hit(HITId=hit_id)

    def get_assignment(self, assignment_id):
        """Gets assignment from mturk by assignment_id. Only works if the
        assignment is in a completed state
        """
        client = mturk_utils.get_mturk_client(self.is_sandbox)
        return client.get_assignment(AssignmentId=assignment_id)

    def expire_all_unassigned_hits(self):
        """Move through the whole hit_id list and attempt to expire the
        HITs, though this only immediately expires those that aren't assigned.
        """
        shared_utils.print_and_log(logging.INFO,
                                   'Expiring all unassigned HITs...',
                                   should_print=not self.is_test)
        for hit_id in self.hit_id_list:
            mturk_utils.expire_hit(self.is_sandbox, hit_id)

    def approve_work(self, assignment_id):
        """approve work for a given assignment through the mturk client"""
        client = mturk_utils.get_mturk_client(self.is_sandbox)
        client.approve_assignment(AssignmentId=assignment_id)

    def reject_work(self, assignment_id, reason):
        """reject work for a given assignment through the mturk client"""
        client = mturk_utils.get_mturk_client(self.is_sandbox)
        client.reject_assignment(
            AssignmentId=assignment_id,
            RequesterFeedback=reason
        )

    def block_worker(self, worker_id, reason):
        """Block a worker by id using the mturk client, passes reason along"""
        client = mturk_utils.get_mturk_client(self.is_sandbox)
        client.create_worker_block(WorkerId=worker_id, Reason=reason)

    def pay_bonus(self, worker_id, bonus_amount, assignment_id, reason,
                  unique_request_token):
        """Handles paying bonus to a turker, fails for insufficient funds.
        Returns True on success and False on failure
        """
        total_cost = mturk_utils.calculate_mturk_cost(
            payment_opt={'type': 'bonus', 'amount': bonus_amount}
        )
        if not mturk_utils.check_mturk_balance(balance_needed=total_cost,
                                               is_sandbox=self.is_sandbox):
            shared_utils.print_and_log(
                logging.WARN,
                'Cannot pay bonus. Reason: Insufficient '
                'funds in your MTurk account.',
                should_print=True
            )
            return False

        client = mturk_utils.get_mturk_client(self.is_sandbox)
        # unique_request_token may be useful for handling future network errors
        client.send_bonus(
            WorkerId=worker_id,
            BonusAmount=str(bonus_amount),
            AssignmentId=assignment_id,
            Reason=reason,
            UniqueRequestToken=unique_request_token
        )
        shared_utils.print_and_log(
            logging.INFO,
            'Paid ${} bonus to WorkerId: {}'.format(
                bonus_amount,
                worker_id
            )
        )
        return True

    def email_worker(self, worker_id, subject, message_text):
        """Send an email to a worker through the mturk client"""
        client = mturk_utils.get_mturk_client(self.is_sandbox)
        response = client.notify_workers(
            Subject=subject,
            MessageText=message_text,
            WorkerIds=[worker_id]
        )
        if len(response['NotifyWorkersFailureStatuses']) > 0:
            failure_message = response['NotifyWorkersFailureStatuses'][0]
            return {'failure': failure_message['NotifyWorkersFailureMessage']}
        else:
            return {'success': True}
