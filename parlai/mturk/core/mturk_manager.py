# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import os
import pickle
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from parlai.mturk.core.server_utils import setup_server
from parlai.mturk.core.mturk_utils import calculate_mturk_cost, \
    check_mturk_balance, create_hit_type, create_hit_with_hit_type, \
    get_mturk_client, setup_aws_credentials, create_hit_config
from parlai.mturk.core.worker_state import WorkerState, AssignState
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.agents import MTurkAgent
from parlai.mturk.core.shared_utils import print_and_log, generate_event_id, \
                                         THREAD_SHORT_SLEEP, THREAD_MEDIUM_SLEEP
import parlai.mturk.core.data_model as data_model
from botocore.exceptions import ClientError

# TODO-1 move these somewhere that makes more sense
ASSIGNMENT_NOT_DONE = 'NotDone'
ASSIGNMENT_DONE = 'Submitted'
ASSIGNMENT_APPROVED = 'Approved'
ASSIGNMENT_REJECTED = 'Rejected'

# Timeout before cancelling a world start
WORLD_START_TIMEOUT = 11

# Multiplier to apply when creating hits to ensure worker availibility
HIT_MULT = 1.5

# Max number of disconnects before a turker should be blocked
MAX_DISCONNECTS = 25

# Time to persist a disconnect before forgetting about it. Combined with the
# above this will block workers that disconnect at least 25 times in a week
DISCONNECT_PERSIST_LENGTH = 60 * 24 * 7

DISCONNECT_FILE_NAME = 'disconnects.pickle'

parent_dir = os.path.dirname(os.path.abspath(__file__))

class MTurkManager():

    def __init__(self, opt, mturk_agent_ids):
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

    ### Helpers and internal manager methods ###

    def _init_state(self):
        self.mturk_agents = {}
        self.hit_id_list = []
        self.worker_pool = []
        self.worker_index = 0
        self.assignment_to_onboard_thread = {}
        self.task_threads = []
        self.conversation_index = 0
        self.started_conversations = 0
        self.completed_conversations = 0
        self.worker_state = {}
        self.conv_to_agent = {}
        self.accepting_workers = True
        self._load_disconnects()


    def _load_disconnects(self):
        self.disconnects = []
        file_path = os.path.join(parent_dir, DISCONNECT_FILE_NAME)
        compare_time = time.time()
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                old_disconnects = pickle.load(f)
                print (old_disconnects)
                self.disconnects = [
                    d
                    for d in old_disconnects
                    if (compare_time - d['time']) < DISCONNECT_PERSIST_LENGTH
                ]
        print (self.disconnects)

    def _save_disconnects(self):
        file_path = os.path.join(parent_dir, DISCONNECT_FILE_NAME)
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self.disconnects, f, pickle.HIGHEST_PROTOCOL)


    def _get_ids_from_pkt(self, pkt):
        """Wrapper to get sender, assignment, and conv ids from a packet"""
        worker_id = pkt.sender_id
        assignment_id = pkt.assignment_id
        agent = self.mturk_agents[worker_id][assignment_id]
        conversation_id = agent.conversation_id
        return worker_id, assignment_id, conversation_id


    def _change_worker_to_conv(self, pkt):
        """Callback to update a worker to a new conversation"""
        worker_id, assignment_id, conversation_id = self._get_ids_from_pkt(pkt)
        self._assign_agent_to_conversation(
            self.mturk_agents[worker_id][assignment_id],
            conversation_id
        )


    def _set_worker_status_to_onboard(self, pkt):
        """Callback for changing conversations to onboarding"""
        worker_id, assignment_id, conversation_id = self._get_ids_from_pkt(pkt)
        assign_state = self.worker_state[worker_id].assignments[assignment_id]
        assign_state.status = AssignState.STATUS_ONBOARDING
        assign_state.conversation_id = conversation_id


    def _set_worker_status_to_waiting(self, pkt):
        """Callback for changing conversations to waiting pool"""
        worker_id, assignment_id, conversation_id = self._get_ids_from_pkt(pkt)
        assign_state = self.worker_state[worker_id].assignments[assignment_id]
        assign_state.status = AssignState.STATUS_WAITING
        assign_state.conversation_id = conversation_id

        # Wait for turker to be in waiting status
        self._wait_for_status(assign_state, AssignState.STATUS_WAITING)

        # Add the worker to pool
        with self.worker_pool_change_condition:
            print("Adding worker to pool...")
            self.worker_pool.append(self.mturk_agents[worker_id][assignment_id])


    def _move_workers_to_waiting(self, workers):
        """Puts all workers into waiting worlds, expires them if no longer
        accepting workers"""
        for worker in workers:
            worker_id = worker.worker_id
            assignment_id = worker.assignment_id
            assignment = self.worker_state[worker_id].assignments[assignment_id]
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


    def _wait_for_status(self, assign_state, desired_status):
        """Suspends a thread until a particular assignment state changes to the
        desired state"""
        while True:
            if assign_state.status == desired_status:
                break
            time.sleep(THREAD_SHORT_SLEEP)


    def _expire_onboarding_pool(self):
        """Expires any worker that is in an onboarding thread"""
        for worker_id in self.worker_state:
            for assign_id in self.worker_state[worker_id].assignments:
                assign = self.worker_state[worker_id].assignments[assign_id]
                if (assign.status == AssignState.STATUS_ONBOARDING):
                    self.force_expire_hit(worker_id, assign_id)


    def _expire_worker_pool(self):
        """Expires all workers in the worker pool"""
        for agent in self.worker_pool:
            self.force_expire_hit(agent.worker_id, agent.assignment_id)


    def _get_unique_pool(self, eligibility_function):
        """Returns a filtered version of the worker pool where each worker is
        only listed a maximum of one time. In sandbox this is overridden for
        testing purposes, and the same worker can be returned more than once"""
        workers = [w for w in self.worker_pool if
                   not w.hit_is_returned and eligibility_function(w)]
        unique_workers = []
        unique_worker_ids = []
        for w in workers:
            if (self.is_sandbox) or (w.worker_id not in unique_worker_ids):
                unique_workers.append(w)
                unique_worker_ids.append(w.worker_id)
        return unique_workers


    def _handle_partner_disconnect(self, worker_id, assignment_id):
        """Sends a message to a worker notifying them that a partner has
        disconnected and we marked the HIT as complete for them"""
        state = self.worker_state[worker_id].assignments[assignment_id]
        if not state.is_final():
            # Update the assignment state
            agent = self.mturk_agents[worker_id][assignment_id]
            agent.some_agent_disconnected = True
            state.status = AssignState.STATUS_PARTNER_DISCONNECT

            # Create and send the command
            data = {
                'text': data_model.COMMAND_DISCONNECT_PARTNER,
                'disconnect_text': ('One of the other agents '
                                    'unexpectedly disconnected.'),
            }
            self.send_command(worker_id, assignment_id, data)


    def _restore_worker_state(self, worker_id, assignment_id):
        """Sends a command to restore the state of an agent who reconnected"""
        assignment = self.worker_state[worker_id].assignments[assignment_id]
        def _push_worker_state(msg):
            if len(assignment.messages) != 0:
                data = {
                    'text': data_model.COMMAND_RESTORE_STATE,
                    'messages': assignment.messages,
                    'last_command': assignment.last_command
                }
                self.send_command(worker_id, assignment_id, data)

        agent = self.mturk_agents[worker_id][assignment_id]
        agent.change_conversation(
            conversation_id=agent.conversation_id,
            agent_id=agent.id,
            change_callback=_push_worker_state
        )


    def _setup_socket(self):
        """Sets up a socket_manager with defined callbacks"""
        self.socket_manager = SocketManager(
            self.server_url,
            self.port,
            self._on_alive,
            self._on_new_message,
            self._on_socket_dead,
            self.task_group_id
        )


    def _on_alive(self, pkt):
        """Handler for updating MTurkManager's state when a worker sends an
        alive packet. This asks the socket manager to open a new channel and
        then handles ensuring the worker state is consistent"""
        print_and_log('on_agent_alive: {}'.format(pkt), False)
        worker_id = pkt.data['worker_id']
        hit_id = pkt.data['hit_id']
        assign_id = pkt.data['assignment_id']
        conversation_id = pkt.data['conversation_id']
        # Open a channel if it doesn't already exist
        self.socket_manager.open_channel(worker_id, assign_id)

        if not worker_id in self.worker_state:
            # First time this worker has connected, start tracking
            self.worker_state[worker_id] = WorkerState(worker_id)

        # Update state of worker based on this connect
        curr_worker_state = self.worker_state[worker_id]

        if conversation_id and not curr_worker_state:
            # This was a request from a previous run and should be expired,
            # send a message and expire when it is acknowledged
            def _close_my_socket(data):
                """Small helper to close the socket after user acknowledges that
                it shouldn't exist"""
                self.socket_manager.close_channel(worker_id, assign_id)

            text = ('You disconnected in the middle of this HIT and the '
                    'HIT expired before you reconnected. It is no longer '
                    'available for completion. Please return this HIT and '
                    'accept a new one if you would like to try again.')
            self.force_expire_hit(worker_id, assign_id, text, _close_my_socket)
            return
        elif not assign_id:
            # invalid assignment_id is an auto-fail
            print_and_log('Agent ({}) with no assign_id called alive'.format(
                worker_id
            ), False)
            return
        elif not assign_id in curr_worker_state.assignments:
            # First time this worker has connected under this assignment, init
            # if we are still accepting workers
            if self.accepting_workers:
                convs = self.worker_state[worker_id].active_conversation_count()
                allowed_convs = self.opt['allowed_conversations']
                if allowed_convs == 0 or convs < allowed_convs:
                    curr_worker_state.add_assignment(assign_id)
                    self._create_agent(hit_id, assign_id, worker_id)
                    self._onboard_new_worker(
                        self.mturk_agents[worker_id][assign_id]
                    )
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
            curr_assign = curr_worker_state.assignments[assign_id]
            curr_assign.log_reconnect(worker_id)
            if curr_assign.status == AssignState.STATUS_NONE:
                # Reconnecting before even being given a world. The retries for
                # switching to the onboarding world should catch this
                return
            elif (curr_assign.status == AssignState.STATUS_ONBOARDING or
                  curr_assign.status == AssignState.STATUS_WAITING):
                # Reconnecting to the onboarding world or to a waiting world
                # should either restore state or expire (if workers are no
                # longer being accepted for this task)
                if not self.accepting_workers:
                    self.force_expire_hit(worker_id, assign_id)
                elif not conversation_id:
                    self._restore_worker_state(worker_id, assign_id)
            elif curr_assign.status == AssignState.STATUS_IN_TASK:
                # Reconnecting to the onboarding world or to a task world should
                # resend the messages already in the conversation
                if not conversation_id:
                    self._restore_worker_state(worker_id, assign_id)
            elif curr_assign.status == AssignState.STATUS_ASSIGNED:
                # Connect after a switch to a task world, mark the switch
                curr_assign.status = AssignState.STATUS_IN_TASK
                curr_assign.last_command = None
                curr_assign.messages = []
            elif (curr_assign.status == AssignState.STATUS_DISCONNECT or
                  curr_assign.status == AssignState.STATUS_DONE or
                  curr_assign.status == AssignState.STATUS_EXPIRED or
                  curr_assign.status == AssignState.STATUS_PARTNER_DISCONNECT or
                  curr_assign.status == AssignState.STATUS_RETURNED):
                # inform the connecting user in all of these cases that the task
                # is no longer workable, generate appropriate text for each.
                data = curr_assign.get_inactive_command_data(worker_id)
                self.send_command(worker_id, assign_id, data)


    def _on_new_message(self, pkt):
        """Put an incoming message onto the correct agent's message queue and
        add it to the proper message thread"""
        worker_id = pkt.sender_id
        assignment_id = pkt.assignment_id
        # Push the message to the message thread ready to send on a reconnect
        self.worker_state[worker_id].assignments[assignment_id].messages.append(
            pkt.data
        )
        # Clear the send message command
        self.worker_state[worker_id].assignments[assignment_id].last_command = \
            None
        self.mturk_agents[worker_id][assignment_id].msg_queue.put(pkt.data)


    def _on_socket_dead(self, worker_id, assignment_id):
        """Handles a disconnect event, updating state as required and notifying
        other agents if the disconnected agent was in conversation with them"""
        if (worker_id not in self.mturk_agents) or (assignment_id not \
                    in self.mturk_agents[worker_id]):
            # This worker never registered, so we don't do anything
            return True
        agent = self.mturk_agents[worker_id][assignment_id]
        agent.disconnected = True
        assignments = self.worker_state[worker_id].assignments
        status = assignments[assignment_id].status
        print_and_log('Worker {} disconnected from {} in status {}'.format(
            worker_id, assignment_id, status))
        self.worker_state[worker_id].disconnects += 1
        # TODO-3 Block worker if disconnects exceed some amount

        if status == AssignState.STATUS_NONE:
            # Agent never made it to onboarding, delete
            assignments[assignment_id].status = AssignState.STATUS_DISCONNECT
            del agent
        elif status == AssignState.STATUS_ONBOARDING:
            # Agent never made it to task pool, kill the onboarding thread
            assignments[assignment_id].status = AssignState.STATUS_DISCONNECT
            self.assignment_to_onboard_thread[assignment_id].terminate()
            self.disconnects.append({'time': time.time(), 'id': worker_id})
            del agent
        elif status == AssignState.STATUS_WAITING:
            # agent is in pool, remove from pool and delete
            if agent in self.worker_pool:
                with self.worker_pool_change_condition:
                    self.worker_pool.remove(agent)
            assignments[assignment_id].status = AssignState.STATUS_DISCONNECT
            self.disconnects.append({'time': time.time(), 'id': worker_id})
            del agent
        elif status == AssignState.STATUS_IN_TASK:
            # Disconnect in conversation is not workable (if its a conversation)
            assignments[assignment_id].status = AssignState.STATUS_DISCONNECT
            # in conversation, inform others about disconnect
            conversation_id = assignments[assignment_id].conversation_id
            if agent in self.conv_to_agent[conversation_id]:
                for other_agent in self.conv_to_agent[conversation_id]:
                    if agent.assignment_id != other_agent.assignment_id:
                        self._handle_partner_disconnect(
                            other_agent.worker_id,
                            other_agent.assignment_id
                        )
            self.disconnects.append({'time': time.time(), 'id': worker_id})
        elif (status == AssignState.STATUS_DONE or
              status == AssignState.STATUS_EXPIRED or
              status == AssignState.STATUS_DISCONNECT or
              status == AssignState.STATUS_PARTNER_DISCONNECT or
              status == AssignState.STATUS_RETURNED):
            # It's okay if a complete assignment socket dies, but wait for the
            # world to clean up the resource
            return True
        else:
            # A disconnect should be ignored in the "Assigned" state, as we dont
            # check alive status when reconnecting after given an assignment
            return False

        # TODO-4 Attempt to notify worker they of disconnect before the below
        # close the sending thread
        self.socket_manager.close_channel(worker_id, assignment_id)
        return True


    def _create_agent(self, hit_id, assignment_id, worker_id):
        """Initializes an agent and adds it to the map"""
        agent = MTurkAgent(self.opt, self, hit_id, assignment_id, worker_id)
        if (worker_id in self.mturk_agents):
            self.mturk_agents[worker_id][assignment_id] = agent
        else:
            self.mturk_agents[worker_id] = {}
            self.mturk_agents[worker_id][assignment_id] = agent


    def _onboard_new_worker(self, mturk_agent):
        """Handles creating an onboarding thread and moving an agent through
        the onboarding process, updating the state properly along the way"""
        # get state variable in question
        worker_id = mturk_agent.worker_id
        assignment_id = mturk_agent.assignment_id
        assign_state = self.worker_state[worker_id].assignments[assignment_id]

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
                self._wait_for_status(
                    assign_state,
                    AssignState.STATUS_ONBOARDING
                )
                # call onboarding function
                self.onboard_function(mturk_agent)

            # once onboarding is done, move into a waiting world
            self._move_workers_to_waiting([mturk_agent])

        if not assignment_id in self.assignment_to_onboard_thread:
            # Start the onboarding thread and run it
            onboard_thread = threading.Thread(target=_onboard_function,
                                              args=(mturk_agent,))
            onboard_thread.daemon = True
            onboard_thread.start()

            self.assignment_to_onboard_thread[assignment_id] = onboard_thread


    def _assign_agent_to_conversation(self, agent, conv_id):
        """Registers an agent object with a conversation id, updates status"""
        worker_id = agent.worker_id
        assignment_id = agent.assignment_id
        assign_state = self.worker_state[worker_id].assignments[assignment_id]
        if assign_state.status != AssignState.STATUS_IN_TASK:
            # Avoid on a second ack if alive already came through
            assign_state.status = AssignState.STATUS_ASSIGNED

        assign_state.conversation_id = conv_id
        if not conv_id in self.conv_to_agent:
            self.conv_to_agent[conv_id] = []
        self.conv_to_agent[conv_id].append(agent)


    def _no_workers_incomplete(self, workers):
        """Helper to determine if all the given workers completed their task"""
        for w in workers:
            state = self.worker_state[w.worker_id].assignments[w.assignment_id]
            if state.is_final() and state.status != AssignState.STATUS_DONE:
                return False
        return True


    ### Manager Lifecycle Functions ###

    def setup_server(self, task_directory_path=None):
        print_and_log('\nYou are going to allow workers from Amazon '
              'Mechanical Turk to be an agent in ParlAI.\nDuring this '
              'process, Internet connection is required, and you should '
              'turn off your computer\'s auto-sleep feature.\n')
        key_input = input('Please press Enter to continue... ')
        print_and_log('')

        setup_aws_credentials()

        # See if there's enough money in the account to fund the HITs requested
        num_assignments = self.required_hits
        payment_opt = {
            'type': 'reward',
            'num_total_assignments': num_assignments,
            'reward': self.opt['reward']  # in dollars
        }
        total_cost = calculate_mturk_cost(payment_opt=payment_opt)
        if not check_mturk_balance(balance_needed=total_cost,
                                   is_sandbox=self.opt['is_sandbox']):
            return

        print_and_log('Setting up MTurk server...')
        create_hit_config(
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
        for mturk_agent_id in self.mturk_agent_ids:
            self.task_files_to_copy.append(os.path.join(
                task_directory_path,
                'html',
                '{}_index.html'.format(mturk_agent_id)
            ))
        # Setup the server
        self.server_url = setup_server(self.task_files_to_copy)
        print_and_log(self.server_url, False)

        print_and_log("MTurk server setup done.\n")


    def ready_to_accept_workers(self):
        """ Sets up socket to start communicating to workers"""
        print_and_log('Local: Setting up SocketIO...')
        self._setup_socket()


    def start_new_run(self):
        """Clears state to prepare for a new run"""
        self.run_id = str(int(time.time()))
        self.task_group_id = '{}_{}'.format(self.opt['task'], self.run_id)
        self._init_state()


    def set_onboard_function(self, onboard_function):
        self.onboard_function = onboard_function


    def start_task(self, eligibility_function, role_function, task_function):
        """Handles running a task by checking to see when enough agents are in
        the pool to start an instance of the task. It continues doing this until
        the desired number of conversations is had."""

        def _task_function(opt, workers, conversation_id):
            """waits for all workers to join world before running the task"""
            print('Starting task...')
            print('Waiting for all workers to join the conversation...')
            start_time = time.time()
            while True:
                all_joined = True
                for worker in workers:
                    # check the status of an individual worker assignment
                    worker_id = worker.worker_id
                    assign_id = worker.assignment_id
                    worker_state = self.worker_state[worker_id]
                    if not assign_id in worker_state.assignments:
                        # This assignment was removed, we should exit this loop
                        print('At least one worker dropped before all joined!')
                        return
                    status = worker_state.assignments[assign_id].status
                    if status != AssignState.STATUS_IN_TASK:
                        all_joined = False
                if all_joined:
                    break
                if time.time() - start_time > WORLD_START_TIMEOUT:
                    # We waited but not all workers rejoined, throw workers
                    # back into the waiting pool. Stragglers will disconnect
                    # from there
                    print('Timeout waiting for workers, moving back to waiting')
                    self._move_workers_to_waiting(workers)
                    return
                time.sleep(THREAD_SHORT_SLEEP)

            print('All workers joined the conversation!')
            self.started_conversations += 1
            task_function(mturk_manager=self, opt=opt, workers=workers)
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
                    new_conversation_id = 't_{}'.format(self.conversation_index)

                    # Add the required number of valid workers to the conv
                    selected_workers = []
                    for w in valid_workers[:needed_workers]:
                        selected_workers.append(w)
                        w.id = role_function(w)
                        w.change_conversation(
                            conversation_id=new_conversation_id,
                            agent_id=w.id,
                            change_callback=self._change_worker_to_conv
                        )

                    # Remove selected workers from the pool
                    for worker in selected_workers:
                        self.worker_pool.remove(worker)

                    # Start a new thread for this task world
                    task_thread = threading.Thread(target=_task_function,
                        args=(self.opt, selected_workers, new_conversation_id))
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
            time.sleep(THREAD_MEDIUM_SLEEP)


    def shutdown(self):
        """Handles any mturk client shutdown cleanup."""
        # Ensure all threads are cleaned and handled
        self.expire_all_unassigned_hits()
        self._expire_onboarding_pool()
        self._expire_worker_pool()
        for assignment_id in self.assignment_to_onboard_thread:
            self.assignment_to_onboard_thread[assignment_id].join()
        self._save_disconnects()
        pass


    ### MTurk Agent Interaction Functions ###

    def force_expire_hit(self, worker_id, assign_id, text=None, ack_func=None):
        """Sends a command to expire a hit to the provided agent, updates State
        to reflect that the HIT is now expired"""
        # Expire in the state
        is_final = True
        if worker_id in self.worker_state:
            if assign_id in self.worker_state[worker_id].assignments:
                state = self.worker_state[worker_id].assignments[assign_id]
                if not state.is_final():
                    is_final = False
                    state.status = AssignState.STATUS_EXPIRED
        if not is_final:
            # Expire in the agent
            if worker_id in self.mturk_agents:
                if assign_id in self.mturk_agents[worker_id]:
                    agent = self.mturk_agents[worker_id][assign_id]
                    agent.hit_is_expired = True

        # Send the expiration command
        if text == None:
            text = ('This HIT is expired, please return and take a new '
                    'one if you\'d want to work on this task.')
        data = {'text': data_model.COMMAND_EXPIRE_HIT, 'inactive_text': text}
        self.send_command(worker_id, assign_id, data, ack_func=ack_func)


    def send_message(self, receiver_id, assignment_id, data,
                     blocking=True, ack_func=None):
        """Sends a message through the socket manager, updates state"""
        data['type'] = data_model.MESSAGE_TYPE_MESSAGE
        # Force messages to have a unique ID
        if 'message_id' not in data:
            data['message_id'] = str(uuid.uuid4())
        event_id = generate_event_id(receiver_id)
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
        # Push outgoing message to the message thread to be able to resend
        assignment = self.worker_state[receiver_id].assignments[assignment_id]
        assignment.messages.append(packet.data)
        self.socket_manager.queue_packet(packet)


    def send_command(self, receiver_id, assignment_id, data, blocking=True,
                     ack_func=None):
        """Sends a command through the socket manager, updates state"""
        data['type'] = data_model.MESSAGE_TYPE_COMMAND
        event_id = generate_event_id(receiver_id)
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

        if (data['text'] != data_model.COMMAND_CHANGE_CONVERSATION and
            data['text'] != data_model.COMMAND_RESTORE_STATE and
            assignment_id in self.worker_state[receiver_id].assignments):
            # Append last command, as it might be necessary to restore state
            assign = self.worker_state[receiver_id].assignments[assignment_id]
            assign.last_command = packet.data

        self.socket_manager.queue_packet(packet)


    def mark_workers_done(self, workers):
        """Mark a group of workers as done to keep state consistent"""
        for worker in workers:
            worker_id = worker.worker_id
            assign_id = worker.assignment_id
            state = self.worker_state[worker_id].assignments[assign_id]
            if not state.is_final():
                state.status = AssignState.STATUS_DONE


    def free_workers(self, workers):
        """end completed worker threads"""
        for worker in workers:
            worker_id = worker.worker_id
            assign_id = worker.assignment_id
            self.socket_manager.close_channel(worker_id, assign_id)


    ### Amazon MTurk Server Functions ###

    def get_agent_work_status(self, assignment_id):
        """Gets the current status of an assignment's work"""
        client = get_mturk_client(self.is_sandbox)
        try:
            response = client.get_assignment(AssignmentId=assignment_id)
            return response['Assignment']['AssignmentStatus']
        except ClientError as e:
            # If the assignment isn't done, asking for the assignment will fail
            not_done_message = ('This operation can be called with a status '
                                'of: Reviewable,Approved,Rejected')
            if not_done_message in e.response['Error']['Message']:
                return ASSIGNMENT_NOT_DONE


    def create_additional_hits(self, num_hits):
        """Helper to handle creation for a specific number of hits/assignments
        Puts created HIT ids into the hit_id_list
        """
        print_and_log('Creating {} hits...'.format(num_hits), False)
        hit_type_id = create_hit_type(
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
        print_and_log(mturk_chat_url, False)
        mturk_page_url = None

        if self.opt['unique_worker'] == True:
            # Use a single hit with many assignments to allow
            # workers to only work on the task once
            mturk_page_url, hit_id = create_hit_with_hit_type(
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
                mturk_page_url, hit_id = create_hit_with_hit_type(
                    page_url=mturk_chat_url,
                    hit_type_id=hit_type_id,
                    num_assignments=1,
                    is_sandbox=self.is_sandbox
                )
                self.hit_id_list.append(hit_id)
        return mturk_page_url


    def create_hits(self):
        """Creates hits based on the managers current config, returns hit url"""
        print_and_log('Creating HITs...')

        mturk_page_url = self.create_additional_hits(
            num_hits=self.required_hits
        )

        print_and_log('Link to HIT: {}\n'.format(mturk_page_url))
        print_and_log('Waiting for Turkers to respond... (Please don\'t close'
            ' your laptop or put your computer into sleep or standby mode.)\n')
        # if self.opt['is_sandbox']:
        #     webbrowser.open(mturk_page_url)
        return mturk_page_url


    def expire_hit(self, hit_id):
        """Expires given HIT from the MTurk side
        Only works if the hit is in the "pending" state
        """
        client = get_mturk_client(self.is_sandbox)
        # Update expiration to a time in the past, the HIT will expire instantly
        past_time = datetime(2015, 1, 1)
        client.update_expiration_for_hit(HITId=hit_id, ExpireAt=past_time)


    def get_hit(self, hit_id):
        """Gets hit from mturk by hit_id"""
        client = get_mturk_client(self.is_sandbox)
        return client.get_hit(HITId=hit_id)


    def get_assignment(self, assignment_id):
        """Gets hit from mturk by assignment_id"""
        client = get_mturk_client(self.is_sandbox)
        return client.get_assignment(AssignmentId=assignment_id)


    def expire_all_unassigned_hits(self):
        """Moves through the whole hit_id list and attempts to expire the hit,
        though this only immediately expires those that are pending.
        """
        print_and_log("Expiring all unassigned HITs...")
        for hit_id in self.hit_id_list:
            self.expire_hit(hit_id)


    def approve_work(self, assignment_id):
        """approves work for a given assignment through the mturk client"""
        client = get_mturk_client(self.is_sandbox)
        client.approve_assignment(AssignmentId=assignment_id)


    def reject_work(self, assignment_id, reason):
        """rejects work for a given assignment through the mturk client"""
        client = get_mturk_client(self.is_sandbox)
        client.reject_assignment(
            AssignmentId=assignment_id,
            RequesterFeedback=reason
        )


    def block_worker(self, worker_id, reason):
        """Blocks a worker by id using the mturk client, passes reason along"""
        client = get_mturk_client(self.is_sandbox)
        client.create_worker_block(WorkerId=worker_id, Reason=reason)


    def pay_bonus(self, worker_id, bonus_amount, assignment_id, reason,
                  unique_request_token):
        """Handles paying bonus to a turker, fails for insufficient funds"""
        total_cost = calculate_mturk_cost(
            payment_opt={'type': 'bonus', 'amount': bonus_amount}
        )
        if not check_mturk_balance(balance_needed=total_cost,
                                   is_sandbox=self.is_sandbox):
            print_and_log('Cannot pay bonus. Reason: Insufficient funds'
                          ' in your MTurk account.')
            return False

        client = get_mturk_client(self.is_sandbox)
        # unique_request_token may be useful for handling future network errors
        client.send_bonus(
            WorkerId=worker_id,
            BonusAmount=str(bonus_amount),
            AssignmentId=assignment_id,
            Reason=reason,
            UniqueRequestToken=unique_request_token
        )
        print_and_log('Paid ${} bonus to WorkerId: {}'.format(
            bonus_amount,
            worker_id
        ))
        return True


    def email_worker(self, worker_id, subject, message_text):
        """Send an email to a worker through the mturk client"""
        client = get_mturk_client(self.is_sandbox)
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
