# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.worlds import display_messages

import os
import time
from datetime import datetime
import random
import string
import json
import uuid
from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.server_utils import setup_server, create_hit_config
from parlai.mturk.core.mturk_utils import calculate_mturk_cost, check_mturk_balance, create_hit_type, create_hit_with_hit_type, get_mturk_client, setup_aws_credentials
import threading
from parlai.mturk.core.data_model import COMMAND_SEND_MESSAGE, COMMAND_SHOW_DONE_BUTTON, COMMAND_EXPIRE_HIT, COMMAND_SUBMIT_HIT, COMMAND_CHANGE_CONVERSATION
from botocore.exceptions import ClientError
import uuid
from socketIO_client_nexus import SocketIO
from queue import Queue, PriorityQueue, Empty
import webbrowser
import requests
import logging
import math

from collections import namedtuple

# TODO move timeouts to constants up here

STATUS_INIT = 0
STATUS_SENT = 1
STATUS_ACK = 2

TYPE_ACK = 'ack'
TYPE_ALIVE = 'alive'
TYPE_MESSAGE = 'message'
TYPE_HEARTBEAT = 'heartbeat'

ACK_TIME = {'alive': 2,
            'message': 2}

ASSIGNMENT_NOT_DONE = 'NotDone'
ASSIGNMENT_DONE = 'Submitted'
ASSIGNMENT_APPROVED = 'Approved'
ASSIGNMENT_REJECTED = 'Rejected'

ASSIGN_STATUS_NONE = 0
ASSIGN_STATUS_ONBOARDING = 1
ASSIGN_STATUS_WAITING = 2
ASSIGN_STATUS_ASSIGNED = 3
ASSIGN_STATUS_IN_TASK = 4
ASSIGN_STATUS_DONE = 5

MTURK_DISCONNECT_MESSAGE = '[DISCONNECT]' # some Turker disconnected from conversation
TIMEOUT_MESSAGE = '[TIMEOUT]' # the Turker did not respond, but didn't return the HIT
RETURN_MESSAGE = '[RETURNED]' # the Turker returned the HIT

INVALID_TASK_RETURN = '[INVALID]'

DEF_SOCKET_TIMEOUT = 8

logging_enabled = True
logger = None
debug = True

if logging_enabled:
    logging.basicConfig(filename=str(time.time())+'.log',
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger('mturk')

def print_and_log(message, should_print=True):
    if logging_enabled:
        logger.info(message)
    if should_print or debug: # Always print message in debug mode
        print(message)

class AssignState():
    """Class for holding state information about an assignment currentyly
    claimed by an agent"""
    def __init__(self, assignment_id, status=ASSIGN_STATUS_NONE,
                 conversation_id=None):
        self.assignment_id = assignment_id
        self.status = status
        self.conversation_id = conversation_id


class WorkerState():
    """Class for holding state information about an mturk worker"""
    def __init__(self, worker_id, disconnects=0):
        self.worker_id = worker_id
        self.assignments = {}
        self.disconnects = disconnects


class Packet():
    """Class for holding information sent over a socket"""
    def __init__(self, id, type, sender_id, receiver_id, assignment_id, data,
                 requires_ack=True, blocking=True, ack_func=None):
        self.id = id
        self.type = type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.assignment_id = assignment_id
        self.data = data
        self.requires_ack = requires_ack
        self.blocking = blocking
        self.ack_func = ack_func
        self.status = STATUS_INIT
        self.time = None


class SocketManager():
    """SocketManager is a wrapper around socketIO to stabilize its packet
    passing. The manager handles resending packet, as well as maintaining alive
    status for all the connections it forms"""


    def __init__(self, server_url, port, alive_callback, message_callback,
                 socket_dead_callback, socket_dead_timeout=DEF_SOCKET_TIMEOUT):
        """
        server_url:           url at which the server is to be run
        port:                 port for the socket to operate on
        alive_callback:       function to be called on alive Packets, defined
                              def alive_callback(self, pkt)
        message_callback:     function to be called on message Packets, defined
                              def message_callback(self, pkt)
        socket_dead_callback: function to be called when a socket dies, defined
                              def on_socket_dead(self, worker_id, assignment_id)
        socket_dead_timeout:  time to wait between heartbeats before dying
        """
        self.server_url = server_url
        self.port = port
        self.alive_callback = alive_callback
        self.message_callback = message_callback
        self.socket_dead_callback = socket_dead_callback
        self.socket_dead_timeout = socket_dead_timeout

        self.socketIO = None

        # initialize the state
        self.listen_thread = None
        self.queues = {}
        self.threads = {}
        self.run = {}
        self.last_heartbeat = {}
        self.packet_map = {}

        # setup the socket
        self.setup_socket()


    def setup_socket(self):
        """Creates socket handlers and registers the socket"""
        self.socketIO = SocketIO(self.server_url, self.port)

        def on_socket_open(*args):
            """Registers world with the passthrough server"""
            print_and_log("Socket open: " + str(args), False)
            self.socketIO.emit('agent alive',
                               {'id': 'WORLD_ALIVE',
                                'sender_id': '[World]'})

        def on_disconnect(*args):
            print_and_log("World server disconnected: " + str(args), False)
            # TODO handle world cleanup? Kill socket?

        def on_message(*args):
            """Incoming message handler for ACKs, ALIVEs, HEARTBEATs,
            and MESSAGEs"""
            packet = args[0]
            packet_id = packet['id']
            packet_type = packet['type']
            sender_id = packet['sender_id']
            receiver_id = packet['receiver_id']
            assignment_id = packet['assignment_id']
            conversation_id = packet['conversation_id']
            connection_id = sender_id + '_' + assignment_id
            if packet_type == TYPE_ACK:
                # Acknowledgements should mark a packet as acknowledged
                print_and_log("On new ack: " + str(args), False)
                self.packet_map[packet_id].status = STATUS_ACK
                # If the packet sender wanted to do something on acknowledge
                if self.packet_map[packet_id].ack_func:
                    self.packet_map[packet_id].ack_func(packet)
            elif packet_type == TYPE_HEARTBEAT:
                # Heartbeats update the last heartbeat time and respond in kind
                self.last_heartbeat[connection_id] = time.time()
                # TODO Is response heartbeat necessary?
                packet = {
                    'id': packet_id,
                    'type': TYPE_HEARTBEAT,
                    'sender_id': receiver_id,
                    'receiver_id': sender_id,
                    'assignment_id': assignment_id,
                    'conversation_id': conversation_id,
                    'data': ''
                }
                self.socketIO.emit('route packet', packet, None)
            else:
                # Remaining packet types need to be acknowledged
                print_and_log("On new message: " + str(args), False)
                ack = {
                    'id': packet_id,
                    'type': TYPE_ACK,
                    'sender_id': receiver_id,
                    'receiver_id': sender_id,
                    'assignment_id': assignment_id,
                    'conversation_id': conversation_id,
                    'data': ''
                }
                self.socketIO.emit('route packet', ack, None)
                if packet_type == TYPE_ALIVE:
                    self.last_heartbeat[connection_id] = time.time()
                    self.alive_callback(packet)
                elif packet_type == TYPE_MESSAGE:
                    self.message_callback(packet)

        # Register Handlers
        # TODO move these magic strings into constants
        self.socketIO.on('socket_open', on_socket_open)
        self.socketIO.on('disconnect', on_disconnect)
        self.socketIO.on('new packet', on_message)

        # Start listening thread
        self.listen_thread = threading.Thread(target=self.socketIO.wait)
        self.listen_thread.daemon = True
        self.listen_thread.start()


    def open_channel(self, worker_id, assignment_id):
        """Opens a channel for a worker on a given assignment, doesn't re-open
        if the channel is already open"""
        connection_id = worker_id + '_' + assignment_id
        if connection_id in self.queues:
            print_and_log('Channel (' + connection_id + ') already open', False)
            return
        self.queues[connection_id] = PriorityQueue()
        self.run[connection_id] = True

        def channel_thread():
            """Handler thread for monitoring a single channel"""
            # while the thread is still alive
            while self.run[connection_id]:
                try:
                    # Check if client is still alive
                    if time.time() - self.last_heartbeat[connection_id] \
                            > self.socket_dead_timeout:
                        self.socket_dead_callback(worker_id, assignment_id)
                        break

                    # Get first item in the queue, check if we can send it yet
                    item = self.queues[connection_id].get(block=False)
                    t = item[0]
                    if time.time() < t:
                        # Put the item back into the queue, it's not time to pop yet
                        self.queues[connection_id].put(item)
                    else:
                        # Try to send the packet
                        packet = item[1]
                        if packet.status is not STATUS_ACK:
                            # either need to send initial packet
                            # or resend not-acked packet
                            pkt = {
                                'id': packet.id,
                                'type': packet.type,
                                'sender_id': packet.sender_id,
                                'receiver_id': packet.receiver_id,
                                'assignment_id': packet.assignment_id,
                                'data': packet.data
                            }

                            # send the packet
                            print_and_log("Send packet: " + str(packet.data))
                            def set_status_to_sent(data):
                                packet.status = STATUS_SENT
                            self.socketIO.emit(
                                'route packet',
                                pkt,
                                set_status_to_sent
                            )

                            if packet.requires_ack:
                                if packet.blocking:
                                    # blocking till ack is received or timeout
                                    start_t = time.time()
                                    while True:
                                        if packet.status == STATUS_ACK:
                                            break
                                        if time.time() - start_t \
                                                > ACK_TIME[packet.type]:
                                            # didn't receive ACK, resend packet
                                            # keep old queue time to ensure this
                                            # packet is processed first
                                            packet.status = STATUS_INIT
                                            self.queues[connection_id].put(item)
                                            break
                                        time.sleep(0.1)
                                else:
                                    # non-blocking ack: add ack-check to queue
                                    t = time.time() + ACK_TIME[packet.type]
                                    self.queues[connection_id].put((t, packet))
                except Empty:
                    pass
                finally:
                    time.sleep(0.2)

        # Setup and run the channel sending thread
        self.threads[connection_id] = threading.Thread(target=channel_thread)
        self.threads[connection_id].daemon = True
        self.threads[connection_id].start()


    def close_channel_internal(self, connection_id):
        """Closes a channel by connection_id"""
        print_and_log("Closing channel " + connection_id, False)
        self.run[connection_id] = False
        del self.queues[connection_id]
        del self.threads[connection_id]


    def close_channel(self, worker_id, assignment_id):
        """Closes a channel by worker_id and assignment_id"""
        self.close_channel_internal(worker_id + '_' + assignment_id)


    def close_all_channels(self):
        """Closes a channel by clearing the list of channels"""
        print_and_log("Closing all channels")
        connection_ids = list(self.queues.keys())
        for connection_id in connection_ids:
            self.close_channel_internal(connection_id)


    def generate_event_id(self, worker_id):
        """Creates a unique id to use for identifying a packet"""
        return worker_id + '_' + str(uuid.uuid4())


    def socket_is_open(self, connection_id):
        return connection_id in self.queues


    def send_packet(self, packet):
        """Queues sending a packet to its intended owner"""
        connection_id = packet.receiver_id + '_' + packet.assignment_id
        if not self.socket_is_open(connection_id):
            # Warn if there is no socket to send through for the expected recip
            print_and_log('Can not send packet to worker_id ' + \
                '{}: packet queue not found. Message: {}'.format(
                    connection_id, packet.data))
            return
        print_and_log('Put packet (' + packet.id + ') in queue (' + \
            connection_id + ')', False)
        # Get the current time to put packet into the priority queue
        self.packet_map[packet.id] = packet
        item = (time.time(), packet)
        self.queues[connection_id].put(item)


    def get_status(self, packet_id):
        """Returns the status of a particular packet by id"""
        return self.packet_map[packet_id].status


class MTurkManager():

    def __init__(self, opt, mturk_agent_ids):
        # TODO clean up these members
        self.opt = opt
        self.server_url = None
        self.port = 443
        self.task_group_id = None
        self.run_id = None
        self.mturk_agent_ids = mturk_agent_ids
        self.mturk_agents = {}
        self.agent_to_world = {}
        self.hit_id_list = []
        self.task_files_to_copy = None
        self.is_sandbox = opt['is_sandbox']
        self.worker_pool = []
        self.worker_pool_change_condition = threading.Condition()
        self.worker_index = 0
        self.assignment_to_onboard_thread = {}
        self.onboard_function = None
        self.task_threads = []
        self.conversation_index = 0
        self.num_completed_conversations = 0
        self.worker_state = {}
        self.socket_manager = None
        self.conv_to_agent = {}

    # TODO clean up this function
    def setup_server(self, task_directory_path=None):
        print_and_log("\nYou are going to allow workers from Amazon Mechanical Turk to be an agent in ParlAI.\nDuring this process, Internet connection is required, and you should turn off your computer's auto-sleep feature.\n")
        key_input = input("Please press Enter to continue... ")
        print_and_log("")

        setup_aws_credentials()

        payment_opt = {
            'type': 'reward',
            'num_total_assignments': self.opt['num_conversations'] * len(self.mturk_agent_ids),
            'reward': self.opt['reward']  # in dollars
        }
        total_cost = calculate_mturk_cost(payment_opt=payment_opt)
        if not check_mturk_balance(balance_needed=total_cost, is_sandbox=self.opt['is_sandbox']):
            return

        print_and_log('Setting up MTurk server...')
        create_hit_config(
            task_description=self.opt['task_description'],
            unique_worker=self.opt['unique_worker'],
            is_sandbox=self.opt['is_sandbox']
        )
        if not self.task_files_to_copy:
            self.task_files_to_copy = []
        if not task_directory_path:
            task_directory_path = os.path.join(self.opt['parlai_home'], 'parlai', 'mturk', 'tasks', self.opt['task'])
        self.task_files_to_copy.append(os.path.join(task_directory_path, 'html', 'cover_page.html'))
        for mturk_agent_id in self.mturk_agent_ids:
            self.task_files_to_copy.append(os.path.join(task_directory_path, 'html', mturk_agent_id+'_index.html'))
        self.server_url, db_host = setup_server(task_files_to_copy = self.task_files_to_copy)
        print_and_log(self.server_url, False)

        # print_and_log('RDS: Cleaning database...')
        # params = {
        #     'db_host': db_host
        # }
        # response = requests.get(self.server_url+'/clean_database', params=params)
        # assert(response.status_code != '200')

        print_and_log("MTurk server setup done.\n")


    def ready_to_accept_workers(self):
        """ Sets up socket to start communicating to workers"""
        print_and_log('Local: Setting up SocketIO...')
        self.setup_socket()


    def start_new_run(self):
        """Clears state to prepare for a new run"""
        self.run_id = str(int(time.time()))
        self.task_group_id = str(self.opt['task']) + '_' + str(self.run_id)

        # Reset state
        # TODO more cleanup, kill things before just clearing
        self.mturk_agents = {}
        self.worker_index = 0
        self.assignment_to_onboard_thread = {}
        self.conversation_index = 0
        self.hit_id_list = []
        self.worker_pool = []
        self.task_threads = []
        self.conversation_index = 0
        self.worker_state = {}


    def set_onboard_function(self, onboard_function):
        self.onboard_function = onboard_function


    def get_ids_from_pkt(self, pkt):
        """Wrapper to get sender, assignment, and conv ids from a packet"""
        return pkt['sender_id'], pkt['assignment_id'], pkt['conversation_id']


    def _change_worker_to_conversation(self, pkt):
        """Callback to update a worker to a new conversation"""
        worker_id, assignment_id, conversation_id = self.get_ids_from_pkt(pkt)
        self.assign_agent_to_conversation(
            self.mturk_agents[worker_id][assignment_id],
            conversation_id
        )


    def _set_status_to_onboard(self, pkt):
        """Callback for changing conversations to onboarding"""
        worker_id, assignment_id, conversation_id = self.get_ids_from_pkt(pkt)
        assign_state = self.worker_state[worker_id].assignments[assignment_id]
        assign_state.status = ASSIGN_STATUS_ONBOARDING
        assign_state.conversation_id = conversation_id


    def _set_status_to_waiting(self, pkt):
        """Callback for changing conversations to waiting pool"""
        worker_id, assignment_id, conversation_id = self.get_ids_from_pkt(pkt)
        assign_state = self.worker_state[worker_id].assignments[assignment_id]
        assign_state.status = ASSIGN_STATUS_WAITING
        assign_state.conversation_id = conversation_id


    # TODO clean up this function
    def onboard_new_worker(self, mturk_agent):
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
                    change_callback=self._set_status_to_onboard
                )
                while True:
                    # TODO refactor this wait into a helper function
                    # Wait for turker to be in onboarding status
                    if assign_state.status == ASSIGN_STATUS_ONBOARDING:
                        break
                    time.sleep(0.1)
                # call onboarding function
                self.onboard_function(mturk_agent)

            # once onboarding is done, move into a waiting world
            conversation_id = 'w_'+str(uuid.uuid4())
            mturk_agent.change_conversation(
                conversation_id=conversation_id,
                agent_id='waiting',
                change_callback=self._set_status_to_waiting
            )
            while True:
                # Wait for turker to be in waiting status
                if assign_state.status == ASSIGN_STATUS_WAITING:
                    break
                time.sleep(0.1)

            with self.worker_pool_change_condition:
                if not mturk_agent.hit_is_returned:
                    print("Adding worker to pool...")
                    self.worker_pool.append(mturk_agent)

        if not mturk_agent.assignment_id in self.assignment_to_onboard_thread:
            onboard_thread = threading.Thread(target=_onboard_function, args=(mturk_agent,))
            onboard_thread.daemon = True
            onboard_thread.start()
            self.assignment_to_onboard_thread[mturk_agent.assignment_id] = onboard_thread


    def get_unique_pool(self, eligibility_function):
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


    # TODO clean up this function
    def start_task(self, eligibility_function, role_function, task_function):
        """Handles running a task by checking to see when enough agents are in
        the pool to start an instance of the task. It continues doing this until
        the desired number of conversations is had."""

        def _task_function(opt, workers, conversation_id):
            """waits for all workers to join world before running the task"""
            print("Starting task...")
            print("Waiting for all workers to join the conversation...")
            while True:
                # TODO Add timeout to return people to the queue if not everyone
                # connects in time, requires conversation_id to check
                all_joined = True
                for worker in workers:
                    # check the status of an individual worker assignment
                    worker_id = worker.worker_id
                    assign_id = worker.assignment_id
                    worker_state = self.worker_state[worker_id]
                    if not assign_id in worker_state.assignments:
                        # This assignment was removed, we should exit this loop
                        print("At least one worker dropped before all joined!")
                        return
                    status = worker_state.assignments[assign_id].status
                    if status != ASSIGN_STATUS_IN_TASK:
                        all_joined = False
                if all_joined:
                    break
                time.sleep(0.1)

            print("All workers joined the conversation!")
            task_function(mturk_manager=self, opt=opt, workers=workers)

        while True:
            # Loop forever starting task worlds until desired convos are had
            with self.worker_pool_change_condition:
                valid_workers = self.get_unique_pool(eligibility_function)
                if len(valid_workers) >= len(self.mturk_agent_ids):
                    # enough workers in pool to start new conversation
                    self.conversation_index += 1
                    new_conversation_id = 't_' + str(self.conversation_index)

                    selected_workers = []
                    for worker in valid_workers:
                        if not worker.hit_is_returned and eligibility_function(worker):
                            selected_workers.append(worker)
                            worker.id = role_function(worker)
                            # TODO suspend checking alives for threads that are
                            # switching to a task conversations
                            worker.change_conversation(
                                conversation_id=new_conversation_id,
                                agent_id=worker.id,
                                change_callback=self._change_worker_to_conversation
                            )

                    # Remove selected workers from the pool
                    for worker in selected_workers:
                        self.worker_pool.remove(worker)

                    task_thread = threading.Thread(target=_task_function, args=(self.opt, selected_workers, new_conversation_id))
                    task_thread.daemon = True
                    task_thread.start()
                    self.task_threads.append(task_thread)

                    if self.conversation_index == self.opt['num_conversations']:
                        self.expire_all_unassigned_hits()

                        # Wait for all conversations to finish, then break from the while loop
                        for thread in self.task_threads:
                            thread.join()
                        break
            time.sleep(0.3)


    def setup_socket(self):
        """Sets up a socket_manager with defined callbacks"""
        self.socket_manager = SocketManager(self.server_url, self.port,
                                            self.on_alive, self.on_new_message,
                                            self.on_socket_dead)


    def on_alive(self, pkt):
        """Handler for updating MTurkManager's state when a worker sends an
        alive packet. This asks the socket manager to open a new channel and
        then handles ensuring the worker state is consistent"""
        print_and_log("on_agent_alive: " + str(pkt), False)
        worker_id = pkt['data']['worker_id']
        hit_id = pkt['data']['hit_id']
        assign_id = pkt['data']['assignment_id']
        conversation_id = pkt['data']['conversation_id']
        # Open a channel if it doesn't already exist
        self.socket_manager.open_channel(worker_id, assign_id)

        if not worker_id in self.worker_state:
            # First time this worker has connected, start tracking
            self.worker_state[worker_id] = WorkerState(worker_id)

        # Update state of worker based on this connect
        curr_worker_assign = self.worker_state[worker_id].assignments

        if conversation_id and not curr_worker_assign:
            # This was a request from a previous run and should be expired
            # TODO send packet to turker noting that their hit is expired
            return
        if not assign_id:
            # invalid assignment_id is an auto-fail
            print_and_log('Agent (' + worker_id + ') with no assign_id ' + \
                'called alive', False)
            return
        if not assign_id in curr_worker_assign:
            # First time this worker has connected under this assignment, init
            curr_worker_assign[assign_id] = AssignState(assign_id)
            self.create_agent(hit_id, assign_id, worker_id)
            self.onboard_new_worker(self.mturk_agents[worker_id][assign_id])
        elif curr_worker_assign[assign_id].status == ASSIGN_STATUS_NONE:
            # Invalid reconnect
            print_and_log('Agent (' + worker_id + ') with invalid status ' + \
                'none called alive', False)
            # TODO handle reconnecting and moving to onboarding
        elif curr_worker_assign[assign_id].status == ASSIGN_STATUS_ASSIGNED:
            # Connect after a switch to a task world
            curr_worker_assign[assign_id].status = ASSIGN_STATUS_IN_TASK
        elif curr_worker_assign[assign_id].status == ASSIGN_STATUS_ONBOARDING:
            # reconnecting to onboarding world is a no-op
            return
        else:
            # Reconnecting while already in a world, check to see if still alive
            print_and_log('Agent (' + worker_id + ') had unexpected reconnect',
                False)
            # TODO handle reconnect logic for workers who send a new alive


    def on_new_message(self, pkt):
        """Put an incoming message onto the correct agent's message queue"""
        worker_id = pkt['sender_id']
        assignment_id = pkt['assignment_id']
        self.mturk_agents[worker_id][assignment_id].msg_queue.put(pkt['data'])


    def on_socket_dead(self, worker_id, assignment_id):
        """Handles a disconnect event, updating state as required and notifying
        other agents if the disconnected agent was in conversation with them"""
        print_and_log("Worker {} disconnected from assignment {}".format(
            worker_id, assignment_id))
        self.worker_state[worker_id].disconnects += 1
        # TODO Block worker if disconnects exceed some amount

        agent = self.mturk_agents[worker_id][assignment_id]
        assignments = self.worker_state[worker_id].assignments
        status = assignments[assignment_id].status
        if status == ASSIGN_STATUS_NONE:
            # Agent never made it to onboarding, delete
            del assignments[assignment_id]
            del agent
        elif status == ASSIGN_STATUS_ONBOARDING:
            # Agent never made it to task pool, delete
            del assignments[assignment_id]
            del agent
            # TODO kill onboarding world's thread
        elif status == ASSIGN_STATUS_WAITING:
            # agent is in pool, remove from pool and delete
            if agent in self.worker_pool:
                with self.worker_pool_change_condition:
                    self.worker_pool.remove(agent)
            del assignments[assignment_id]
            del agent
        elif status == ASSIGN_STATUS_IN_TASK:
            # in conversation, inform world about disconnect
            conversation_id = assignments[assignment_id].conversation_id
            if agent in self.conv_to_agent[conversation_id]:
                for other_agent in self.conv_to_agent[conversation_id]:
                    if agent.id != other_agent.id:
                        # TODO this should be handled more cleanly
                        data = {
                            'text': 'COMMAND_DISCONNECT_PARTNER',
                            'disconnect_text': 'One of the other agents ' + \
                                               'unexpectedly disconnected.',
                            'conversation_id': conversation_id,
                            'agent_id': other_agent.id
                        }
                        self.send_command(
                            '[World]',
                            worker_id,
                            assignment_id,
                            data
                        )
                    other_agent.some_agent_disconnected = True
                    # TODO logic to delete these assignments from other workers
        elif status == ASSIGN_STATUS_DONE:
            # It's okay if a complete assignment dies, but wait for the world
            # to clean up the socket
            return
        else:
            # A disconnect shouldn't happen in the "Assigned" state, as we don't
            # check alive status when reconnecting after given an assignment
            print_and_log("Disconnect had invalid status " + str(status))

        # TODO Attempt to notify worker they have disconnected before the below
        # close the sending thread
        self.socket_manager.close_channel(worker_id, assignment_id)


    def send_through_socket(self, sender_id, receiver_id, assignment_id, data,
                            blocking, content_type, ack_func):
        """Wrapper for pushing through socket"""
        data['type'] = content_type
        event_id = self.socket_manager.generate_event_id(receiver_id)
        packet = Packet(
            event_id,
            TYPE_MESSAGE,
            sender_id,
            receiver_id,
            assignment_id,
            data,
            blocking=blocking,
            ack_func=ack_func
        )
        self.socket_manager.send_packet(packet)


    def send_message(self, sender_id, receiver_id, assignment_id, data,
                     blocking=True, ack_func=None):
        """Sends a message through the socket manager"""
        # TODO move content type into constant, redo constants
        self.send_through_socket(
            sender_id,
            receiver_id,
            assignment_id,
            data,
            blocking,
            'MESSAGE',
            ack_func
        )


    def send_command(self, sender_id, receiver_id, assignment_id, data,
                     blocking=True, ack_func=None):
        """Sends a command through the socket manager"""
        # TODO move content type into constant, redo constants
        self.send_through_socket(
            sender_id,
            receiver_id,
            assignment_id,
            data,
            blocking,
            'COMMAND',
            ack_func
        )


    def create_agent(self, hit_id, assignment_id, worker_id):
        """Initializes an agent and adds it to the map"""
        agent = MTurkAgent(self.opt, self, hit_id, assignment_id, worker_id)
        if (worker_id in self.mturk_agents):
            self.mturk_agents[worker_id][assignment_id] = agent
        else:
            self.mturk_agents[worker_id] = {}
            self.mturk_agents[worker_id][assignment_id] = agent


    def assign_agent_to_conversation(self, agent, conv_id):
        """Registers an agent object with a conversation id, updates status"""
        worker_id = agent.worker_id
        assignment_id = agent.assignment_id
        print("ASSIGNING " + worker_id + " assign " + assignment_id)
        assign_state = self.worker_state[worker_id].assignments[assignment_id]
        if assign_state.status == ASSIGN_STATUS_WAITING and 't_' in conv_id:
            # An agent didn't acknowledge the conversation change before
            # refreshing, so we didn't put them in assigned before this call
            assign_state.status = ASSIGN_STATUS_IN_TASK
            print_and_log("Worker reconnected in waiting")
        elif assign_state.status != ASSIGN_STATUS_IN_TASK:
            # Avoid on a second ack if alive already came through
            assign_state.status = ASSIGN_STATUS_ASSIGNED

        assign_state.conversation_id = conv_id
        if not conv_id in self.conv_to_agent:
            self.conv_to_agent[conv_id] = []
        self.conv_to_agent[conv_id].append(agent)


    def remove_agent_from_conversation(self, agent, conv_id):
        """Deregisters an agent object from a conversation id"""
        # TODO handle dealing with state changes
        if not conv_id in self.conv_to_agent:
            return
        if not agent in self.conv_to_agent[conv_id]:
            return
        self.conv_to_agent[conv_id].delete(agent)


    def get_agent_work_status(self, assignment_id):
        """Gets the current status of an assignment's work"""
        client = get_mturk_client(self.is_sandbox)
        try:
            response = client.get_assignment(AssignmentId=assignment_id)
            return response['Assignment']['AssignmentStatus']
        except ClientError as e:
            # If the assignment isn't done, asking for the assignment will fail
            not_done_message = 'This operation can be called with a status ' + \
                                'of: Reviewable,Approved,Rejected'
            if not_done_message in e.response['Error']['Message']:
                return ASSIGNMENT_NOT_DONE


    def create_additional_hits(self, num_hits):
        """Helper to handle creation for a specific number of hits/assignments
        Puts created HIT ids into the hit_id_list
        """
        print_and_log('Creating '+str(num_hits)+' hits...', False)
        hit_type_id = create_hit_type(
            hit_title=self.opt['hit_title'],
            hit_description=self.opt['hit_description'] + \
                            ' (ID: ' + self.task_group_id + ')',
            hit_keywords=self.opt['hit_keywords'],
            hit_reward=self.opt['reward'],
            assignment_duration_in_seconds= # Set to 30 minutes by default
                self.opt.get('assignment_duration_in_seconds', 30 * 60),
            is_sandbox=self.opt['is_sandbox']
        )
        mturk_chat_url = self.server_url + "/chat_index?task_group_id=" + \
            str(self.task_group_id)
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
            num_hits=self.opt['num_conversations'] * len(self.mturk_agent_ids)
        )

        print_and_log("Link to HIT: " + mturk_page_url + "\n")
        print_and_log("Waiting for Turkers to respond... (Please don't close" +\
            " your laptop or put your computer into sleep or standby mode.)\n")
        # if self.opt['is_sandbox']:
        #     webbrowser.open(mturk_page_url)
        return mturk_page_url


    def expire_hit(self, hit_id):
        """Expires given HIT
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
            print_and_log("Cannot pay bonus. Reason: Insufficient funds" + \
                          " in your MTurk account.")
            return False

        client = get_mturk_client(self.is_sandbox)
        # unique_request_token may be useful for handling future network errors
        client.send_bonus(
            WorkerId=worker_id,
            BonusAmount=bonus_amount,
            AssignmentId=assignment_id,
            Reason=reason,
            UniqueRequestToken=unique_request_token
        )

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


    def mark_workers_done(self, workers):
        """Mark a group of workers as done to keep state consistent"""
        for worker in workers:
            worker_id = worker.worker_id
            assign_id = worker.assignment_id
            state = self.worker_state[worker_id].assignments[assign_id]
            state.status = ASSIGN_STATUS_DONE


    def free_workers(self, workers):
        """end completed worker threads and update state"""
        for worker in workers:
            worker_id = worker.worker_id
            assign_id = worker.assignment_id
            self.socket_manager.close_channel(worker_id, assign_id)
            del self.worker_state[worker_id].assignments[assign_id]



    def shutdown(self):
        """Handles any mturk client shutdown cleanup."""
        # TODO save worker state (disconnects to local db)
        pass # Current implementation has no cleanup


# TODO clean up this class
class MTurkAgent(Agent):
    """Class for an MTurkAgent that can act in a ParlAI world"""
    def __init__(self, opt, manager, hit_id, assignment_id, worker_id):
        super().__init__(opt)

        self.conversation_id = None
        self.manager = manager
        self.id = None
        self.assignment_id = assignment_id
        self.hit_id = hit_id
        self.worker_id = worker_id
        self.some_agent_disconnected = False
        self.hit_is_abandoned = False
        self.hit_is_accepted = False # state from Amazon MTurk system
        self.hit_is_returned = False # state from Amazon MTurk system

        self.msg_queue = Queue()
        #
        # self.check_hit_status_thread = threading.Thread(target=self._check_hit_status)
        # self.check_hit_status_thread.daemon = True
        # self.check_hit_status_thread.start()

    def _check_hit_status(self):
        # Check if HIT is returned
        while True:
            if self.hit_id:
                response = self.manager.get_hit(hit_id=self.hit_id)
                if response['HIT']['NumberOfAssignmentsPending'] == 1: # Amazon MTurk system acknowledges that the HIT is accepted
                    print_and_log('Worker has accepted the HIT (acknowledged by MTurk API).', False)
                    self.hit_is_accepted = True
                    break
            time.sleep(5) # ThrottlingException might happen if we poll too frequently
        while True:
            if self.hit_id:
                response = self.manager.get_hit(hit_id=self.hit_id)
                if response['HIT']['NumberOfAssignmentsAvailable'] == 1: # HIT is returned
                    self.hit_is_returned = True
                    # If the worker is still in onboarding, then we don't need to expire the HIT.
                    # If the worker is already in a conversation, then we should expire the HIT to keep the total number of available HITs consistent with the number of conversations left.
                    if self.is_in_task():
                        print_and_log('Worker has returned the HIT. Since the worker is already in a task conversation, we are expiring the HIT.', False)
                        self.manager.expire_hit(hit_id=self.hit_id)
                    else:
                        print_and_log('Worker has returned the HIT. Since the worker is still in onboarding, we will not expire the HIT.', False)
                    return # we will not be using this MTurkAgent object for other worker, so no need to check its status anymore
            time.sleep(5) # ThrottlingException might happen if we poll too frequently

    def is_in_task(self):
        if self.conversation_id:
            return 't_' in self.conversation_id
        return False

    def observe(self, msg):
        self.manager.send_message('[World]', self.worker_id, self.assignment_id, msg)

    def act(self, timeout=None): # Timeout in seconds, after which the HIT will be expired automatically
        self.manager.send_command('[World]', self.worker_id, self.assignment_id, {'text': 'COMMAND_SEND_MESSAGE'})

        if timeout:
            start_time = time.time()

        # Wait for agent's new message
        while True:
            # Check if Turker sends a message
            if not self.msg_queue.empty():
                return self.msg_queue.get()

            if self.some_agent_disconnected:
                print("SOME AGENT DISCONNECTED")
                msg = {
                    'id': self.id,
                    'text': MTURK_DISCONNECT_MESSAGE,
                    'episode_done': True
                }
                return msg

            # Check if the Turker already returned the HIT
            if self.hit_is_returned:
                msg = {
                    'id': self.id,
                    'text': RETURN_MESSAGE,
                    'episode_done': True
                }
                return msg

            # Check if the Turker waited too long to respond
            if timeout:
                current_time = time.time()
                if (current_time - start_time) > timeout:
                    print_and_log(self.id+' is timeout.', False)
                    self.set_hit_is_abandoned()
                    msg = {
                        'id': self.id,
                        'text': TIMEOUT_MESSAGE,
                        'episode_done': True
                    }
                    return msg
            time.sleep(0.1)

    def change_conversation(self, conversation_id, agent_id, change_callback):
        """Handles changing a conversation for an agent, takes a callback for
        when the command is acknowledged"""
        data = {
            'text': 'COMMAND_CHANGE_CONVERSATION',
            'conversation_id': conversation_id,
            'agent_id': agent_id
        }
        self.manager.send_command(
            '[World]',
            self.worker_id,
            self.assignment_id,
            data,
            ack_func=change_callback
        )


    def episode_done(self):
        # TODO provide documentation for what this is supposed to be used for
        return False


    def approve_work(self):
        if self.hit_is_abandoned:
            print_and_log('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is abandoned and thus not available for review.')
        else:
            if self.manager.get_agent_work_status(assignment_id=self.assignment_id) == ASSIGNMENT_DONE:
                self.manager.approve_work(assignment_id=self.assignment_id)
                print_and_log('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is approved.')
            else:
                print_and_log("Cannot approve HIT. Reason: Turker hasn't completed the HIT yet.")

    def reject_work(self, reason='unspecified'):
        if self.hit_is_abandoned:
            print_and_log('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is abandoned and thus not available for review.')
        else:
            if self.manager.get_agent_work_status(assignment_id=self.assignment_id) == ASSIGNMENT_DONE:
                self.manager.reject_work(assignment_id=self.assignment_id, reason=reason)
                print_and_log('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is rejected.')
            else:
                print_and_log("Cannot reject HIT. Reason: Turker hasn't completed the HIT yet.")

    def block_worker(self, reason='unspecified'):
        self.manager.block_worker(worker_id=self.worker_id, reason=reason)
        print_and_log("Blocked worker ID: " + str(self.worker_id) + ". Reason: " + reason)

    def pay_bonus(self, bonus_amount, reason='unspecified'):
        if self.hit_is_abandoned:
            print_and_log('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is abandoned and thus not available for bonus.')
        else:
            if self.manager.get_agent_work_status(assignment_id=self.assignment_id) != ASSIGNMENT_NOT_DONE:
                unique_request_token = str(uuid.uuid4())
                if self.manager.pay_bonus(worker_id=self.worker_id, bonus_amount=bonus_amount, assignment_id=self.assignment_id, reason=reason, unique_request_token=unique_request_token):
                    print_and_log("Paid $" + str(bonus_amount) + " bonus to WorkerId: " + self.worker_id)
            else:
                print_and_log("Cannot pay bonus for HIT. Reason: Turker hasn't completed the HIT yet.")

    def email_worker(self, subject, message_text):
        response = self.manager.email_worker(worker_id=self.worker_id, subject=subject, message_text=message_text)
        if 'success' in response:
            print_and_log("Email sent to worker ID: "+str(self.worker_id)+": Subject: "+str(subject)+": Text: "+str(message_text))
            return True
        elif 'failure' in response:
            print_and_log("Unable to send email to worker ID: "+str(self.worker_id)+". Error: "+str(response['failure']))
            return False

    def set_hit_is_abandoned(self):
        if not self.hit_is_abandoned:
            self.hit_is_abandoned = True
            self.manager.send_command('[World]', self.worker_id, self.assignment_id,
                                      {'text': 'COMMAND_EXPIRE_HIT'})

    def wait_for_hit_completion(self, timeout=None): # Timeout in seconds, after which the HIT will be expired automatically
        if timeout:
            start_time = time.time()
        while self.manager.get_agent_work_status(assignment_id=self.assignment_id) != ASSIGNMENT_DONE:
            # Check if the Turker already returned the HIT
            if self.hit_is_returned:
                return False
            if timeout:
                current_time = time.time()
                if (current_time - start_time) > timeout:
                    print_and_log("Timed out waiting for Turker to complete the HIT.")
                    self.set_hit_is_abandoned()
                    return False
            print_and_log("Waiting for Turker to complete the HIT...", False)
            status = self.manager.get_agent_work_status(assignment_id=self.assignment_id)
            time.sleep(2)
        print_and_log('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is done.')
        return True

    def shutdown(self, timeout=None, direct_submit=False): # Timeout in seconds, after which the HIT will be expired automatically
        command_to_send = COMMAND_SHOW_DONE_BUTTON
        if direct_submit:
            command_to_send = COMMAND_SUBMIT_HIT
        if not (self.hit_is_abandoned or self.hit_is_returned):
            self.manager.send_command('[World]', self.worker_id, self.assignment_id,
                                      {'text': command_to_send})
            return self.wait_for_hit_completion(timeout=timeout)
