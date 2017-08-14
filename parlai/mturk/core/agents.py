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

MTURK_DISCONNECT_MESSAGE = '[DISCONNECT]' # some Turker disconnected from conversation
TIMEOUT_MESSAGE = '[TIMEOUT]' # the Turker did not respond, but didn't return the HIT
RETURN_MESSAGE = '[RETURNED]' # the Turker returned the HIT

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


class Message():
    def __init__(self, id, type, sender_id, receiver_id, data, requires_ack=True, blocking=True):
        self.id = id
        self.type = type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.data = data
        self.status = STATUS_INIT
        self.blocking = blocking
        self.requires_ack = requires_ack
        self.time = None


class SocketManager():
    """SocketManager is a wrapper around socketIO to stabilize its message passing.

    The manager handles resending messages. """

    def __init__(self, server_url, port, alive_callback, message_callback, socket_dead_callback, socket_dead_timeout=4):
        self.server_url = server_url
        self.port = port
        self.alive_callback = alive_callback
        self.message_callback = message_callback
        self.socket_dead_callback = socket_dead_callback
        self.socket_dead_timeout = socket_dead_timeout
        self.queues = {}
        self.threads = {}
        self.run = {}
        self.last_heartbeat = {}

        self.message_map = {}
        self.socketIO = None
        self.listen_thread = None
        self.setup_socket()

    def setup_socket(self):
        self.socketIO = SocketIO(self.server_url, self.port)

        def on_socket_open(*args):
            print_and_log("Socket open: " + str(args), False)
            self.socketIO.emit('agent alive',
                               {'id': 'WORLD_ALIVE',
                                'sender_id': '[World]'})

        def on_disconnect(*args):
            print_and_log("Server disconnected: " + str(args), False)

        def on_message(*args):
            msg = args[0]
            id = msg['id']
            type = msg['type']
            sender_id = msg['sender_id']
            receiver_id = msg['receiver_id']
            if type == TYPE_ACK:
                print_and_log("On new ack: " + str(args), False)
                self.message_map[id].status = STATUS_ACK
            elif type == TYPE_HEARTBEAT:
                self.last_heartbeat[sender_id] = time.time()
                msg = {'id': id,
                       'type': TYPE_HEARTBEAT,
                       'sender_id': receiver_id,
                       'receiver_id': sender_id,
                       'data': ''}
                self.socketIO.emit('route message', msg, None)
            else:
                print_and_log("On new message: " + str(args), False)
                ack = {'id': id,
                       'type': TYPE_ACK,
                       'sender_id': receiver_id,
                       'receiver_id': sender_id,
                       'data': ''}
                self.socketIO.emit('route message', ack, None)
                if type == TYPE_ALIVE:
                    self.alive_callback(msg)
                elif type == TYPE_MESSAGE:
                    self.message_callback(msg)



        self.socketIO.on('socket_open', on_socket_open)
        self.socketIO.on('disconnect', on_disconnect)
        self.socketIO.on('new message', on_message)

        self.listen_thread = threading.Thread(target=self.socketIO.wait)
        self.listen_thread.daemon = True
        self.listen_thread.start()

    def open_channel(self, id):
        if id in self.queues:
            print_and_log("Channel ("+id+") already open", False)
            return
        self.queues[id] = PriorityQueue()
        self.run[id] = True

        def channel_thread():
            while self.run[id]:
                try:
                    if time.time() - self.last_heartbeat[id] > self.socket_dead_timeout:
                        self.socket_dead_callback(id)
                        break

                    item = self.queues[id].get(block=False)
                    t = item[0]
                    if time.time() > t:

                        # send message
                        message = item[1]

                        if message.status is not STATUS_ACK:
                            # either need to send initial message
                            # or resend not-acked message

                            msg = {
                                'id': message.id,
                                'type': message.type,
                                'sender_id': message.sender_id,
                                'receiver_id': message.receiver_id,
                                'data': message.data
                            }

                            print_and_log("Send message: " + str(message.data));

                            def set_status_to_sent(data):
                                message.status = STATUS_SENT
                            self.socketIO.emit('route message', msg, set_status_to_sent)

                            if message.requires_ack:
                                if message.blocking:
                                    # blocking till ack is received
                                    start_t = time.time()
                                    while True:
                                        if message.status == STATUS_ACK:
                                            break
                                        if time.time() - start_t > ACK_TIME[message.type]:
                                            # didn't receive ACK, resend message
                                            # keep old queue time to ensure this message is processed first
                                            message.status = STATUS_INIT
                                            self.queues[id].put(item)
                                            break
                                        time.sleep(0.1)
                                else:
                                    # non-blocking ack: add ack-check to queue
                                    t = time.time() + ACK_TIME[message.type]
                                    self.queues[id].put((t, message))

                    else:
                        self.queues[id].put(item)
                except Empty:
                    pass
                finally:
                    time.sleep(0.2)

        self.threads[id] = threading.Thread(target=channel_thread)
        self.threads[id].daemon = True
        self.threads[id].start()

    def close_channel(self, id):
        print_and_log("Closing channel " + id, False)
        self.run[id] = False
        del self.queues[id]
        del self.threads[id]

    def generate_event_id(self, worker_id):
        return worker_id + '_' + str(uuid.uuid4())

    def send_message(self, message):
        if not message.receiver_id in self.queues:
            raise RuntimeError('Can not send message to worker_id ' + message.receiver_id + ': message queue not found.')
        print_and_log("Put message ("+message.id+") in queue ("+message.receiver_id+")", False)
        item = (time.time(), message)
        self.queues[message.receiver_id].put(item)
        self.message_map[message.id] = message

    def get_status(self, message_id):
        return self.message_map[message_id].status


class MTurkManager():

    def __init__(self, opt, mturk_agent_ids):
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
        self.assignment_state = {}
        self.socket_manager = None


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

        print_and_log('RDS: Cleaning database...')
        params = {
            'db_host': db_host
        }
        response = requests.get(self.server_url+'/clean_database', params=params)
        assert(response.status_code != '200')

        print_and_log("MTurk server setup done.\n")

    def ready_to_accept_workers(self):
        print_and_log('Local: Setting up SocketIO...')
        self.setup_socket()

    def start_new_run(self):
        self.run_id = str(int(time.time()))
        self.task_group_id = str(self.opt['task']) + '_' + str(self.run_id)

    def set_onboard_function(self, onboard_function):
        self.onboard_function = onboard_function

    def onboard_new_worker(self, mturk_agent):
        def _onboard_function(mturk_agent):
            if self.onboard_function:
                conversation_id = 'o_'+str(uuid.uuid4())
                mturk_agent.change_conversation(conversation_id=conversation_id, agent_id='onboarding')
                while True:
                    if self.assignment_state[mturk_agent.assignment_id]['conversation_id'] == conversation_id:
                        break
                    time.sleep(0.1)
                self.onboard_function(mturk_agent)
            self.assignment_state[mturk_agent.assignment_id]['status'] = 'onboarded'

            with self.worker_pool_change_condition:
                if not mturk_agent.hit_is_returned:
                    print("Adding worker to pool...")
                    self.worker_pool.append(mturk_agent)

        if not mturk_agent.assignment_id in self.assignment_to_onboard_thread:
            onboard_thread = threading.Thread(target=_onboard_function, args=(mturk_agent,))
            onboard_thread.daemon = True
            onboard_thread.start()
            self.assignment_to_onboard_thread[mturk_agent.assignment_id] = onboard_thread

    def start_task(self, eligibility_function, role_function, task_function):
        def _task_function(opt, workers, conversation_id):
            print("Starting task...")
            print("Waiting for all workers to join the conversation...")
            while True:
                all_joined = True
                for worker in workers:
                    if self.assignment_state[worker.assignment_id]['conversation_id'] != conversation_id:
                        all_joined = False
                if all_joined:
                    break
                time.sleep(0.1)

            print("All workers joined the conversation!")
            task_function(mturk_manager=self, opt=opt, workers=workers)

        while True:
            with self.worker_pool_change_condition:
                if len([w for w in self.worker_pool if not w.hit_is_returned and eligibility_function(w)]) >= len(self.mturk_agent_ids):
                    # enough workers in pool to start new conversation
                    self.conversation_index += 1
                    new_conversation_id = 't_' + str(self.conversation_index)

                    selected_workers = []
                    for worker in self.worker_pool:
                        if not worker.hit_is_returned and eligibility_function(worker):
                            selected_workers.append(worker)
                            worker.id = role_function(worker)
                            worker.change_conversation(conversation_id=new_conversation_id, agent_id=worker.id)
                    self.worker_pool = []

                    task_thread = threading.Thread(target=_task_function, args=(self.opt, selected_workers, new_conversation_id))
                    task_thread.daemon = True
                    task_thread.start()
                    self.task_threads.append(task_thread)

                    if self.conversation_index == self.opt['num_conversations']:
                        # Wait for all conversations to finish, then break from the while loop
                        for thread in self.task_threads:
                            thread.join() 
                        break
            time.sleep(0.1)


    def setup_socket(self):
        self.socket_manager = SocketManager(self.server_url, self.port,
                                            self.on_alive, self.on_new_message, self.on_socket_dead)

    def on_alive(self, msg):
        print_and_log("on_agent_alive: " + str(msg), False)
        worker_id = msg['data']['worker_id']
        hit_id = msg['data']['hit_id']
        assignment_id = msg['data']['assignment_id']
        conversation_id = msg['data']['conversation_id']
        self.socket_manager.open_channel(assignment_id)

        if not conversation_id:
            self.assignment_state[assignment_id] = {'conversation_id': None,
                                             'status': 'init'}
            self.create_agent(hit_id, assignment_id, worker_id)
            self.onboard_new_worker(self.mturk_agents[assignment_id])
        elif assignment_id in self.assignment_state and conversation_id.startswith('o_'):
            self.assignment_state[assignment_id] = {'conversation_id': conversation_id,
                                             'status': 'onboarding'}
        elif assignment_id in self.assignment_state and conversation_id.startswith('t_'):
            self.assignment_state[assignment_id] = {'conversation_id': conversation_id,
                                             'status': 'joined_conversation'}
        else:
            print_and_log("Agent (" + worker_id + ") with invalid status tried to join", False)


    def on_new_message(self, msg):
        worker_id = msg['sender_id']
        self.mturk_agents[worker_id].msg_queue.put(msg['data'])

    def on_socket_dead(self, id):
        print_and_log("Channel {id} disconnected".format(id=id))

        # if in worker pool, remove
        agent = self.mturk_agents[id]
        if agent in self.worker_pool:
            with self.worker_pool_change_condition:
                self.worker_pool.remove(agent)

        # if in conversation, inform world about disconnect
        if id in self.agent_to_world:
            world = self.agent_to_world[id]
            for other_agent in world.agents:
                if agent.id != other_agent.id:
                    msg = {'id': 'World',
                           'text': 'COMMAND_DISCONNECT_PARTNER',
                           'disconnect_text': 'One of the other agents unexpectedly disconnected.',
                           'type': 'COMMAND'}
                    other_agent.observe(msg)
                other_agent.some_agent_disconnected = True

        # close the sending thread
        self.socket_manager.close_channel(id)

    def send_message(self, sender_id, receiver_id, data, blocking=True):
        id = self.socket_manager.generate_event_id(receiver_id)
        if 'type' not in data:
            data['type'] = 'MESSAGE'
        msg = Message(id, TYPE_MESSAGE, sender_id, receiver_id, data, blocking=blocking)
        self.socket_manager.send_message(msg)

    def send_command(self, sender_id, receiver_id, data, blocking=True):
        if 'type' not in data:
            data['type'] = 'COMMAND'
        id = self.socket_manager.generate_event_id(receiver_id)
        msg = Message(id, TYPE_MESSAGE, sender_id, receiver_id, data, blocking=blocking)
        self.socket_manager.send_message(msg)

    def create_agent(self, hit_id, assignment_id, worker_id):
        agent = MTurkAgent(self.opt, self, hit_id, assignment_id, worker_id)
        self.mturk_agents[assignment_id] = agent

    def register_agent_to_world(self, agent, world):
        self.agent_to_world[agent.assignment_id] = world

    def deregister_agent_to_world(self, agent):
        del self.agent_to_world[agent.assignment_id]

    def get_agent_work_status(self, assignment_id):
        client = get_mturk_client(self.is_sandbox)
        try:
            response = client.get_assignment(AssignmentId=assignment_id)
            return response['Assignment']['AssignmentStatus']
        except ClientError as e:
            if 'This operation can be called with a status of: Reviewable,Approved,Rejected' in e.response['Error']['Message']:
                return ASSIGNMENT_NOT_DONE

    def create_additional_hits(self, num_hits):
        print_and_log('Creating '+str(num_hits)+' hits...', False)
        hit_type_id = create_hit_type(
            hit_title=self.opt['hit_title'],
            hit_description=self.opt['hit_description'] + ' (ID: ' + self.task_group_id + ')',
            hit_keywords=self.opt['hit_keywords'],
            hit_reward=self.opt['reward'],
            assignment_duration_in_seconds=self.opt.get('assignment_duration_in_seconds', 30 * 60), # Set to 30 minutes by default
            is_sandbox=self.opt['is_sandbox']
        )
        mturk_chat_url = self.server_url + "/chat_index?task_group_id="+str(self.task_group_id)
        print_and_log(mturk_chat_url, False)
        mturk_page_url = None

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
        print_and_log('Creating HITs...')

        mturk_page_url = self.create_additional_hits(num_hits=self.opt['num_conversations'] * len(self.mturk_agent_ids))

        print_and_log("Link to HIT: " + mturk_page_url + "\n")
        print_and_log("Waiting for Turkers to respond... (Please don't close your laptop or put your computer into sleep or standby mode.)\n")
        # if self.opt['is_sandbox']:
        #     webbrowser.open(mturk_page_url)
        return mturk_page_url

    def expire_hit(self, hit_id):
        # This will only expire HITs that are in "pending" state
        client = get_mturk_client(self.is_sandbox)
        client.update_expiration_for_hit(HITId=hit_id, ExpireAt=datetime(2015, 1, 1)) # Update it to a time in the past, and the HIT will be immediately expired

    def get_hit(self, hit_id):
        client = get_mturk_client(self.is_sandbox)
        return client.get_hit(HITId=hit_id)

    def expire_all_unassigned_hits(self):
        print_and_log("Expiring all unassigned HITs...")
        for hit_id in self.hit_id_list:
            self.expire_hit(hit_id)

    def approve_work(self, assignment_id):
        client = get_mturk_client(self.is_sandbox)
        client.approve_assignment(AssignmentId=assignment_id)

    def reject_work(self, assignment_id, reason):
        client = get_mturk_client(self.is_sandbox)
        client.reject_assignment(AssignmentId=assignment_id, RequesterFeedback=reason)

    def block_worker(self, worker_id, reason):
        client = get_mturk_client(self.is_sandbox)
        client.create_worker_block(WorkerId=worker_id, Reason=reason)

    def pay_bonus(self, worker_id, bonus_amount, assignment_id, reason, unique_request_token):
        total_cost = calculate_mturk_cost(payment_opt={'type': 'bonus', 'amount': bonus_amount})
        if not check_mturk_balance(balance_needed=total_cost, is_sandbox=self.is_sandbox):
            print_and_log("Cannot pay bonus. Reason: Insufficient fund in your MTurk account.")
            return False

        client = get_mturk_client(self.is_sandbox)
        client.send_bonus(
            WorkerId=worker_id,
            BonusAmount=bonus_amount,
            AssignmentId=assignment_id,
            Reason=reason,
            UniqueRequestToken=unique_request_token # Could be useful in the future, for handling network errors
        )

        return True

    def email_worker(self, worker_id, subject, message_text):
        client = get_mturk_client(self.is_sandbox)
        response = client.notify_workers(
            Subject=subject,
            MessageText=message_text,
            WorkerIds=[worker_id]
        )
        if len(response['NotifyWorkersFailureStatuses']) > 0:
            return {'failure': response['NotifyWorkersFailureStatuses'][0]['NotifyWorkersFailureMessage']}
        else:
            return {'success': True}

    def shutdown(self):
        setup_aws_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server_utils.py')
        print_and_log("Remote database instance will accumulate cost over time (about $30/month for t2.medium instance). Please run `python "+setup_aws_file_path+" remove_rds` to remove RDS instance if you don't plan to use MTurk often.")


class MTurkAgent(Agent):
    def __init__(self, opt, manager, hit_id, assignment_id, worker_id):
        super().__init__(opt)

        self.conversation_id = None
        self.manager = manager
        self.socket_id = worker_id + "_" + hit_id
        self.id = None
        self.assignment_id = assignment_id
        self.hit_id = hit_id
        self.worker_id = worker_id
        self.some_agent_disconnected = False
        self.hit_is_abandoned = False
        self.hit_is_accepted = False # state from Amazon MTurk system
        self.hit_is_returned = False # state from Amazon MTurk system

        self.msg_queue = Queue()

        self.check_hit_status_thread = threading.Thread(target=self._check_hit_status)
        self.check_hit_status_thread.daemon = True
        self.check_hit_status_thread.start()

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
        self.manager.send_message('[World]', self.assignment_id, msg)

    def act(self, timeout=None): # Timeout in seconds, after which the HIT will be expired automatically
        self.manager.send_command('[World]', self.assignment_id, {'text': 'COMMAND_SEND_MESSAGE'})

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

    def change_conversation(self, conversation_id, agent_id):
        data = {'text': 'COMMAND_CHANGE_CONVERSATION',
                'conversation_id': conversation_id,
                'agent_id': agent_id}
        self.manager.send_command('[World]', self.assignment_id, data)
        # self.manager.socket_manager.close_channel(self.worker_id)

    def episode_done(self):
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
            self.manager.send_command('[World]', self.assignment_id,
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
            time.sleep(1)
        print_and_log('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is done.')
        return True

    def shutdown(self, timeout=None, direct_submit=False): # Timeout in seconds, after which the HIT will be expired automatically
        command_to_send = COMMAND_SHOW_DONE_BUTTON
        if direct_submit:
            command_to_send = COMMAND_SUBMIT_HIT
        if not (self.hit_is_abandoned or self.hit_is_returned):
            self.manager.send_command('[World]', self.assignment_id,
                                      {'text': command_to_send})
            return self.wait_for_hit_completion(timeout=timeout)
