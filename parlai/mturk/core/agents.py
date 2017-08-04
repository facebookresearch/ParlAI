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
from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.server_utils import setup_server, create_hit_config
from parlai.mturk.core.mturk_utils import calculate_mturk_cost, check_mturk_balance, create_hit_type, create_hit_with_hit_type, get_mturk_client, setup_aws_credentials
import threading
from parlai.mturk.core.data_model import COMMAND_SEND_MESSAGE, COMMAND_SHOW_DONE_BUTTON, COMMAND_EXPIRE_HIT, COMMAND_SUBMIT_HIT
from botocore.exceptions import ClientError
import uuid
from socketIO_client_nexus import SocketIO
import webbrowser
import requests
import logging

ASSIGNMENT_NOT_DONE = 'NotDone'
ASSIGNMENT_DONE = 'Submitted'
ASSIGNMENT_APPROVED = 'Approved'
ASSIGNMENT_REJECTED = 'Rejected'

TIMEOUT_MESSAGE = '[TIMEOUT]'

logging_enabled = True
logger = None
debug = False

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

class MTurkManager():
    def __init__(self, opt, mturk_agent_ids):
        self.server_url = None
        self.task_group_id = None
        self.socket_listen_thread = None
        self.socketIO = None
        self.run_id = None
        self.mturk_agent_ids = mturk_agent_ids
        self.mturk_agent_hit_id_dict = {}
        self.mturk_agents = {} # keep track of all mturk agents
        self.agent_to_world = {} # map agent_id to world it's involved in
        self.task_files_to_copy = None
        self.is_sandbox = opt['is_sandbox']
        self.logger = None
        self.agents_last_heartbeat = {}
        self.check_heartbeat_thread = None
        self.timeout_after = 4 # Consider client disconnected if no heartbeat is received for `timeout_after` seconds

    def init_aws(self, opt, task_directory_path=None):
        print_and_log("\nYou are going to allow workers from Amazon Mechanical Turk to be an agent in ParlAI.\nDuring this process, Internet connection is required, and you should turn off your computer's auto-sleep feature.\n")
        key_input = input("Please press Enter to continue... ")
        print_and_log("")

        setup_aws_credentials()

        payment_opt = {
            'type': 'reward',
            'num_hits': opt['num_hits'],
            'num_assignments': opt['num_assignments'],
            'reward': opt['reward']  # in dollars
        }
        total_cost = calculate_mturk_cost(payment_opt=payment_opt)
        if not check_mturk_balance(balance_needed=total_cost, is_sandbox=opt['is_sandbox']):
            return

        print_and_log('Setting up MTurk server...')
        create_hit_config(task_description=opt['task_description'], num_hits=opt['num_hits'], num_assignments=opt['num_assignments'], mturk_agent_ids=self.mturk_agent_ids, is_sandbox=opt['is_sandbox'])
        if not self.task_files_to_copy:
            self.task_files_to_copy = []
        if not task_directory_path:
            task_directory_path = os.path.join(opt['parlai_home'], 'parlai', 'mturk', 'tasks', opt['task'])
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

        print_and_log('Local: Setting up SocketIO...')
        self.setup_socket(mturk_server_url=self.server_url, port=443)

        print_and_log("MTurk server setup done.\n")

    def register_agent(self, agent):
        """Register mturk agent"""
        self.mturk_agents[agent.agent_global_id] = agent

    def deregister_agent(self, agent):
        if agent.agent_global_id in self.mturk_agents:
            del self.mturk_agents[agent.agent_global_id]

    def register_heartbeat(self, agent):
        if isinstance(agent, MTurkAgent):
            agent_global_id = agent.agent_global_id
            self.agents_last_heartbeat[agent_global_id] = time.time()

        if self.check_heartbeat_thread is None:
            self.check_heartbeat_thread = threading.Thread(target=self.check_heartbeat)
            self.check_heartbeat_thread.daemon = True
            self.check_heartbeat_thread.start()

    def deregister_heartbeat(self, agent):
        if isinstance(agent, MTurkAgent):
            if agent.agent_global_id in self.agents_last_heartbeat:
                del self.agents_last_heartbeat[agent.agent_global_id]

    def check_heartbeat(self):
        while True:
            for agent_id, t in self.agents_last_heartbeat.items():
                if self.agents_last_heartbeat[agent_id] and time.time() - self.agents_last_heartbeat[agent_id] > 4:
                    self.disconnected(agent_id)
                    break
            time.sleep(0.5)

    def disconnected(self, agent_id):
        print("DISCONNECT" + agent_id)
        agent = self.mturk_agents[agent_id]
        if agent.agent_global_id in self.agent_to_world:
            # agent is involved in world (dialog) => notify other agents about disconnect
            world = self.agent_to_world[agent.agent_global_id]
            for other_agent in world.agents:
                if other_agent.id != agent.id:
                    self.send_command_to_agent(self.task_group_id, other_agent.conversation_id, '[World]', other_agent.id, 'COMMAND_DISCONNECT_PARTNER')
                self.deregister_heartbeat(other_agent)
                self.deregister_agent(other_agent)
        else:
            self.deregister_heartbeat(agent)
            self.deregister_agent(agent)


    def register_world(self, world):
        for agent in world.agents:
            self.agent_to_world[agent.agent_global_id] = world


    def deregister_world(self, world):
        for agent in world.agents:
            self.deregister_agent(agent)


    def start_new_run(self, opt):
        self.run_id = str(int(time.time()))
        self.task_group_id = str(opt['task']) + '_' + str(self.run_id)
        
    def _send_event_to_socket(self, event_name, event_data, response_handler=None):
        event_sent = threading.Event()
        def on_event_sent(*args):
            if not event_sent.is_set():
                event_sent.set()
        emit_success = False
        while not emit_success:
            try:
                self.socketIO.emit(event_name, event_data, on_event_sent)
                emit_success = True
            except Exception as e:
                print_and_log(e, False)
                emit_success = False

        def check_event_sent():
            if event_sent.wait(timeout=1): # Timeout in seconds
                if response_handler:
                    response_handler()
            else:
                print_and_log(event_name + ' not acknowledged by server. Sending event again.', False)
                self._send_event_to_socket(event_name, event_data, response_handler)

        thread = threading.Thread(target=check_event_sent)
        thread.daemon = True
        thread.start()

    def setup_socket(self, mturk_server_url, port):
        self.socketIO = SocketIO(mturk_server_url, port)

        def on_socket_open(*args):
            print_and_log("on_socket_open: " + str(args), False)
            self._send_event_to_socket(
                'agent_alive', 
                {
                    'task_group_id': self.task_group_id,
                    'conversation_id': None,
                    'agent_id': '[World]',
                    'assignment_id': None,
                    'hit_id': None,
                    'worker_id': None
                }
            )

        def on_disconnect(*args):
            print_and_log("Server disconnected: " + str(args), False)

        def on_agent_alive(*args):
            print_and_log("on_agent_alive: " + str(args), False)
            agent_info = args[0]
            task_group_id = agent_info['task_group_id']
            conversation_id = agent_info['conversation_id']
            agent_id = agent_info['agent_id']
            assignment_id = agent_info['assignment_id']
            hit_id = agent_info['hit_id']
            worker_id = agent_info['worker_id']

            self._send_event_to_socket(
                'agent_alive_received', 
                {
                    'task_group_id': task_group_id,
                    'conversation_id': conversation_id,
                    'agent_id': '[World]'
                }
            )

            if task_group_id != self.task_group_id:
                return

            global_agent_id = MTurkAgent.get_agent_global_id(task_group_id=task_group_id,
                                                             agent_id=agent_id,
                                                             conversation_id=conversation_id)
            agent = self.mturk_agents[global_agent_id]
            agent.assignment_id = assignment_id
            agent.hit_id = hit_id
            agent.worker_id = worker_id

            agent.notify_alive()


        def on_new_message(*args):
            print_and_log("on_new_message: " + str(args), False)
            message = args[0]

            self._send_event_to_socket(
                'new_message_received', 
                {
                    'task_group_id': message['task_group_id'],
                    'conversation_id': message['conversation_id'],
                    'agent_id': '[World]'
                }
            )

            if message['task_group_id'] != self.task_group_id:
                return

            global_agent_id = MTurkAgent.get_agent_global_id(task_group_id=message['task_group_id'],
                                                             agent_id=message['sender_agent_id'],
                                                             conversation_id=message['conversation_id'])
            agent = self.mturk_agents[global_agent_id]
            with agent.new_message_lock:
                if message and message['message_id'] not in agent.received_message_ids:
                    agent.new_message = message
                    agent.received_message_ids.append(message['message_id'])

            # MTurkAgent.notify_agent_new_message(
            #     task_group_id=message['task_group_id'],
            #     conversation_id=message['conversation_id'],
            #     agent_id=message['sender_agent_id']
            # )

        def on_new_heartbeat(*args):
            print_and_log("on_new_heartbeat: " + str(args), False)
            message = args[0]

            self._send_event_to_socket(
                'new_heartbeat_received',
                {
                    'task_group_id': message['task_group_id'],
                    'conversation_id': message['conversation_id'],
                    'agent_id': '[World]'
                }
            )
            agent_global_id = MTurkAgent.get_agent_global_id(message['task_group_id'],
                                                             message['conversation_id'],
                                                             message['sender_agent_id'])
            if agent_global_id in self.agents_last_heartbeat:
                self.agents_last_heartbeat[agent_global_id] = time.time()


        self.socketIO.on('socket_open', on_socket_open)
        self.socketIO.on('agent_alive', on_agent_alive)
        self.socketIO.on('new_message', on_new_message)
        self.socketIO.on('disconnect', on_disconnect)
        self.socketIO.on('new_heartbeat', on_new_heartbeat)

        self.socket_listen_thread = threading.Thread(target=self._socket_receive_events)
        self.socket_listen_thread.daemon = True
        self.socket_listen_thread.start()
        
    def _socket_receive_events(self):
        self.socketIO.wait()

    def send_command_to_agent(self, task_group_id, conversation_id, sender_agent_id, receiver_agent_id, command):
        command_dict = {
            'task_group_id': task_group_id,
            'conversation_id': conversation_id,
            'sender_agent_id': sender_agent_id,
            'receiver_agent_id': receiver_agent_id,
            'command': command,
            'command_id': str(uuid.uuid4()),
        }

        def on_agent_send_command_response(*args):
            print_and_log("on_agent_send_command_response: "+str(args), False)

        self._send_event_to_socket(
            'agent_send_command',
            command_dict,
            on_agent_send_command_response
        )
        self.socketIO.wait_for_callbacks(seconds=0.1)

    def send_message_to_agent(self, task_group_id, conversation_id, sender_agent_id, receiver_agent_id, message):
        timestamp = None
        response = requests.get(self.server_url+'/get_timestamp')
        try:
            ret = response.json()
            timestamp = ret['timestamp']
        except Exception as e:
            print(response.content)
            raise e

        message['task_group_id'] = task_group_id
        message['conversation_id'] = conversation_id
        message['sender_agent_id'] = sender_agent_id
        message['receiver_agent_id'] = receiver_agent_id
        message['message_id'] = str(uuid.uuid4())
        message['timestamp'] = timestamp

        def on_agent_send_message_response(*args):
            print_and_log("on_agent_send_message_response: "+str(args), False)

        self._send_event_to_socket(
            'agent_send_message',
            message,
            on_agent_send_message_response
        )
        self.socketIO.wait_for_callbacks(seconds=0.1)

    def send_new_message(self, task_group_id, conversation_id, sender_agent_id, receiver_agent_id, message_text=None, reward=None, episode_done=False):
        message = {
            'id': sender_agent_id,
            'episode_done': episode_done,
        }
        if message_text:
            message['text'] = message_text
        if reward:
            message['reward'] = reward

        self.send_message_to_agent(
            task_group_id=task_group_id,
            conversation_id=conversation_id,
            sender_agent_id=sender_agent_id,
            receiver_agent_id=receiver_agent_id,
            message=message
        )

    def send_new_command(self, task_group_id, conversation_id, sender_agent_id, receiver_agent_id, command):
        self.send_command_to_agent(
            task_group_id=task_group_id,
            conversation_id=conversation_id,
            sender_agent_id=sender_agent_id,
            receiver_agent_id=receiver_agent_id,
            command=command
        )

    def get_agent_work_status(self, assignment_id):
        client = get_mturk_client(self.is_sandbox)
        try:
            response = client.get_assignment(AssignmentId=assignment_id)
            return response['Assignment']['AssignmentStatus']
        except ClientError as e:
            if 'This operation can be called with a status of: Reviewable,Approved,Rejected' in e.response['Error']['Message']:
                return ASSIGNMENT_NOT_DONE

    def create_hits(self, opt):
        print_and_log('Creating HITs...')
        self.mturk_agent_hit_id_dict = {}
        hit_type_id = create_hit_type(
            hit_title=opt['hit_title'],
            hit_description=opt['hit_description'] + ' (ID: ' + self.task_group_id + ')',
            hit_keywords=opt['hit_keywords'],
            hit_reward=opt['reward'],
            assignment_duration_in_seconds=opt.get('assignment_duration_in_seconds', 30 * 60), # Set to 30 minutes by default
            is_sandbox=opt['is_sandbox']
        )
        mturk_chat_url = self.server_url + "/chat_index?task_group_id="+str(self.task_group_id)
        print_and_log(mturk_chat_url, False)
        mturk_page_url = None
        for mturk_agent_id in self.mturk_agent_ids:
            self.mturk_agent_hit_id_dict[mturk_agent_id] = {}
            for hit_index in range(1, opt['num_hits']+1):
                mturk_page_url, hit_id = create_hit_with_hit_type(
                    page_url=mturk_chat_url,
                    hit_type_id=hit_type_id,
                    num_assignments=opt['num_assignments'],
                    is_sandbox=opt['is_sandbox']
                )
                self.mturk_agent_hit_id_dict[mturk_agent_id][hit_index] = hit_id
        print_and_log("Link to HIT: " + mturk_page_url + "\n")
        print_and_log("Waiting for Turkers to respond... (Please don't close your laptop or put your computer into sleep or standby mode.)\n")
        # if opt['is_sandbox']:
        #     webbrowser.open(mturk_page_url)
        return mturk_page_url

    def expire_hit(self, hit_id):
        client = get_mturk_client(self.is_sandbox)
        client.update_expiration_for_hit(HITId=hit_id, ExpireAt=datetime(2015, 1, 1)) # Update it to a time in the past, and the HIT will be immediately expired

    def expire_all_unassigned_hits(self):
        print_and_log("Expiring all unassigned HITs...")
        print_and_log(self.mturk_agent_hit_id_dict, False)
        for mturk_agent_id in self.mturk_agent_hit_id_dict:
            for hit_index in self.mturk_agent_hit_id_dict[mturk_agent_id]:
                hit_id = self.mturk_agent_hit_id_dict[mturk_agent_id][hit_index]
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
    agent_global_id_to_instance = {}
    
    @classmethod
    def get_agent_global_id(cls, task_group_id, conversation_id, agent_id):
        # Global ID within the local system
        return task_group_id + '_' + conversation_id + '_' + agent_id

    def __init__(self, id, manager, hit_index, assignment_index, opt, shared=None):
        super().__init__(opt)

        self.conversation_id = str(hit_index) + '_' + str(assignment_index)
        self.manager = manager
        self.id = id
        self.assignment_id = None
        self.hit_id = None
        self.worker_id = None
        self.hit_is_abandoned = False
        self.agent_global_id = MTurkAgent.get_agent_global_id(
                                    task_group_id=self.manager.task_group_id,
                                    conversation_id=self.conversation_id,
                                    agent_id=id
                                )
        self.alive_event = threading.Event()
        self.received_message_ids = []
        self.new_message = None
        self.new_message_lock = threading.Lock()

        self.manager.register_agent(self)

        # Wait for Turker to accept the HIT
        if self.alive_event.is_set():
            self.alive_event.clear()
        self.alive_event.wait()
        self.manager.register_heartbeat(self)

    def notify_alive(self):
        if not self.alive_event.is_set():
            self.alive_event.set()


    # def notify_new_message(self):
    #     with self.new_message_condition:
    #         self.new_message_condition.notify()

    def observe(self, msg):
        self.manager.send_new_message(
            task_group_id=self.manager.task_group_id,
            conversation_id=self.conversation_id,
            sender_agent_id=msg['id'],
            receiver_agent_id=self.id,
            message_text=msg.get('text', None),
            reward=msg.get('reward', None),
            episode_done=msg.get('episode_done', False),
        )

    def act(self, timeout=None): # Timeout in seconds, after which the HIT will be expired automatically
        self.manager.send_new_command(
            task_group_id=self.manager.task_group_id,
            conversation_id=self.conversation_id,
            sender_agent_id='[World]',
            receiver_agent_id=self.id,
            command=COMMAND_SEND_MESSAGE
        )

        if timeout:
            start_time = time.time()

        # Wait for agent's new message
        while not self.new_message:
            if timeout:
                current_time = time.time()
                if (current_time - start_time) > timeout:
                    self.set_hit_is_abandoned()
                    msg = {
                        'id': self.id,
                        'text': TIMEOUT_MESSAGE,
                        'episode_done': True
                    }
                    return msg
            time.sleep(0.1)

        with self.new_message_lock:
            new_message = self.new_message
            self.new_message = None
            return new_message

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
            self.manager.send_new_command(
                task_group_id=self.manager.task_group_id,
                conversation_id=self.conversation_id,
                sender_agent_id='[World]',
                receiver_agent_id=self.id,
                command=COMMAND_EXPIRE_HIT
            )

    def wait_for_hit_completion(self, timeout=None): # Timeout in seconds, after which the HIT will be expired automatically
        if timeout:
            start_time = time.time()
        while self.manager.get_agent_work_status(assignment_id=self.assignment_id) != ASSIGNMENT_DONE:
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
        if not self.hit_is_abandoned:
            self.manager.send_new_command(
                task_group_id=self.manager.task_group_id,
                conversation_id=self.conversation_id,
                sender_agent_id='[World]',
                receiver_agent_id=self.id,
                command=command_to_send
            )
            return self.wait_for_hit_completion(timeout=timeout)
