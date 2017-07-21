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
import webbrowser
import json
import requests
from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.setup_aws import setup_aws, calculate_mturk_cost, check_mturk_balance, create_hit_type, create_hit_with_hit_type, setup_aws_credentials, create_hit_config, get_mturk_client
import threading
from parlai.mturk.core.data_model import Base, Message
from parlai.mturk.core.data_model import get_new_messages as _get_new_messages
from parlai.mturk.core.data_model import COMMAND_GET_NEW_MESSAGES, COMMAND_SEND_MESSAGE, COMMAND_SUBMIT_HIT
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from botocore.exceptions import ClientError
import uuid
try:
    import sqlite3
except ModuleNotFoundError:
    raise SystemExit("Please install sqlite3 by running: pip install sqlite3")

ASSIGNMENT_NOT_DONE = 'NotDone'
ASSIGNMENT_DONE = 'Submitted'
ASSIGNMENT_APPROVED = 'Approved'
ASSIGNMENT_REJECTED = 'Rejected'

TIMEOUT_MESSAGE = '[TIMEOUT]'

polling_interval = 1 # in seconds
local_db_lock = threading.Lock()
debug = False

class MTurkManager():
    def __init__(self, opt, mturk_agent_ids):
        self.html_api_endpoint_url = None
        self.json_api_endpoint_url = None
        self.task_group_id = None
        self.db_last_message_id = 0
        self.db_thread = None
        self.db_thread_stop_event = None
        self.run_id = None
        self.mturk_agent_ids = mturk_agent_ids
        self.task_files_to_copy = None
        self.unsent_messages_lock = threading.Lock()
        self.unsent_messages = []
        self.is_sandbox = opt['is_sandbox']

    def init_aws(self, opt, task_directory_path=None):
        print("\nYou are going to allow workers from Amazon Mechanical Turk to be an agent in ParlAI.\nDuring this process, Internet connection is required, and you should turn off your computer's auto-sleep feature.\n")
        key_input = input("Please press Enter to continue... ")
        print("")

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

        print('Setting up MTurk backend...')
        create_hit_config(task_description=opt['task_description'], num_hits=opt['num_hits'], num_assignments=opt['num_assignments'], is_sandbox=opt['is_sandbox'])
        if not self.task_files_to_copy:
            self.task_files_to_copy = []
        if not task_directory_path:
            task_directory_path = os.path.join(opt['parlai_home'], 'parlai', 'mturk', 'tasks', opt['task'])
        for mturk_agent_id in self.mturk_agent_ids:
            self.task_files_to_copy.append(os.path.join(task_directory_path, 'html', mturk_agent_id+'_cover_page.html'))
            self.task_files_to_copy.append(os.path.join(task_directory_path, 'html', mturk_agent_id+'_index.html'))
        html_api_endpoint_url, json_api_endpoint_url = setup_aws(task_files_to_copy = self.task_files_to_copy)
        self.html_api_endpoint_url = html_api_endpoint_url
        self.json_api_endpoint_url = json_api_endpoint_url
        if debug:
            print(self.json_api_endpoint_url)
        print("MTurk setup done.\n")

        # Create an engine connected to the in-memory database
        engine = create_engine('sqlite://',
                                connect_args={'check_same_thread':False},
                                poolclass=StaticPool)
         
        # Create all tables in the engine
        Base.metadata.create_all(engine)
        Base.metadata.bind = engine

        session_maker = sessionmaker(bind=engine)

        self.db_session = scoped_session(session_maker)

    def start_new_run(self, opt):
        if self.db_thread_stop_event:
            self.db_thread_stop_event.set()

        self.run_id = str(int(time.time()))
        self.task_group_id = str(opt['task']) + '_' + str(self.run_id)

        self.db_thread_stop_event = threading.Event()
        self.db_thread = threading.Thread(target=self._sync_with_remote_db, args=())
        self.db_thread.daemon = True
        self.db_thread.start()

    def _sync_with_remote_db(self):
        while not self.db_thread_stop_event.is_set():
            if debug:
                print("Syncing with remote db...")
            self.get_new_messages_and_save_to_db()
            time.sleep(polling_interval)

    def get_new_messages_and_save_to_db(self):
        params = {
            'method_name': 'get_new_messages',
            'task_group_id': self.task_group_id,
            'last_message_id': self.db_last_message_id,
            'receiver_agent_id': '[World]'
        }
        response = requests.get(self.json_api_endpoint_url, params=params)
        try:
            ret = json.loads(response.json())
        except Exception as e:
            print(response.content)
            raise e
        conversation_dict = ret['conversation_dict']
        if ret['last_message_id']:
            self.db_last_message_id = ret['last_message_id']

        # Go through conversation_dict and save data in local db
        for conversation_id, new_messages in conversation_dict.items():
            for new_message in new_messages:
                with local_db_lock:
                    if self.db_session.query(Message).filter(Message.id==new_message['message_id']).count() == 0:
                        obs_act_dict = {k:new_message[k] for k in new_message if k not in ['message_id']}
                        new_message_in_local_db = Message(
                                                    id = new_message['message_id'],
                                                    task_group_id = self.task_group_id,
                                                    conversation_id = conversation_id,
                                                    sender_agent_id = new_message['id'],
                                                    receiver_agent_id = new_message['receiver_agent_id'],
                                                    message_content = json.dumps(obs_act_dict)
                                                )
                        self.db_session.add(new_message_in_local_db)
                        self.db_session.commit()
    
    # Only gets new messages from local db, which syncs with remote db every `polling_interval` seconds.
    def get_new_messages(self, task_group_id, conversation_id, receiver_agent_id, after_message_id, excluded_sender_agent_id=None, included_sender_agent_id=None):
        with local_db_lock:
            return _get_new_messages(
                db_session=self.db_session,
                task_group_id=task_group_id,
                conversation_id=conversation_id,
                receiver_agent_id=receiver_agent_id,
                after_message_id=after_message_id,
                excluded_sender_agent_id=excluded_sender_agent_id,
                included_sender_agent_id=included_sender_agent_id,
                populate_meta_info=True
            )

    def send_new_message(self, task_group_id, conversation_id, sender_agent_id, receiver_agent_id, message_text=None, reward=None, episode_done=False):
        post_data_dict = {
            'method_name': 'send_new_message',
            'task_group_id': task_group_id,
            'conversation_id': conversation_id,
            'sender_agent_id': sender_agent_id,
            'receiver_agent_id': receiver_agent_id,
            'episode_done': episode_done,
        }
        if message_text:
            post_data_dict['text'] = message_text
        if reward:
            post_data_dict['reward'] = reward

        response = requests.post(self.json_api_endpoint_url, data=json.dumps(post_data_dict))
        try:
            ret = json.loads(response.json())
            return ret
        except Exception as e:
            print(response.content)
            raise e

    def send_new_command(self, task_group_id, conversation_id, receiver_agent_id, command):
        post_data_dict = {
            'method_name': 'send_new_command',
            'task_group_id': task_group_id,
            'conversation_id': conversation_id,
            'receiver_agent_id': receiver_agent_id,
            'command': command,
        }
        response = requests.post(self.json_api_endpoint_url, data=json.dumps(post_data_dict))
        try:
            ret = json.loads(response.json())
            return ret
        except Exception as e:
            print(response.content)
            raise e

    def get_hit_assignment_info(self, task_group_id, conversation_id, agent_id):
        params = {
            'method_name': 'get_hit_assignment_info',
            'task_group_id': task_group_id,
            'agent_id': agent_id,
            'conversation_id': conversation_id
        }
        response = requests.get(self.json_api_endpoint_url, params=params)
        try:
            ret = json.loads(response.json())
            return ret['assignment_id'], ret['hit_id'], ret['worker_id']
        except Exception as e:
            print(response.content)
            raise e

    def get_agent_work_status(self, assignment_id):
        client = get_mturk_client(self.is_sandbox)
        try:
            response = client.get_assignment(AssignmentId=assignment_id)
            return response['Assignment']['AssignmentStatus']
        except ClientError as e:
            if 'This operation can be called with a status of: Reviewable,Approved,Rejected' in e.response['Error']['Message']:
                return ASSIGNMENT_NOT_DONE

    def create_hits(self, opt):
        print('Creating HITs...')
        mturk_agent_HIT_url_dict = {}
        for mturk_agent_id in self.mturk_agent_ids:
            for hit_index in range(1, opt['num_hits']+1):
                hit_type_id = create_hit_type(
                    hit_title=opt['hit_title'],
                    hit_description=opt['hit_description'] + ' (ID: ' + self.task_group_id + ', Role: ' + mturk_agent_id + ')',
                    hit_keywords=opt['hit_keywords'],
                    hit_reward=opt['reward'],
                    assignment_duration_in_seconds=opt.get('assignment_duration_in_seconds', 30 * 60), # Set to 30 minutes by default
                    is_sandbox=opt['is_sandbox']
                )
                mturk_chat_url = self.html_api_endpoint_url + "?method_name=chat_index&task_group_id="+str(self.task_group_id)+"&cur_agent_id="+str(mturk_agent_id)
                mturk_page_url = create_hit_with_hit_type(
                    page_url=mturk_chat_url,
                    hit_type_id=hit_type_id,
                    num_assignments=opt['num_assignments'],
                    is_sandbox=opt['is_sandbox']
                )
            print("Link to HIT for " + str(mturk_agent_id) + ": " + mturk_page_url + "\n")
            print("Waiting for Turkers to respond... (Please don't close your laptop or put your computer into sleep or standby mode.)\n")
            mturk_agent_HIT_url_dict[mturk_agent_id] = mturk_page_url
        return mturk_agent_HIT_url_dict

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
            print("Cannot pay bonus. Reason: Insufficient fund in your MTurk account.")
            return False

        client = get_mturk_client(self.is_sandbox)
        client.send_bonus(
            WorkerId=worker_id,
            BonusAmount=str(bonus_amount),
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
        setup_aws_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'setup_aws.py')
        print("Remote database instance will accumulate cost over time (about $30/month for t2.medium instance). Please run `python "+setup_aws_file_path+" remove_rds` to remove RDS instance if you don't plan to use MTurk often.")
        self.db_thread_stop_event.set()


class MTurkAgent(Agent):
    def __init__(self, id, manager, hit_index, assignment_index, opt, shared=None):
        super().__init__(opt)

        self.conversation_id = str(hit_index) + '_' + str(assignment_index)
        self.manager = manager
        self.id = id
        self.last_message_id = 0
        self.assignment_id = None
        self.hit_id = None
        self.worker_id = None
        self.hit_is_abandoned = False

        # Wait for MTurk-specific info
        while not (self.assignment_id and self.hit_id and self.worker_id):
            self.assignment_id, self.hit_id, self.worker_id = self.manager.get_hit_assignment_info(self.manager.task_group_id, self.conversation_id, self.id)
            time.sleep(polling_interval)

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
        self.manager.send_new_command(
            task_group_id=self.manager.task_group_id,
            conversation_id=self.conversation_id,
            receiver_agent_id=self.id,
            command=COMMAND_GET_NEW_MESSAGES
        )

    def act(self, timeout=None): # timeout in seconds
        if timeout:
            start_time = time.time()

        self.manager.send_new_command(
            task_group_id=self.manager.task_group_id,
            conversation_id=self.conversation_id,
            receiver_agent_id=self.id,
            command=COMMAND_SEND_MESSAGE
        )

        while True:
            if timeout:
                current_time = time.time()
                if (current_time - start_time) > timeout:
                    self.hit_is_abandoned = True
                    msg = {
                        'id': self.id,
                        'text': TIMEOUT_MESSAGE,
                        'episode_done': True
                    }
                    return msg

            conversation_dict, new_last_message_id = self.manager.get_new_messages(
                task_group_id=self.manager.task_group_id,
                conversation_id=self.conversation_id,
                receiver_agent_id='[World]',
                after_message_id=self.last_message_id,
                included_sender_agent_id=self.id
            )

            if self.conversation_id in conversation_dict:
                if new_last_message_id:
                    self.last_message_id = new_last_message_id

                new_messages = conversation_dict[self.conversation_id]

                return new_messages[0]

            time.sleep(polling_interval)

    def episode_done(self):
        return False

    def approve_work(self):
        if self.hit_is_abandoned:
            print('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is abandoned and thus not available for review.')
        else:
            if self.manager.get_agent_work_status(assignment_id=self.assignment_id) == ASSIGNMENT_DONE:
                self.manager.approve_work(assignment_id=self.assignment_id)
                print('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is approved.')
            else:
                print("Cannot approve HIT. Reason: Turker hasn't completed the HIT yet.")

    def reject_work(self, reason='unspecified'):
        if self.hit_is_abandoned:
            print('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is abandoned and thus not available for review.')
        else:
            if self.manager.get_agent_work_status(assignment_id=self.assignment_id) == ASSIGNMENT_DONE:
                self.manager.reject_work(assignment_id=self.assignment_id, reason=reason)
                print('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is rejected.')
            else:
                print("Cannot reject HIT. Reason: Turker hasn't completed the HIT yet.")

    def block_worker(self, reason='unspecified'):
        self.manager.block_worker(worker_id=self.worker_id, reason=reason)
        print("Blocked worker ID: " + str(self.worker_id) + ". Reason: " + reason)

    def pay_bonus(self, bonus_amount, reason='unspecified'):
        if self.hit_is_abandoned:
            print('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is abandoned and thus not available for bonus.')
        else:
            if self.manager.get_agent_work_status(assignment_id=self.assignment_id) != ASSIGNMENT_NOT_DONE:
                unique_request_token = str(uuid.uuid4())
                if self.manager.pay_bonus(worker_id=self.worker_id, bonus_amount=bonus_amount, assignment_id=self.assignment_id, reason=reason, unique_request_token=unique_request_token):
                    print("Paid $" + str(bonus_amount) + " bonus to WorkerId: " + self.worker_id)
            else:
                print("Cannot pay bonus for HIT. Reason: Turker hasn't completed the HIT yet.")

    def email_worker(self, subject, message_text):
        response = self.manager.email_worker(worker_id=self.worker_id, subject=subject, message_text=message_text)
        if 'success' in response:
            print("Email sent to worker ID: "+str(self.worker_id)+": Subject: "+str(subject)+": Text: "+str(message_text))
            return True
        elif 'failure' in response:
            print("Unable to send email to worker ID: "+str(self.worker_id)+". Error: "+str(response['failure']))
            return False

    def wait_for_hit_completion(self, timeout=None): # Timeout in seconds
        if timeout:
            start_time = time.time()
        while self.manager.get_agent_work_status(assignment_id=self.assignment_id) != ASSIGNMENT_DONE:
            if timeout:
                current_time = time.time()
                if (current_time - start_time) > timeout:
                    print("Timed out waiting for Turker to complete the HIT.")
                    self.hit_is_abandoned = True
                    return False
            if debug:
                print("Waiting for Turker to complete the HIT...")
            time.sleep(polling_interval)
        print('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is done.')

    def shutdown(self, timeout=None): # Timeout in seconds
        if not self.hit_is_abandoned:
            self.manager.send_new_command(
                task_group_id=self.manager.task_group_id,
                conversation_id=self.conversation_id,
                receiver_agent_id=self.id,
                command=COMMAND_SUBMIT_HIT
            )
            self.wait_for_hit_completion(timeout=timeout)
