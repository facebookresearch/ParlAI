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
from .setup_aws import setup_aws, check_mturk_balance, create_hit_type, create_hit_with_hit_type, setup_aws_credentials
import threading
from .data_model import Base, Message
from .data_model import get_new_messages as _get_new_messages
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
try:
    import sqlite3
except ModuleNotFoundError:
    raise SystemExit("Please install sqlite3 by running: pip install sqlite3")

local_db_file_path_template = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'tmp', 'parlai_mturk_<run_id>.db')
polling_interval = 1 # in seconds
create_hit_type_lock = threading.Lock()
local_db_lock = threading.Lock()

class MTurkManager():
    def __init__(self):
        self.html_api_endpoint_url = None
        self.json_api_endpoint_url = None
        self.requester_key_gt = None
        self.task_group_id = None
        self.db_last_message_id = 0
        self.db_thread = None
        self.db_thread_stop_event = None
        self.local_db_file_path = None
        self.run_id = None
        self.mturk_agent_ids = None
        self.all_agent_ids = None

    def init_aws(self, opt):
        self.run_id = str(int(time.time()))

        print("\nYou are going to allow workers from Amazon Mechanical Turk to be an agent in ParlAI.\nDuring this process, Internet connection is required, and you should turn off your computer's auto-sleep feature.\n")
        key_input = input("Please press Enter to continue... ")
        print("")

        setup_aws_credentials()

        if not check_mturk_balance(num_hits=opt['num_hits'], hit_reward=opt['reward'], is_sandbox=opt['is_sandbox']):
            return

        print('Setting up MTurk backend...')
        html_api_endpoint_url, json_api_endpoint_url, requester_key_gt = setup_aws(task_description=opt['task_description'], num_hits=opt['num_hits'], num_assignments=opt['num_assignments'], is_sandbox=opt['is_sandbox'])
        self.html_api_endpoint_url = html_api_endpoint_url
        self.json_api_endpoint_url = json_api_endpoint_url
        self.requester_key_gt = requester_key_gt
        print("MTurk setup done.\n")

        self.task_group_id = str(opt['task']) + '_' + str(self.run_id)

        # self.connection = sqlite3.connect(local_db_file_name)

        self.local_db_file_path = local_db_file_path_template.replace('<run_id>', self.run_id)

        if not os.path.exists(os.path.dirname(self.local_db_file_path)):
            os.makedirs(os.path.dirname(self.local_db_file_path))

        # Create an engine
        engine = create_engine('sqlite:///'+self.local_db_file_path,
                                connect_args={'check_same_thread':False},
                                poolclass=StaticPool)
         
        # Create all tables in the engine
        Base.metadata.create_all(engine)
        Base.metadata.bind = engine

        session_maker = sessionmaker(bind=engine)

        self.db_session = scoped_session(session_maker)

        self.db_thread_stop_event = threading.Event()
        self.db_thread = threading.Thread(target=self._poll_new_messages_and_save_to_db, args=())
        self.db_thread.daemon = True
        self.db_thread.start()

    def _poll_new_messages_and_save_to_db(self):
        while not self.db_thread_stop_event.is_set():
            self.get_new_messages_and_save_to_db()
            time.sleep(polling_interval)

    def get_new_messages_and_save_to_db(self):
        params = {
            'method_name': 'get_new_messages',
            'task_group_id': self.task_group_id,
            'last_message_id': self.db_last_message_id,
        }
        request = requests.get(self.json_api_endpoint_url, params=params)
        try:
            ret = json.loads(request.json())
        except TypeError as e:
            print(request.json())
            raise e
        conversation_dict = ret['conversation_dict']
        if ret['last_message_id']:
            self.db_last_message_id = ret['last_message_id']

        # Go through conversation_dict and save data in local db
        for conversation_id, new_messages in conversation_dict.items():
            for new_message in new_messages:
                with local_db_lock:
                    if self.db_session.query(Message).filter(Message.id==new_message['message_id']).count() == 0:
                        obs_act_dict = {k:new_message[k] for k in new_message if k != 'message_id'}
                        new_message_in_local_db = Message(
                                                    id = new_message['message_id'],
                                                    task_group_id = self.task_group_id,
                                                    conversation_id = conversation_id,
                                                    agent_id = new_message['id'],
                                                    message_content = json.dumps(obs_act_dict)
                                                )
                        self.db_session.add(new_message_in_local_db)
                        self.db_session.commit()
    
    # Only gets new messages from local db, which syncs with remote db every `polling_interval` seconds.
    def get_new_messages(self, task_group_id, conversation_id, after_message_id, excluded_agent_id=None, included_agent_id=None):
        with local_db_lock:
            return _get_new_messages(
                db_session=self.db_session,
                task_group_id=task_group_id,
                conversation_id=conversation_id,
                after_message_id=after_message_id,
                excluded_agent_id=excluded_agent_id,
                included_agent_id=included_agent_id,
                populate_meta_info=True
            )

    def send_new_message(self, task_group_id, conversation_id, agent_id, message_text=None, reward=None, episode_done=False):
        post_data_dict = {
            'method_name': 'send_new_message',
            'task_group_id': task_group_id,
            'conversation_id': conversation_id,
            'cur_agent_id': agent_id,
            'episode_done': episode_done,
        }
        if message_text:
            post_data_dict['text'] = message_text
        if reward:
            post_data_dict['reward'] = reward

        request = requests.post(self.json_api_endpoint_url, data=json.dumps(post_data_dict))
        try:
            ret = json.loads(request.json())
            return ret
        except TypeError as e:
            print(request.json())
            raise e

    def get_approval_status_count(self, task_group_id, approval_status, requester_key, conversation_id=None):
        params = {
            'method_name': 'get_approval_status_count',
            'task_group_id': task_group_id,
            'approval_status': approval_status,
            'requester_key': requester_key
        }
        if conversation_id:
            params['conversation_id'] = conversation_id
        request = requests.get(self.json_api_endpoint_url, params=params)
        return request.json()

    def create_hits(self, opt):
        print('Creating HITs...')
        for mturk_agent_id in self.mturk_agent_ids:
            for hit_index in range(1, opt['num_hits']+1):
                with create_hit_type_lock:
                    hit_type_id = create_hit_type(
                        hit_title=opt['hit_title'],
                        hit_description=opt['hit_description'] + ' (ID: ' + self.task_group_id + ', Role: ' + mturk_agent_id + ')',
                        hit_keywords=opt['hit_keywords'],
                        hit_reward=opt['reward'],
                        is_sandbox=opt['is_sandbox']
                    )
                all_agent_ids_string = str(self.all_agent_ids).replace("'", '''"''')
                mturk_chat_url = self.html_api_endpoint_url + "?method_name=chat_index&task_group_id="+str(self.task_group_id)+"&all_agent_ids="+all_agent_ids_string+"&cur_agent_id="+str(mturk_agent_id)+"&task_additional_info="+str(opt.get('task_additional_info', ''))
                mturk_page_url = create_hit_with_hit_type(
                    page_url=mturk_chat_url,
                    hit_type_id=hit_type_id,
                    num_assignments=opt['num_assignments'],
                    is_sandbox=opt['is_sandbox']
                )
            print("Link to HIT for " + str(mturk_agent_id) + ": " + mturk_page_url + "\n")
            print("Waiting for Turkers to respond... (Please don't close your laptop or put your computer into sleep or standby mode.)\n")

    def review_hits(self):
        mturk_agent_ids_string = str(self.mturk_agent_ids).replace("'", '''"''')
        mturk_approval_url = self.html_api_endpoint_url + "?method_name=approval_index&task_group_id="+str(self.task_group_id)+"&hit_index=1&assignment_index=1&mturk_agent_ids="+mturk_agent_ids_string+"&requester_key="+self.requester_key_gt

        print("\nAll HITs are done! Please go to the following link to approve/reject them (or they will be auto-approved in 4 weeks if no action is taken):\n")
        print(mturk_approval_url)
        print("")

        # Loop for checking approval status
        while self.get_approval_status_count(task_group_id=self.task_group_id, approval_status='pending', requester_key=self.requester_key_gt) > 0:
            time.sleep(polling_interval)

        print("All reviews are done!")

    def shutdown(self):
        self.db_thread_stop_event.set()
        if os.path.exists(self.local_db_file_path):
            os.remove(self.local_db_file_path)


class MTurkAgent(Agent):
    def __init__(self, id, manager, conversation_id, opt, shared=None):
        super().__init__(opt)

        self.conversation_id = conversation_id
        self.manager = manager
        self.id = id
        self.last_message_id = 0

    def observe(self, msg):
        if msg['id'] not in self.manager.mturk_agent_ids: # If the message sender is an mturk agent, then there is no need to upload this message to db since it's already been done on the message sender side.
            self.manager.get_new_messages_and_save_to_db() # Force a refresh for local db.
            conversation_dict, _ = self.manager.get_new_messages(
                task_group_id=self.manager.task_group_id,
                conversation_id=self.conversation_id,
                after_message_id=self.last_message_id,
                included_agent_id=msg['id'])
            if self.conversation_id in conversation_dict:
                agent_last_message_in_db = conversation_dict[self.conversation_id][-1]
                agent_last_message_in_db.pop('message_id', None)
                if 'episode_done' not in msg:
                    msg['episode_done'] = False
                if agent_last_message_in_db == msg:
                    return

            self.manager.send_new_message(
                task_group_id=self.manager.task_group_id,
                conversation_id=self.conversation_id,
                agent_id=msg['id'],
                message_text=msg.get('text', None),
                reward=msg.get('reward', None),
                episode_done=msg.get('episode_done', False),
            )

    def act(self):
        while True:
            conversation_dict, new_last_message_id = self.manager.get_new_messages(
                task_group_id=self.manager.task_group_id,
                conversation_id=self.conversation_id,
                after_message_id=self.last_message_id,
                included_agent_id=self.id
            )
            
            if self.conversation_id in conversation_dict:
                if new_last_message_id:
                    self.last_message_id = new_last_message_id

                new_messages = conversation_dict[self.conversation_id]

                return new_messages[0]

            time.sleep(polling_interval)

    def episode_done(self):
        return False

    def shutdown(self):
        # Loop to ensure all HITs are done
        while self.manager.get_approval_status_count(task_group_id=self.manager.task_group_id, conversation_id=self.conversation_id, approval_status='pending', requester_key=self.manager.requester_key_gt) < len(self.manager.mturk_agent_ids):
            time.sleep(polling_interval)
        print('Conversation ID: ' + str(self.conversation_id) + ', Agent ID: ' + self.id + ' - HIT is done.')
