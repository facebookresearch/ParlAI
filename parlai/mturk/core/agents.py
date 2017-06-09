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


def _get_new_messages(json_api_endpoint_url, task_group_id, conversation_id, after_message_id, excluded_agent_id=None, included_agent_id=None):
    params = {
        'method_name': 'get_new_messages',
        'task_group_id': task_group_id,
        'last_message_id': after_message_id,
        'conversation_id': conversation_id,
    }
    if excluded_agent_id:
        params['excluded_agent_id'] = excluded_agent_id
    if included_agent_id:
        params['included_agent_id'] = included_agent_id

    request = requests.get(json_api_endpoint_url, params=params)
    return json.loads(request.json())

def _send_new_message(json_api_endpoint_url, task_group_id, conversation_id, agent_id, message_text=None, reward=None, episode_done=False):
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

    request = requests.post(json_api_endpoint_url, data=json.dumps(post_data_dict))
    return json.loads(request.json())

def _get_review_status_count(json_api_endpoint_url, task_group_id, conversation_id, review_status, requester_key):
    params = {
        'method_name': 'get_review_status_count',
        'task_group_id': task_group_id,
        'conversation_id': conversation_id,
        'review_status': review_status,
        'requester_key': requester_key
    }
    request = requests.get(json_api_endpoint_url, params=params)
    return request.json()

class MTurkAgent(Agent):

    skip_init = False
    html_api_endpoint_url = None
    json_api_endpoint_url = None
    requester_key_gt = None
    
    def __init__(self, opt, shared=None):
        super().__init__(opt)

        self.id = opt['agent_id']
        self.task_name = opt['task']
        self.is_sandbox = opt['is_sandbox']
        self.conversation_id = opt['conversation_id']
        self.mturk_agent_ids = opt['mturk_agent_ids']
        self.all_agent_ids = opt['all_agent_ids']
        self.hit_reward = opt['reward']
        self.hit_title = opt['hit_title']
        self.hit_description = opt['hit_description']
        self.hit_keywords = opt['hit_keywords']
        self.task_description = opt['task_description']

        self.last_message_id = 0

        if not self.__class__.skip_init:
            print("\nYou are going to allow workers from Amazon Mechanical Turk to be an agent in ParlAI.\nDuring this process, Internet connection is required, and you should turn off your computer's auto-sleep feature.\n")
            key_input = input("Please press Enter to continue... ")
            print("")
            
        setup_aws_credentials()

        if not check_mturk_balance(num_hits=1, hit_reward=self.hit_reward, is_sandbox=self.is_sandbox):
            return
            
        if not self.__class__.skip_init:
            print('Setting up MTurk backend...')
            html_api_endpoint_url, json_api_endpoint_url, requester_key_gt = setup_aws(self.task_description, 1, self.is_sandbox)
            self.__class__.html_api_endpoint_url = html_api_endpoint_url
            self.__class__.json_api_endpoint_url = json_api_endpoint_url
            self.__class__.requester_key_gt = requester_key_gt
            print("MTurk setup done.\n")

        self.__class__.skip_init = True
        self.html_api_endpoint_url = self.__class__.html_api_endpoint_url
        self.json_api_endpoint_url = self.__class__.json_api_endpoint_url
        self.requester_key_gt = self.__class__.requester_key_gt

        self.task_group_id = str(self.task_name) + '_' + str(self.conversation_id)

        print('Creating HITs...')
        hit_type_id = create_hit_type(
            hit_title=self.hit_title,
            hit_description=self.hit_description + ' (ID: ' + self.task_group_id + ', Role: ' + self.id + ')',
            hit_keywords=self.hit_keywords,
            hit_reward=self.hit_reward,
            is_sandbox=self.is_sandbox
        )
        all_agent_ids_string = str(self.all_agent_ids).replace("'", '''"''')
        mturk_chat_url = self.html_api_endpoint_url + "?method_name=chat_index&task_group_id="+str(self.task_group_id)+"&conversation_id="+str(self.conversation_id)+"&all_agent_ids="+all_agent_ids_string+"&cur_agent_id="+str(self.id)
        mturk_page_url = create_hit_with_hit_type(
            page_url=mturk_chat_url,
            hit_type_id=hit_type_id,
            is_sandbox=self.is_sandbox
        )

        print("Link to HIT for " + self.id + ": " + mturk_page_url + "\n")
        print("Waiting for Turkers to respond... (Please don't close your laptop or put your computer into sleep or standby mode.)\n")

    def observe(self, msg):
        if msg['id'] not in self.mturk_agent_ids: # If the message sender is an mturk agent, then there is no need to upload this message to db since it's already been done on the message sender side.
            conversation_dict = _get_new_messages(
                json_api_endpoint_url=self.json_api_endpoint_url, 
                task_group_id=self.task_group_id,
                conversation_id=self.conversation_id,
                after_message_id=self.last_message_id,
                included_agent_id=msg['id'])['conversation_dict']
            if self.conversation_id in conversation_dict:
                agent_last_message_in_db = conversation_dict[self.conversation_id][0]
                agent_last_message_in_db.pop('message_id', None)
                if 'episode_done' not in msg:
                    msg['episode_done'] = False
                if agent_last_message_in_db == msg:
                    return

            _send_new_message(
                json_api_endpoint_url=self.json_api_endpoint_url,
                task_group_id=self.task_group_id,
                conversation_id=self.conversation_id,
                agent_id=msg['id'],
                message_text=msg.get('text', None),
                reward=msg.get('reward', None),
                episode_done=msg.get('episode_done', False),
            )

    def act(self):
        while True:
            ret = _get_new_messages(
                json_api_endpoint_url=self.json_api_endpoint_url,
                task_group_id=self.task_group_id,
                conversation_id=self.conversation_id,
                after_message_id=self.last_message_id,
                included_agent_id=self.id
            )
            conversation_dict = ret['conversation_dict']
            
            if str(self.conversation_id) in conversation_dict:
                new_last_message_id = ret['last_message_id']
                if new_last_message_id:
                    self.last_message_id = new_last_message_id

                new_messages = conversation_dict[str(self.conversation_id)]

                return new_messages[0]

            time.sleep(1) # Wait for 1 second, so that we are not polling too frequently.

    def episode_done(self):
        return False

    def shutdown(self):
        if _get_review_status_count(json_api_endpoint_url=self.json_api_endpoint_url, task_group_id=self.task_group_id, conversation_id=self.conversation_id, review_status='approved', requester_key=self.requester_key_gt) + \
            _get_review_status_count(json_api_endpoint_url=self.json_api_endpoint_url, task_group_id=self.task_group_id, conversation_id=self.conversation_id, review_status='rejected', requester_key=self.requester_key_gt) > 0:
            return
        else:
            # Loop to ensure all HITs are done
            while _get_review_status_count(json_api_endpoint_url=self.json_api_endpoint_url, task_group_id=self.task_group_id, conversation_id=self.conversation_id, review_status='pending', requester_key=self.requester_key_gt) < len(self.mturk_agent_ids):
                time.sleep(2)

            mturk_agent_ids_string = str(self.mturk_agent_ids).replace("'", '''"''')
            mturk_approval_url = self.html_api_endpoint_url + "?method_name=approval_index&task_group_id="+str(self.task_group_id)+"&conversation_id="+str(self.conversation_id)+"&mturk_agent_ids="+mturk_agent_ids_string+"&requester_key="+self.requester_key_gt
            print("\nAll HITs are done! Please go to the following link to approve/reject them (or they will be auto-approved in 4 weeks if no action is taken):\n")
            print(mturk_approval_url)
            print("")

            # Loop for checking review status
            while _get_review_status_count(json_api_endpoint_url=self.json_api_endpoint_url, task_group_id=self.task_group_id, conversation_id=self.conversation_id, review_status='pending', requester_key=self.requester_key_gt) > 0:
                time.sleep(2)

            print("All reviews are done!")