# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

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


def _get_random_alphanumeric_string(N):
    return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(N))


def _setup_relay(task_config, num_hits, is_sandbox):
    """Sets up relay server
    """
    # set up relay server
    html_api_endpoint_url, json_api_endpoint_url, requester_key_gt = setup_aws(task_config, num_hits, is_sandbox)

    return html_api_endpoint_url, json_api_endpoint_url, requester_key_gt

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

def _get_new_messages(json_api_endpoint_url, task_group_id, after_message_id, excluded_agent_id=None):
    params = {
        'method_name': 'get_new_messages',
        'task_group_id': task_group_id,
        'last_message_id': after_message_id,
    }
    if excluded_agent_id:
        params['excluded_agent_id'] = excluded_agent_id

    request = requests.get(json_api_endpoint_url, params=params)
    return json.loads(request.json())

def _get_pending_review_count(json_api_endpoint_url, task_group_id, requester_key):
    params = {
        'method_name': 'get_pending_review_count',
        'task_group_id': task_group_id,
        'requester_key': requester_key
    }
    request = requests.get(json_api_endpoint_url, params=params)
    return request.json()

def _get_all_review_status(json_api_endpoint_url, task_group_id, requester_key):
    params = {
        'method_name': 'get_all_review_status',
        'task_group_id': task_group_id,
        'requester_key': requester_key
    }
    request = requests.get(json_api_endpoint_url, params=params)
    return request.json()

def create_hits(opt, task_config, task_module_name, bot, chat_page_only=False):
    num_hits = opt['num_hits']
    hit_reward = opt['reward']
    is_sandbox = opt['is_sandbox']
    verbose = opt['verbose']

    print("\nYou are going to allow workers from Amazon Mechanical Turk to chat with your dialog model running on your local machine.\nDuring this process, Internet connection is required, and you should turn off your computer's auto-sleep feature.\n")
    key_input = input("Please press Enter to continue... ")
    print("")

    setup_aws_credentials()
    if not check_mturk_balance(num_hits=num_hits, hit_reward=hit_reward, is_sandbox=is_sandbox):
        return

    task_group_id = str(int(time.time())) + '_' + _get_random_alphanumeric_string(10) # Random string to further avoid collision

    print('Setting up MTurk backend...')
    html_api_endpoint_url, json_api_endpoint_url, requester_key_gt = _setup_relay(task_config, num_hits, is_sandbox)

    approval_index_url_template = html_api_endpoint_url + "?method_name=approval_index&task_group_id={{task_group_id}}&conversation_id=1&cur_agent_id={{cur_agent_id}}&requester_key="+requester_key_gt

    worker_agent_id = task_config['worker_agent_id']
    bot_agent_id = bot.getID()
    cids = range(1, num_hits+1)
    cid_map = {cid: i for i, cid in enumerate(cids)}
    c_done_map = {cid: False for cid in cids}
    logs = {cid: [] for cid in cids}

    shared = bot.share()
    bots = []
    last_message_id = -1

    # If the bot needs to send the first message of the conversation, it will send it here
    for cid in cids:
        new_bot = create_agent_from_shared(shared)
        new_bot.conversation_id = cid
        bots.append(new_bot)
        response = new_bot.act()
        if response:
            if response.get('episode_done', False):
                c_done_map[cid] = True
            if verbose:
                print('Conversation '+str(cid)+' - Bot says: ' + str(response))
            logs[cid].append(response)
            new_message = _send_new_message(
                json_api_endpoint_url=json_api_endpoint_url,
                task_group_id=task_group_id,
                conversation_id=cid,
                agent_id=bot_agent_id,
                message_text=response.get('text', None),
                reward=response.get('reward', None),
                episode_done=response.get('episode_done', False),
            )
            if new_message['message_id'] > last_message_id:
                last_message_id = new_message['message_id']

    hits_created = False
    conversations_remaining = set(cids)

    # Main loop for polling and handling new messages
    while len(conversations_remaining) > 0:
        ret = _get_new_messages(
            json_api_endpoint_url=json_api_endpoint_url,
            task_group_id=task_group_id,
            after_message_id=last_message_id,
            excluded_agent_id=bot_agent_id,
        )
        conversation_dict = ret['conversation_dict']
        new_last_message_id = ret['last_message_id']

        if new_last_message_id:
            last_message_id = new_last_message_id

        time.sleep(1)

        for conversation_id, new_messages in conversation_dict.items():
            conversation_id = int(conversation_id)
            if conversation_id in conversations_remaining and len(new_messages) > 0:
                agent = bots[cid_map[conversation_id]]
                for new_message in new_messages:
                    if verbose:
                        print('Conversation '+str(conversation_id)+' - Bot received: ' + str(new_message))
                    logs[conversation_id].append(new_message)
                    agent.observe(new_message)
                    if new_message.get('episode_done', False) or c_done_map[conversation_id]:
                        # We're done here
                        conversations_remaining.remove(conversation_id)
                        print('Conversation '+str(conversation_id)+' is DONE!\n')
                    else:
                        # Agent still needs to reply
                        response = agent.act()
                        if response:
                            if response.get('episode_done', False):
                                c_done_map[conversation_id] = True
                            if verbose:
                                print('Conversation '+str(conversation_id)+' - Bot says: ' + str(response))
                            logs[conversation_id].append(response)
                            _send_new_message(
                                json_api_endpoint_url=json_api_endpoint_url,
                                task_group_id=task_group_id,
                                conversation_id=conversation_id,
                                agent_id=bot_agent_id,
                                message_text=response.get('text', None),
                                reward=response.get('reward', None),
                                episode_done=response.get('episode_done', False),
                            )

        # We don't create new HITs until this point, so that the HIT page will always have the conversation fully populated.
        if not hits_created:
            print('Creating HITs...')
            hit_type_id = create_hit_type(
                hit_title=task_config['hit_title'],
                hit_description=task_config['hit_description'] + ' (ID: ' + task_group_id + ')',
                hit_keywords=task_config['hit_keywords'],
                hit_reward=hit_reward,
                is_sandbox=is_sandbox
            )
            mturk_chat_url = None
            mturk_page_url = None
            for cid in cids:
                mturk_chat_url = html_api_endpoint_url + "?method_name=chat_index&task_group_id="+str(task_group_id)+"&conversation_id="+str(cid)+"&cur_agent_id="+str(worker_agent_id)
                if not chat_page_only:
                    mturk_page_url = create_hit_with_hit_type(
                        page_url=mturk_chat_url,
                        hit_type_id=hit_type_id,
                        is_sandbox=is_sandbox
                    )

            print("MTurk setup done.\n")
            if chat_page_only:
                webbrowser.open(mturk_chat_url)
            else:
                print("Link to your HIT: " + mturk_page_url + "\n")
                print("Waiting for Turkers to complete the tasks... (Please don't close your laptop or put your computer into sleep or standby mode.)\n")
            hits_created = True

    while _get_pending_review_count(json_api_endpoint_url=json_api_endpoint_url, task_group_id=task_group_id, requester_key=requester_key_gt) != num_hits:
        time.sleep(2)

    mturk_approval_url = html_api_endpoint_url + "?method_name=approval_index&task_group_id="+str(task_group_id)+"&conversation_id=1&cur_agent_id="+worker_agent_id+"&requester_key="+requester_key_gt
    print("\nAll HITs are done! Please go to the following link to approve/reject them (or they will be auto-approved in 4 weeks if no action is taken):\n")
    print(mturk_approval_url)
    print("")

    approval_status_dict = {cid: '' for cid in cids}
    # Loop for checking approval status
    while _get_pending_review_count(json_api_endpoint_url=json_api_endpoint_url, task_group_id=task_group_id, requester_key=requester_key_gt) > 0:
        time.sleep(2)

    print("Approvals are done!")

    for hit_info in _get_all_review_status(json_api_endpoint_url=json_api_endpoint_url, task_group_id=task_group_id, requester_key=requester_key_gt):
        conversation_id = hit_info['conversation_id']
        approval_status_dict[conversation_id] = hit_info['approval_status']

    logs_approved = {cid:log for (cid,log) in logs.items() if approval_status_dict[cid] == 'approved'}
    logs_rejected = {cid:log for (cid,log) in logs.items() if approval_status_dict[cid] == 'rejected'}

    # Saving logs to file
    # Log format: {conversation_id: [list of messages in the conversation]}
    mturk_log_path = opt['mturk_log_path']
    task_group_path = os.path.join(mturk_log_path,
                                   task_module_name + '_' +
                                   datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    os.makedirs(task_group_path)
    with open(os.path.join(task_group_path, 'approved.json'), 'w') as fout:
        fout.write(json.dumps(logs_approved))
    with open(os.path.join(task_group_path, 'rejected.json'), 'w') as fout:
        fout.write(json.dumps(logs_rejected))

    print("All conversations are saved to "+opt['mturk_log_path']+" in JSON format.\n")
