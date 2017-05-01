# Copyright 2004-present Facebook. All Rights Reserved.
"""Contains basic functionality for setting up a simple MTurk bot evaluation.
In this example, a bot will be paired with a human, given the default
instructions and opening message, and then will chat with the bot.
"""

import os
import time
from datetime import datetime
import random
import string
import webbrowser
import json
from parlai.core.agents import create_agent_from_shared
from .setup_aws import rds_db_name, rds_username, rds_password, setup_aws, check_mturk_balance, create_hit_type, create_hit_with_hit_type, setup_aws_credentials
from .data_model import Message, init_database, send_new_message, get_new_messages, get_pending_approval_count, get_all_approval_status


def _get_random_alphanumeric_string(N):
    return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(N))


def setup_relay(task_config, num_hits, is_sandbox):
    """Sets up relay server and returns a database session object which can be used to poll
    new messages and send messages
    """
    # set up relay server
    rds_host, mturk_chat_url_template, mturk_approval_url_template = setup_aws(task_config, num_hits, is_sandbox)

    db_engine, db_session_maker = init_database(rds_host, rds_db_name, rds_username, rds_password)
    db_session = db_session_maker()

    return db_session, mturk_chat_url_template, mturk_approval_url_template


def create_hits(opt, task_config, task_module_name, bot, num_hits, hit_reward, is_sandbox=False, chat_page_only=False, verbose=False):
    print("\nYou are going to allow workers from Amazon Mechanical Turk to chat with your dialog model running on your local machine.\nDuring this process, Internet connection is required, and you cannot close your laptop or put your computer into sleep or standby mode.\n")
    key_input = input("Please press Enter to continue:")
    print("")

    setup_aws_credentials()
    if not check_mturk_balance(num_hits=num_hits, hit_reward=hit_reward, is_sandbox=is_sandbox):
        return

    task_group_id = str(int(time.time())) + '_' + _get_random_alphanumeric_string(10) # Random string to further avoid collision

    print('Setting up MTurk backend...')
    db_session, mturk_chat_url_template, mturk_approval_url_template = setup_relay(task_config, num_hits, is_sandbox)

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
            new_message_object = send_new_message(
                db_session=db_session, 
                task_group_id=task_group_id, 
                conversation_id=cid, 
                agent_id=bot_agent_id, 
                message_text=response.get('text', None), 
                reward=response.get('reward', None),
                episode_done=response.get('episode_done', False), 
            )
            if new_message_object.id > last_message_id:
                last_message_id = new_message_object.id

    hits_created = False
    conversations_remaining = set(cids)

    # Main loop for polling and handling new messages
    while len(conversations_remaining) > 0:
        conversation_dict, new_last_message_id = get_new_messages(
            db_session=db_session, 
            task_group_id=task_group_id, 
            after_message_id=last_message_id, 
            excluded_agent_id=bot_agent_id,
        )

        if new_last_message_id:
            last_message_id = new_last_message_id

        time.sleep(1)

        for conversation_id, new_messages in conversation_dict.items():
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
                            send_new_message(
                                db_session=db_session, 
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
                mturk_chat_url = mturk_chat_url_template.replace('{{task_group_id}}', str(task_group_id)).replace('{{conversation_id}}', str(cid)).replace('{{cur_agent_id}}', str(worker_agent_id))
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

    while get_pending_approval_count(db_session, task_group_id) != num_hits:
        time.sleep(2)

    mturk_approval_url = mturk_approval_url_template.replace('{{task_group_id}}', str(task_group_id)).replace('{{cur_agent_id}}', str(worker_agent_id))
    print("\nAll HITs are done! Please go to the following link to approve/reject them (they will be auto-approved in 4 weeks if no action is taken):\n")
    print(mturk_approval_url)
    print("")

    approval_status_dict = {cid: '' for cid in cids}
    # Loop for checking approval status
    while get_pending_approval_count(db_session, task_group_id) > 0:
        time.sleep(2)

    print("Approvals are done!")

    for hit_info in get_all_approval_status(db_session, task_group_id):
        conversation_id = hit_info.conversation_id
        approval_status_dict[conversation_id] = hit_info.approval_status

    logs_approved = {cid:log for (cid,log) in logs.items() if approval_status_dict[cid] == 'approved'}
    logs_rejected = {cid:log for (cid,log) in logs.items() if approval_status_dict[cid] == 'rejected'}

    # Saving logs to file
    # Log format: {conversation_id: [list of messages in the conversation]}
    mturk_log_path = opt['mturk_log_path']
    task_group_path = mturk_log_path + task_module_name + '_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '/'
    os.makedirs(task_group_path)
    with open(task_group_path+'approved.json', 'w') as file:
        file.write(json.dumps(logs_approved))
    with open(task_group_path+'rejected.json', 'w') as file:
        file.write(json.dumps(logs_rejected))

    print("All conversations are saved to "+opt['mturk_log_path']+" in JSON format.\n")
