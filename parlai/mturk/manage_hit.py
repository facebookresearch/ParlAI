# Copyright 2004-present Facebook. All Rights Reserved.
"""Contains basic functionality for setting up a simple MTurk bot evaluation.
In this example, a bot will be paired with a human, given the default
instructions and opening message, and then will chat with the bot.
"""

import time
import random
import string
import webbrowser
from parlai.core.agents import create_agent_from_shared
from setup_aws import rds_db_name, rds_username, rds_password, setup_aws, create_hit_type, create_hit_with_hit_type
from data_model import Message, init_database, send_new_message, get_new_messages


def _get_random_alphanumeric_string(N):
    return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(N))


def setup_relay(task_config):
    """Sets up relay server and returns a database session object which can poll
    new messages and send messages
    """
    # set up relay server
    rds_host, mturk_chat_url_template = setup_aws(task_config)

    db_engine, db_session_maker = init_database(rds_host, rds_db_name, rds_username, rds_password)
    db_session = db_session_maker()

    return db_session, mturk_chat_url_template


def create_hits(task_config, bot, num_hits, is_sandbox, chat_page_only, verbose):
    task_group_id = str(int(time.time())) + '_' + _get_random_alphanumeric_string(10) # Random string to further avoid collision
    print('Setting up MTurk backend...')
    db_session, mturk_chat_url_template = setup_relay(task_config)

    worker_agent_id = task_config['worker_agent_id']    
    cids = range(1, num_hits+1)

    shared = bot.share()
    bots = []
    for cid in cids:
        new_bot = create_agent_from_shared(shared)
        new_bot.conversation_id = cid
        bots.append(new_bot)
        response = agent.act()  # Assuming agent returns None if it's still expecting more messages
        if response:
            if verbose:
                print('Bot ' + str(conversation_id) + 'response: ' + response)
            send_new_message(
                db_session=db_session, 
                task_group_id=task_group_id, 
                conversation_id=conversation_id, 
                agent_id=task_config['bot_agent_id'], 
                message_text=response.get('text', None), 
                reward=response.get('reward', None), 
                action=response.get('action', None), 
                episode_done=response.get('episode_done', False), 
            )

    cid_map = {cid: i for i, cid in enumerate(cids)}

    hits_created = False
    conversations_remaining = set(cids)
    last_message_id = -1

    while len(conversations_remaining) > 0:
        conversation_dict, new_last_message_id = get_new_messages(
            db_session=db_session, 
            task_group_id=task_group_id, 
            after_message_id=last_message_id, 
            excluded_agent_id=task_config['bot_agent_id'],
        )

        if new_last_message_id:
            last_message_id = new_last_message_id

        time.sleep(1)

        for conversation_id, new_messages in conversation_dict.items():
            if conversation_id in conversations_remaining and len(new_messages) > 0:
                agent = bots[cid_map[conversation_id]]
                for new_message in new_messages:
                    # observe could be in the else block?
                    if verbose:
                        print('Bot ' + str(conversation_id) + 'received: ' + new_message)
                    agent.observe(new_message)
                    if new_message.get('done', False):
                        # We're done here
                        conversations_remaining.remove(conversation_id)
                        print('Conversation '+str(conversation_id)+' is DONE!')
                    else:
                        # Agent still needs to reply
                        response = agent.act()  # Assuming agent returns None if it's still expecting more messages
                        if response:
                            if verbose:
                                print('Bot ' + str(conversation_id) + 'response: ' + response)
                            send_new_message(
                                db_session=db_session, 
                                task_group_id=task_group_id, 
                                conversation_id=conversation_id, 
                                agent_id=task_config['bot_agent_id'], 
                                message_text=response.get('text', None), 
                                reward=response.get('reward', None), 
                                action=response.get('action', None), 
                                episode_done=response.get('episode_done', False), 
                            )

        if not hits_created:
            print('Creating HITs...')
            hit_type_id = create_hit_type(
                hit_title=task_config['hit_title'], 
                hit_description=task_config['hit_description'] + ' (ID: ' + task_group_id + ')', 
                hit_keywords=task_config['hit_keywords'], 
                hit_reward=task_config['hit_reward'], 
                num_hits=num_hits,
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
                        is_sandbox=True
                    )

            print('MTurk setup done. Waiting for Turkers to work on the tasks...')
            if chat_page_only:
                webbrowser.open(mturk_chat_url)
            else:
                webbrowser.open(mturk_page_url)
            hits_created = True

