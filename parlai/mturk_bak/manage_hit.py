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


def setup_context(db_session, data_loader, task_group_id, conversation_id):
    context_dict = data_loader.load_context(conversation_id)
    # TODO: address the multiple fields case
    context_text = None
    context_question = None
    if '\n' in context_dict['text']:
        comp_list = context_dict['text'].split('\n')
        context_text = comp_list[0]
        context_question = comp_list[1]
    else:
        context_text = context_dict['text']
    if context_text:
        send_new_message(
            db_session = db_session,
            task_group_id = task_group_id, 
            conversation_id = conversation_id,
            agent_id = 'context',
            message_text=context_text, 
            done=False,
            binary_file_bytes=None, 
            binary_file_type=None
        )
    if context_question:
        send_new_message(
            db_session = db_session,
            task_group_id = task_group_id, 
            conversation_id = conversation_id,
            agent_id = 'teacher',
            message_text=context_question, 
            done=False,
            binary_file_bytes=None, 
            binary_file_type=None
        )


def create_hits(opt, task_config, data_loader, bot, num_hits, is_sandbox, chat_page_only):
    # shared = bot.share(opt=opt)
    # bots = [create_agent_from_shared(shared) for _ in range(num_hits)]
    # TODO: unable to import create_agent_from_shared, need to fix
    bots = [bot]

    task_group_id = str(int(time.time())) + '_' + _get_random_alphanumeric_string(10) # Random string to further avoid collision
    print('Initializing MTurk...')
    db_session, mturk_chat_url_template = setup_relay(task_config)
    print('MTurk initialization done. Opening web interface...')
    print('')

    worker_agent_id = task_config['worker_agent_id']    
    cids = range(1, num_hits+1)
    hit_type_id = create_hit_type(
        hit_title=task_config['hit_title'], 
        hit_description=task_config['hit_description'] + ' (ID: ' + task_group_id + ')', 
        hit_keywords=task_config['hit_keywords'], 
        hit_reward=task_config['hit_reward'], 
        num_hits=num_hits,
        is_sandbox=is_sandbox
    )
    for cid in cids:
        mturk_chat_url = mturk_chat_url_template.replace('{{task_group_id}}', str(task_group_id)).replace('{{conversation_id}}', str(cid)).replace('{{cur_agent_id}}', str(worker_agent_id))
        if chat_page_only:
            webbrowser.open(mturk_chat_url)
        else:
            mturk_page_url = create_hit_with_hit_type(
                page_url=mturk_chat_url, 
                hit_type_id=hit_type_id, 
                is_sandbox=True
            )
    if not chat_page_only:
        webbrowser.open(mturk_page_url)

    cid_map = {cid: i for i, cid in enumerate(cids)}
    conversations_remaining = set(cids)

    # Set up opening messages
    for cid in cids:
        setup_context(db_session, data_loader, task_group_id, cid)
    
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
                    agent.observe(new_message)
                    if new_message.get('done', False):
                        # We're done here
                        conversations_remaining.remove(conversation_id)
                        print('Conversation '+str(conversation_id)+' is DONE!')
                    else:
                        # Agent still needs to reply
                        response, action = agent.act()  # Assuming agent returns None if it's still expecting more messages
                        if response:
                            send_new_message(
                                db_session=db_session, 
                                task_group_id=task_group_id, 
                                conversation_id=conversation_id, 
                                agent_id=task_config['bot_agent_id'], 
                                message_text=response,
                                action=action,
                                done=False
                            )

