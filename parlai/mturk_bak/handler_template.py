# Copyright 2004-present Facebook. All Rights Reserved.
# Python 2

from __future__ import print_function

import os
import time
import json
import boto3
import calendar
import requests
from jinja2 import Environment
from jinja2 import FileSystemLoader
from data_model import Message, init_database, send_new_message, get_new_messages

# Dynamically generated code begin
# Expects rds_host, rds_db_name, rds_username, rds_password, agent_display_names, task_description, state_config
# {{block_task_config}}
# Dynamically generated code end

db_engine, db_session_maker = init_database(rds_host, rds_db_name, rds_username, rds_password)
db_session = db_session_maker()


def _render_template(template_context, template_file_name):
    env = Environment(loader=FileSystemLoader(os.path.abspath(os.path.dirname(__file__))))
    template = env.get_template(template_file_name)
    rendered_template = template.render(template_context)
    return rendered_template


def lambda_handler(event, context):
    params = None
    if event['method'] == 'GET':
        params = event['query']
    elif event['method'] == 'POST':
        params = event['body']

    if params['endpoint'] == 'index':
        return index(event, context)
    elif params['endpoint'] == 'message':
        return message(event, context)
    else:
        return None


def index(event, context):
    """
    Handler for chat page endpoint. 
    Expects <task_group_id>, <conversation_id> and <cur_agent_id> as query parameters.
    """
    template_context = {}

    try:
        task_group_id = event['query']['task_group_id']
        conversation_id = event['query']['conversation_id']
        cur_agent_id = event['query']['cur_agent_id']

        template_context['task_group_id'] = task_group_id
        template_context['conversation_id'] = conversation_id
        template_context['cur_agent_id'] = cur_agent_id
        template_context['agent_display_names_string'] = json.dumps(agent_display_names)
        template_context['task_description'] = task_description
        template_context['state_config_string'] = json.dumps(state_config)
        template_context['debug_log'] = None

        return _render_template(template_context, 'mturk_index.html')

    except KeyError:
        raise Exception('400')


def message(event, context):
    if event['method'] == 'GET':
        """
        return all new message from all other agents as JSON
        Expects <task_group_id>, <conversation_id> and <last_message_id> as GET body parameters
        """
        task_group_id = event['query']['task_group_id']
        conversation_id = int(event['query']['conversation_id'])
        last_message_id = int(event['query']['last_message_id'])

        conversation_dict, _ = get_new_messages(
            db_session=db_session, 
            task_group_id=task_group_id, 
            conversation_id=conversation_id,
            after_message_id=last_message_id,
            populate_state_info=True
        )

        ret = []
        if conversation_id in conversation_dict:
            ret = conversation_dict[conversation_id]
            ret.sort(key=lambda x: x['timestamp'])
            
        return json.dumps(ret)
    if event['method'] == 'POST':
        """
        Send new message for this agent.
        Expects <task_group_id>, <conversation_id>, <cur_agent_id> and <msg> as POST body parameters
        """
        params = event['body']
        task_group_id = params['task_group_id']
        conversation_id = int(params['conversation_id'])
        cur_agent_id = params['cur_agent_id']
        message_text = params['msg'] if 'msg' in params else None
        reward = params['reward'] if 'reward' in params else None
        done = params['done']

        new_message_object = send_new_message(
            db_session=db_session, 
            task_group_id=task_group_id, 
            conversation_id=conversation_id, 
            agent_id=cur_agent_id, 
            message_text=message_text, 
            reward=reward,
            done=done,
            binary_file_bytes=None, 
            binary_file_type=None
        )

        new_message = { 
            "message_id": new_message_object.id,
            "id": cur_agent_id,
            "text": message_text,
            "timestamp": time.mktime(new_message_object.created_time.timetuple()) + new_message_object.created_time.microsecond * 1e-6,
        }
        if reward:
            new_message['reward'] = reward    
        new_message['done'] = done
        
        return json.dumps(new_message)
