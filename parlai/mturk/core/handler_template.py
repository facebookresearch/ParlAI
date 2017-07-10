# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Python 2
from __future__ import print_function

import os
import time
import json
import boto3
import calendar
from jinja2 import Environment
from jinja2 import FileSystemLoader
import data_model

# Dynamically generated code begin
# Expects mturk_submit_url, frame_height, rds_host, rds_db_name, rds_username, rds_password
# {{block_task_config}}
# Dynamically generated code end

data_model.setup_database_engine(rds_host, rds_db_name, rds_username, rds_password)
db_engine, db_session_maker = data_model.init_database()
db_session = db_session_maker()


def _render_template(template_context, template_file_name):
    env = Environment(loader=FileSystemLoader(os.path.abspath(os.path.dirname(__file__))))
    template = env.get_template(template_file_name)
    rendered_template = template.render(template_context)
    return rendered_template

def lambda_handler(event, context):
    global db_engine, db_session

    params = None
    if event['method'] == 'GET':
        params = event['query']
    elif event['method'] == 'POST':
        params = event['body']

    method_name = params['method_name']
    if method_name in globals():
        result = globals()[method_name](event, context)
        data_model.close_connection(db_engine, db_session)
        return result

def chat_index(event, context):
    if event['method'] == 'GET':
        """
        Handler for chat page endpoint. 
        """
        template_context = {}

        try:
            task_group_id = event['query']['task_group_id']
            hit_index = event['query'].get('hit_index', 'Pending')
            assignment_index = event['query'].get('assignment_index', 'Pending')
            all_agent_ids = event['query']['all_agent_ids']
            cur_agent_id = event['query']['cur_agent_id']
            assignment_id = event['query']['assignmentId'] # from mturk

            if assignment_id == 'ASSIGNMENT_ID_NOT_AVAILABLE':
                custom_cover_page = cur_agent_id + '_cover_page.html'
                if os.path.exists(custom_cover_page):
                    return _render_template(template_context, custom_cover_page)
                else:
                    return _render_template(template_context, 'cover_page.html')
            else:
                template_context['task_group_id'] = task_group_id
                template_context['hit_index'] = hit_index
                template_context['assignment_index'] = assignment_index
                template_context['cur_agent_id'] = cur_agent_id
                template_context['all_agent_ids'] = all_agent_ids
                template_context['frame_height'] = frame_height

                custom_index_page = cur_agent_id + '_index.html'
                if os.path.exists(custom_index_page):
                    return _render_template(template_context, custom_index_page)
                else:
                    return _render_template(template_context, 'mturk_index.html')

        except KeyError:
            raise Exception('400')

def get_hit_config(event, context):
    if event['method'] == 'GET':
        with open('hit_config.json', 'r') as hit_config_file:
            return json.loads(hit_config_file.read().replace('\n', ''))

def get_new_messages(event, context):
    if event['method'] == 'GET':
        """
        return messages as JSON
        Expects in GET query parameters:
        <task_group_id>
        <last_message_id>
        <conversation_id> (optional)
        <excluded_agent_id> (optional)
        """
        task_group_id = event['query']['task_group_id']
        last_message_id = int(event['query']['last_message_id'])
        conversation_id = None
        if 'conversation_id' in event['query']:
            conversation_id = event['query']['conversation_id']
        excluded_agent_id = event['query'].get('excluded_agent_id', None)
        included_agent_id = event['query'].get('included_agent_id', None)

        conversation_dict, new_last_message_id = data_model.get_new_messages(
            db_session=db_session, 
            task_group_id=task_group_id, 
            conversation_id=conversation_id,
            after_message_id=last_message_id,
            excluded_agent_id=excluded_agent_id,
            included_agent_id=included_agent_id,
            populate_meta_info=True,
            populate_hit_info=True
        )

        ret = {}
        ret['last_message_id'] = new_last_message_id
        ret['conversation_dict'] = conversation_dict

        for cid, message_list in conversation_dict.items():
            message_list.sort(key=lambda x: x['message_id'])
            
        return json.dumps(ret)

def send_new_message(event, context):
    if event['method'] == 'POST':
        """
        Send new message for this agent.
        Expects <task_group_id>, <conversation_id>, <cur_agent_id> and <text> as POST body parameters
        """
        params = event['body']
        task_group_id = params['task_group_id']
        conversation_id = params['conversation_id']
        cur_agent_id = params['cur_agent_id']
        message_text = params['text'] if 'text' in params else None
        reward = params['reward'] if 'reward' in params else None
        episode_done = params['episode_done']
        assignment_id = params['assignment_id']
        hit_id = params['hit_id']
        worker_id = params['worker_id']

        new_message_object = data_model.send_new_message(
            db_session=db_session, 
            task_group_id=task_group_id, 
            conversation_id=conversation_id, 
            agent_id=cur_agent_id, 
            message_text=message_text, 
            reward=reward,
            episode_done=episode_done,
            assignment_id=assignment_id,
            hit_id=hit_id,
            worker_id=worker_id
        )

        new_message = { 
            "message_id": new_message_object.id,
            "id": cur_agent_id,
            "text": message_text,
        }
        if reward:
            new_message['reward'] = reward
        new_message['episode_done'] = episode_done
        
        return json.dumps(new_message)

def send_new_messages_in_bulk(event, context):
    if event['method'] == 'POST':
        """
        Send new messages in bulk.
        Expects <new_messages> as POST body parameters
        """
        params = event['body']
        new_messages = params['new_messages']

        data_model.send_new_messages_in_bulk(
            db_session=db_session,
            new_messages=new_messages
        )

def get_hit_index_and_assignment_index(event, context):
    if event['method'] == 'GET':
        """
        Handler for get assignment index endpoint. 
        Expects <task_group_id>, <agent_id> as query parameters.
        """
        try:
            task_group_id = event['query']['task_group_id']
            agent_id = event['query']['agent_id']
            num_assignments = event['query']['num_assignments']

            return data_model.get_hit_index_and_assignment_index(
                db_session=db_session,
                task_group_id=task_group_id,
                agent_id=agent_id,
                num_assignments=int(num_assignments)
            )
        except KeyError:
            raise Exception('400')

