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
            cur_agent_id = event['query']['cur_agent_id']
            assignment_id = event['query']['assignmentId'] # from mturk

            if assignment_id == 'ASSIGNMENT_ID_NOT_AVAILABLE':
                template_context['is_cover_page'] = True

                custom_cover_page = cur_agent_id + '_cover_page.html'
                if os.path.exists(custom_cover_page):
                    return _render_template(template_context, custom_cover_page)
                else:
                    return _render_template(template_context, 'cover_page.html')
            else:
                template_context['is_cover_page'] = False
                template_context['task_group_id'] = task_group_id
                template_context['hit_index'] = 'Pending'
                template_context['assignment_index'] = 'Pending'
                template_context['cur_agent_id'] = cur_agent_id
                template_context['frame_height'] = frame_height

                custom_index_page = cur_agent_id + '_index.html'
                if os.path.exists(custom_index_page):
                    return _render_template(template_context, custom_index_page)
                else:
                    return _render_template(template_context, 'mturk_index.html')

        except KeyError:
            raise

def get_hit_config(event, context):
    if event['method'] == 'GET':
        with open('hit_config.json', 'r') as hit_config_file:
            return json.loads(hit_config_file.read().replace('\n', ''))

def send_new_command(event, context):
    if event['method'] == 'POST':
        params = event['body']
        task_group_id = params['task_group_id']
        conversation_id = params['conversation_id']
        receiver_agent_id = params['receiver_agent_id']
        command = params['command']

        new_command_object = data_model.send_new_command(
            db_session=db_session, 
            task_group_id=task_group_id, 
            conversation_id=conversation_id, 
            receiver_agent_id=receiver_agent_id, 
            command=command
        )
        
        return json.dumps(data_model.object_as_dict(new_command_object))

def get_command(event, context):
    if event['method'] == 'GET':
        task_group_id = event['query']['task_group_id']
        conversation_id = event['query']['conversation_id']
        receiver_agent_id = event['query']['receiver_agent_id']
        last_command_id = int(event['query']['last_command_id'])

        command_object = data_model.get_command(
            db_session=db_session, 
            task_group_id=task_group_id, 
            conversation_id=conversation_id, 
            receiver_agent_id=receiver_agent_id, 
            after_command_id=last_command_id
        )
         
        if command_object:   
            return json.dumps(data_model.object_as_dict(command_object))
        else:
            return None

def get_new_messages(event, context):
    if event['method'] == 'GET':
        """
        return messages as JSON
        Expects in GET query parameters:
        <task_group_id>
        <last_message_id>
        <receiver_agent_id>
        <conversation_id> (optional)
        <excluded_sender_agent_id> (optional)
        """
        task_group_id = event['query']['task_group_id']
        last_message_id = int(event['query']['last_message_id'])
        receiver_agent_id = event['query']['receiver_agent_id']
        conversation_id = None
        if 'conversation_id' in event['query']:
            conversation_id = event['query']['conversation_id']
        excluded_sender_agent_id = event['query'].get('excluded_sender_agent_id', None)
        included_sender_agent_id = event['query'].get('included_sender_agent_id', None)

        conversation_dict, new_last_message_id = data_model.get_new_messages(
            db_session=db_session, 
            task_group_id=task_group_id, 
            receiver_agent_id=receiver_agent_id,
            conversation_id=conversation_id,
            after_message_id=last_message_id,
            excluded_sender_agent_id=excluded_sender_agent_id,
            included_sender_agent_id=included_sender_agent_id,
            populate_meta_info=True
        )

        ret = {}
        ret['last_message_id'] = new_last_message_id
        ret['conversation_dict'] = conversation_dict

        for cid, message_list in conversation_dict.items():
            message_list.sort(key=lambda x: x['message_id'])
            
        return json.dumps(ret)

def send_new_message(event, context):
    if event['method'] == 'POST':
        params = event['body']
        task_group_id = params['task_group_id']
        conversation_id = params['conversation_id']
        sender_agent_id = params['sender_agent_id']
        receiver_agent_id = params['receiver_agent_id'] if 'receiver_agent_id' in params else None
        message_text = params['text'] if 'text' in params else None
        reward = params['reward'] if 'reward' in params else None
        episode_done = params['episode_done']

        new_message_object = data_model.send_new_message(
            db_session=db_session, 
            task_group_id=task_group_id, 
            conversation_id=conversation_id, 
            sender_agent_id=sender_agent_id, 
            receiver_agent_id=receiver_agent_id,
            message_text=message_text, 
            reward=reward,
            episode_done=episode_done
        )

        new_message = { 
            "message_id": new_message_object.id,
            "id": sender_agent_id,
            "text": message_text,
        }
        if reward:
            new_message['reward'] = reward
        new_message['episode_done'] = episode_done
        
        return json.dumps(new_message)


def sync_hit_assignment_info(event, context):
    if event['method'] == 'POST':
        """
        Handler for syncing HIT assignment info between webpage client and remote database.
        """
        try:
            params = event['body']
            task_group_id = params['task_group_id']
            agent_id = params['agent_id']
            num_assignments = params['num_assignments']
            assignment_id = params['assignment_id']
            hit_id = params['hit_id']
            worker_id = params['worker_id']

            return data_model.sync_hit_assignment_info(
                db_session=db_session,
                task_group_id=task_group_id,
                agent_id=agent_id,
                num_assignments=int(num_assignments),
                assignment_id=assignment_id,
                hit_id=hit_id,
                worker_id=worker_id
            )
        except KeyError:
            raise

def get_hit_assignment_info(event, context):
    if event['method'] == 'GET':
        """
        Handler for getting HIT assignment info.
        """
        try:
            task_group_id = event['query']['task_group_id']
            agent_id = event['query']['agent_id']
            conversation_id = event['query']['conversation_id']

            return json.dumps(data_model.get_hit_assignment_info(
                db_session=db_session,
                task_group_id=task_group_id,
                agent_id=agent_id,
                conversation_id=conversation_id
            ))
        except KeyError:
            raise

