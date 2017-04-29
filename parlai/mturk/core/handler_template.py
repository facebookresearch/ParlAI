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
from data_model import Message, MTurkHITInfo, init_database, send_new_message, get_new_messages, set_hit_info, get_hit_info

# Dynamically generated code begin
# Expects mturk_submit_url, rds_host, rds_db_name, rds_username, rds_password, task_description, requester_key_gt, num_hits, is_sandbox
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
    elif params['endpoint'] == 'approval':
        return approval(event, context)
    else:
        return None


def index(event, context):
    if event['method'] == 'GET':
        """
        Handler for chat page endpoint. 
        Expects <task_group_id>, <conversation_id> and <cur_agent_id> as query parameters.
        """
        template_context = {}

        try:
            task_group_id = event['query']['task_group_id']
            conversation_id = event['query']['conversation_id']
            cur_agent_id = event['query']['cur_agent_id']
            assignment_id = event['query']['assignmentId'] # from mturk

            if assignment_id == 'ASSIGNMENT_ID_NOT_AVAILABLE':
                template_context['task_description'] = task_description
                template_context['is_cover_page'] = True
            else:
                template_context['task_group_id'] = task_group_id
                template_context['conversation_id'] = conversation_id
                template_context['cur_agent_id'] = cur_agent_id
                template_context['task_description'] = task_description
                template_context['mturk_submit_url'] = mturk_submit_url
                template_context['is_cover_page'] = False

            return _render_template(template_context, 'mturk_index.html')

        except KeyError:
            raise Exception('400')

    elif event['method'] == 'POST':
        """
        Saves HIT info to DB.
        Expects <task_group_id>, <conversation_id>, <assignmentId>, <hitId>, <workerId> as POST body parameters
        """
        params = event['body']
        task_group_id = params['task_group_id']
        conversation_id = int(params['conversation_id'])
        assignment_id = params['assignmentId']
        hit_id = params['hitId']
        worker_id = params['workerId']

        set_hit_info(
            db_session = db_session, 
            task_group_id = task_group_id, 
            conversation_id = conversation_id, 
            assignment_id = assignment_id, 
            hit_id = hit_id, 
            worker_id = worker_id,
            is_sandbox = is_sandbox
        )


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
            populate_meta_info=True
        )

        ret = []
        if conversation_id in conversation_dict:
            ret = conversation_dict[conversation_id]
            ret.sort(key=lambda x: x['message_id'])
            
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
        episode_done = params['episode_done']

        new_message_object = send_new_message(
            db_session=db_session, 
            task_group_id=task_group_id, 
            conversation_id=conversation_id, 
            agent_id=cur_agent_id, 
            message_text=message_text, 
            reward=reward,
            episode_done=episode_done,
            binary_file_bytes=None, 
            binary_file_type=None
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


def approval(event, context):
    if event['method'] == 'GET':
        """
        Handler for approval page endpoint. 
        Expects <requester_key>, <task_group_id>, <conversation_id>, <cur_agent_id> as query parameters.
        """
        try:
            requester_key = event['query']['requester_key']
            if not requester_key == requester_key_gt:
                raise Exception('403')

            task_group_id = event['query']['task_group_id']
            conversation_id = event['query']['conversation_id']
            cur_agent_id = event['query']['cur_agent_id']

            template_context = {}
            template_context['task_group_id'] = task_group_id
            template_context['conversation_id'] = conversation_id
            template_context['cur_agent_id'] = cur_agent_id
            template_context['task_description'] = task_description
            template_context['is_cover_page'] = False
            template_context['is_approval_page'] = True
            template_context['num_hits'] = int(num_hits)

            return _render_template(template_context, 'mturk_index.html')

        except KeyError:
            raise Exception('400')

    elif event['method'] == 'POST':
        """
        Approve or reject assignment.
        Expects <requester_key>, <task_group_id>, <conversation_id>, <action> as POST body parameters
        """
        try:
            params = event['body']
            requester_key = params['requester_key']
            if not requester_key == requester_key_gt:
                raise Exception('403')

            task_group_id = params['task_group_id']
            conversation_id = int(params['conversation_id'])
            action = params['action'] # 'approve' or 'reject'

            hit_info = get_hit_info(
                db_session=db_session, 
                task_group_id=task_group_id, 
                conversation_id=conversation_id
            )

            if hit_info:
                assignment_id = hit_info.assignment_id
                client = boto3.client(
                    service_name = 'mturk', 
                    region_name = 'us-east-1',
                    endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
                )
                # Region is always us-east-1
                if not hit_info.is_sandbox:
                    client = boto3.client(service_name = 'mturk', region_name='us-east-1')

                if action == 'approve':
                    client.approve_assignment(AssignmentId=assignment_id)
                elif action == 'reject':
                    client.reject_assignment(AssignmentId=assignment_id, RequesterFeedback='')

        except KeyError:
            raise Exception('400')
