# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import boto3
import os
import json
from datetime import datetime

from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound

region_name = 'us-east-1'
# TODO make this parlai_aws and do the same for the mturk stuff?
aws_profile_name = 'parlai_mturk'

parent_dir = os.path.dirname(os.path.abspath(__file__))


def setup_aws_credentials():
    try:
        # Use existing credentials
        boto3.Session(profile_name=aws_profile_name)
    except ProfileNotFound:
        # Setup new credentials
        print(
            'AWS credentials not found. Please create an IAM user with '
            'programmatic access and AdministratorAccess policy at '
            'https://console.aws.amazon.com/iam/ (On the "Set permissions" '
            'page, choose "Attach existing policies directly" and then select '
            '"AdministratorAccess" policy). After creating the IAM user, '
            'please enter the user\'s Access Key ID and Secret Access '
            'Key below:'
        )
        aws_access_key_id = input('Access Key ID: ')
        aws_secret_access_key = input('Secret Access Key: ')
        if not os.path.exists(os.path.expanduser('~/.aws/')):
            os.makedirs(os.path.expanduser('~/.aws/'))
        aws_credentials_file_path = '~/.aws/credentials'
        aws_credentials_file_string = None
        expanded_aws_file_path = os.path.expanduser(aws_credentials_file_path)
        if os.path.exists(expanded_aws_file_path):
            with open(expanded_aws_file_path, 'r') as aws_credentials_file:
                aws_credentials_file_string = aws_credentials_file.read()
        with open(expanded_aws_file_path, 'a+') as aws_credentials_file:
            # Clean up file
            if aws_credentials_file_string:
                if aws_credentials_file_string.endswith("\n\n"):
                    pass
                elif aws_credentials_file_string.endswith("\n"):
                    aws_credentials_file.write("\n")
                else:
                    aws_credentials_file.write("\n\n")
            # Write login details
            aws_credentials_file.write('[{}]\n'.format(aws_profile_name))
            aws_credentials_file.write(
                'aws_access_key_id={}\n'.format(aws_access_key_id)
            )
            aws_credentials_file.write(
                'aws_secret_access_key={}\n'.format(aws_secret_access_key)
            )
        print('AWS credentials successfully saved in {} file.\n'.format(
            aws_credentials_file_path
        ))
    os.environ['AWS_PROFILE'] = aws_profile_name


def get_mturk_client(is_sandbox):
    """Returns the appropriate mturk client given sandbox option"""
    client = boto3.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )
    # Region is always us-east-1
    if not is_sandbox:
        client = boto3.client(service_name='mturk', region_name='us-east-1')
    return client


def setup_sns_topic(task_name, server_url, task_group_id):
    # Create the topic and subscribe to it so that our server receives notifs
    client = boto3.client('sns', region_name='us-east-1',)
    response = client.create_topic(Name=task_name)
    arn = response['TopicArn']
    topic_sub_url = \
        '{}/sns_posts?task_group_id={}'.format(server_url, task_group_id)
    client.subscribe(TopicArn=arn, Protocol='https', Endpoint=topic_sub_url)
    response = client.get_topic_attributes(
        TopicArn=arn
    )
    policy_json = '''{{
    "Version": "2008-10-17",
    "Id": "{}/MTurkOnlyPolicy",
    "Statement": [
        {{
            "Sid": "MTurkOnlyPolicy",
            "Effect": "Allow",
            "Principal": {{
                "Service": "mturk-requester.amazonaws.com"
            }},
            "Action": "SNS:Publish",
            "Resource": "{}"
        }}
    ]}}'''.format(arn, arn)
    client.set_topic_attributes(
        TopicArn=arn,
        AttributeName='Policy',
        AttributeValue=policy_json
    )
    return arn
