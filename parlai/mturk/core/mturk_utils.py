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
aws_profile_name = 'parlai_mturk'

parent_dir = os.path.dirname(os.path.abspath(__file__))
mturk_hit_frame_height = 650


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


def calculate_mturk_cost(payment_opt):
    """MTurk Pricing: https://requester.mturk.com/pricing
    20% fee on the reward and bonus amount (if any) you pay Workers.
    HITs with 10 or more assignments will be charged an additional
    20% fee on the reward you pay Workers.

    Example payment_opt format for paying reward:
    {
        'type': 'reward',
        'num_total_assignments': 1,
        'reward': 0.05  # in dollars
        'unique': False # Unique workers requires multiple assignments to 1 HIT
    }

    Example payment_opt format for paying bonus:
    {
        'type': 'bonus',
        'amount': 1000  # in dollars
    }
    """
    total_cost = 0
    if payment_opt['type'] == 'reward':
        mult = 1.2
        if payment_opt['unique'] and payment_opt['num_total_assignments'] > 10:
            mult = 1.4
        total_cost = \
            payment_opt['num_total_assignments'] * payment_opt['reward'] * mult
    elif payment_opt['type'] == 'bonus':
        total_cost = payment_opt['amount'] * 1.2
    return total_cost


def check_mturk_balance(balance_needed, is_sandbox):
    """Checks to see if there is at least balance_needed amount in the
    requester account, returns True if the balance is greater than
    balance_needed
    """
    client = boto3.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )

    # Region is always us-east-1
    if not is_sandbox:
        client = boto3.client(service_name='mturk', region_name='us-east-1')

    # Test that you can connect to the API by checking your account balance
    # In Sandbox this always returns $10,000
    try:
        user_balance = float(client.get_account_balance()['AvailableBalance'])
    except ClientError as e:
        if e.response['Error']['Code'] == 'RequestError':
            print(
                'ERROR: To use the MTurk API, you will need an Amazon Web '
                'Services (AWS) Account. Your AWS account must be linked to '
                'your Amazon Mechanical Turk Account. Visit '
                'https://requestersandbox.mturk.com/developer to get started. '
                '(Note: if you have recently linked your account, please wait '
                'for a couple minutes before trying again.)\n'
            )
            quit()
        else:
            raise

    if user_balance < balance_needed:
        print(
            'You might not have enough money in your MTurk account. Please go '
            'to https://requester.mturk.com/account and increase your balance '
            'to at least ${}, and then try again.'.format(balance_needed))
        return False
    else:
        return True


def create_hit_config(task_description, unique_worker, is_sandbox):
    """Writes a HIT config to file"""
    mturk_submit_url = 'https://workersandbox.mturk.com/mturk/externalSubmit'
    if not is_sandbox:
        mturk_submit_url = 'https://www.mturk.com/mturk/externalSubmit'
    hit_config = {
        'task_description': task_description,
        'is_sandbox': is_sandbox,
        'mturk_submit_url': mturk_submit_url,
        'unique_worker': unique_worker,
        'frame_height': mturk_hit_frame_height
    }
    hit_config_file_path = os.path.join(parent_dir, 'hit_config.json')
    if os.path.exists(hit_config_file_path):
        os.remove(hit_config_file_path)
    with open(hit_config_file_path, 'w') as hit_config_file:
        hit_config_file.write(json.dumps(hit_config))


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


def delete_qualification(qualification_id):
    """Deletes a qualification by id"""
    client = boto3.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )
    client.delete_qualification_type(
        QualificationTypeId=qualification_id
    )


def find_qualification(qualification_name, must_be_owned=True):
    """Query amazon to find the existing qualification name, return the Id,
    otherwise return none.
    If must_be_owned is true, it only returns qualifications owned by the user.
    Will return False if it finds another's qualification
    """
    client = boto3.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )

    # Search for the qualification owned by the current user
    response = client.list_qualification_types(
        Query=qualification_name,
        MustBeRequestable=True,
        MustBeOwnedByCaller=True,
    )

    for qualification in response['QualificationTypes']:
        if qualification['Name'] == qualification_name:
            return qualification['QualificationTypeId']

    # Qualification was not found to exist, check to see if someone else has it
    response = client.list_qualification_types(
        Query=qualification_name,
        MustBeRequestable=True,
        MustBeOwnedByCaller=False,
    )

    for qualification in response['QualificationTypes']:
        if qualification['Name'] == qualification_name:
            if must_be_owned:
                print(
                    'Sorry, the qualification name {} is already owned, '
                    'please use a different name for your qualification.'
                    ''.format(qualification_name)
                )
                return False
            return qualification['QualificationTypeId']
    return None


def find_or_create_qualification(qualification_name, description,
                                 must_be_owned=True):
    """Query amazon to find the existing qualification name, return the Id. If
    it exists and must_be_owned is true but we don't own it, this prints an
    error and returns none. If it doesn't exist, the qualification is created
    """
    qual_id = find_qualification(
        qualification_name,
        must_be_owned=must_be_owned
    )

    if qual_id is False:
        return None
    if qual_id is not None:
        return qual_id

    # Create the qualification, as it doesn't exist yet
    client = boto3.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )
    response = client.create_qualification_type(
        Name=qualification_name,
        Description=description,
        QualificationTypeStatus='Active',
    )
    return response['QualificationType']['QualificationTypeId']


def give_worker_qualification(worker_id, qualification_id, value=None):
    """Give a qualification to the given worker"""
    client = boto3.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )

    if value is not None:
        client.associate_qualification_with_worker(
            QualificationTypeId='string',
            WorkerId='string',
            IntegerValue=value,
            SendNotification=False
        )
    else:
        client.associate_qualification_with_worker(
            QualificationTypeId='string',
            WorkerId='string',
            SendNotification=False
        )


def create_hit_type(hit_title, hit_description, hit_keywords, hit_reward,
                    assignment_duration_in_seconds, is_sandbox,
                    qualifications=None):
    """Create a HIT type to be used to generate HITs of the requested params"""
    client = boto3.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )

    # Region is always us-east-1
    if not is_sandbox:
        client = boto3.client(service_name='mturk', region_name='us-east-1')

    # Create a qualification with Locale In('US', 'CA') requirement attached
    localRequirements = [{
        'QualificationTypeId': '00000000000000000071',
        'Comparator': 'In',
        'LocaleValues': [
            {'Country': 'US'},
            {'Country': 'CA'},
            {'Country': 'GB'},
            {'Country': 'AU'},
            {'Country': 'NZ'}
        ],
        'RequiredToPreview': True
    }]
    if qualifications is not None:
        localRequirements += qualifications

    # Create the HIT type
    response = client.create_hit_type(
        AutoApprovalDelayInSeconds=4*7*24*3600,  # auto-approve after 4 weeks
        AssignmentDurationInSeconds=assignment_duration_in_seconds,
        Reward=str(hit_reward),
        Title=hit_title,
        Keywords=hit_keywords,
        Description=hit_description,
        QualificationRequirements=localRequirements
    )
    hit_type_id = response['HITTypeId']
    return hit_type_id


def create_hit_with_hit_type(page_url, hit_type_id, num_assignments,
                             is_sandbox):
    """Creates the actual HIT given the type and page to direct clients to"""
    page_url = page_url.replace('&', '&amp;')
    amazon_ext_url = (
        'http://mechanicalturk.amazonaws.com/'
        'AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd'
    )
    question_data_struture = (
        '<ExternalQuestion xmlns="{}">'
            '<ExternalURL>{}</ExternalURL>'  # noqa: E131
            '<FrameHeight>{}</FrameHeight>'
        '</ExternalQuestion>'
        ''.format(amazon_ext_url, page_url, mturk_hit_frame_height)
    )

    client = boto3.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )

    # Region is always us-east-1
    if not is_sandbox:
        client = boto3.client(service_name='mturk', region_name='us-east-1')

    # Create the HIT
    response = client.create_hit_with_hit_type(
        HITTypeId=hit_type_id,
        MaxAssignments=num_assignments,
        LifetimeInSeconds=31536000,
        Question=question_data_struture,
    )

    # The response included several fields that will be helpful later
    hit_type_id = response['HIT']['HITTypeId']
    hit_id = response['HIT']['HITId']

    # Construct the hit URL
    url_target = 'workersandbox'
    if not is_sandbox:
        url_target = 'www'
    hit_link = 'https://{}.mturk.com/mturk/preview?groupId={}'.format(
        url_target,
        hit_type_id
    )
    return hit_link, hit_id


def expire_hit(is_sandbox, hit_id):
    client = get_mturk_client(is_sandbox)
    # Update expiration to a time in the past, the HIT expires instantly
    past_time = datetime(2015, 1, 1)
    client.update_expiration_for_hit(HITId=hit_id, ExpireAt=past_time)


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


def subscribe_to_hits(hit_type_id, is_sandbox, sns_arn):
    # Get the mturk client and create notifications for our hits
    client = get_mturk_client(is_sandbox)
    client.update_notification_settings(
        HITTypeId=hit_type_id,
        Notification={
            'Destination': sns_arn,
            'Transport': 'SNS',
            'Version': '2006-05-05',
            'EventTypes': ['AssignmentAbandoned', 'AssignmentReturned',
                           'AssignmentSubmitted']
        },
        Active=True
    )


def send_test_notif(topic_arn, event_type):
    client = get_mturk_client(True)
    client.send_test_event_notification(
        Notification={
            'Destination': topic_arn,
            'Transport': 'SNS',
            'Version': '2006-05-05',
            'EventTypes': ['AssignmentAbandoned', 'AssignmentReturned',
                           'AssignmentSubmitted']
        },
        TestEventType=event_type
    )


def delete_sns_topic(topic_arn):
    client = boto3.client('sns', region_name='us-east-1',)
    client.delete_topic(TopicArn=topic_arn)
