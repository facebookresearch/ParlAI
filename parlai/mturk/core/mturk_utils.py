import os
import sys
import shutil
from subprocess import call
import zipfile
import boto3
import botocore
import time
import json
import webbrowser
import hashlib
import getpass
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound

# TODO: remove unused imports

region_name = 'us-east-1'
aws_profile_name = 'parlai_mturk'

parent_dir = os.path.dirname(os.path.abspath(__file__))
mturk_hit_frame_height = 650

def setup_aws_credentials():
    try:
        session = boto3.Session(profile_name=aws_profile_name)
    except ProfileNotFound as e:
        print('''AWS credentials not found. Please create an IAM user with programmatic access and AdministratorAccess policy at https://console.aws.amazon.com/iam/ (On the "Set permissions" page, choose "Attach existing policies directly" and then select "AdministratorAccess" policy). \nAfter creating the IAM user, please enter the user's Access Key ID and Secret Access Key below:''')
        aws_access_key_id = input('Access Key ID: ')
        aws_secret_access_key = input('Secret Access Key: ')
        if not os.path.exists(os.path.expanduser('~/.aws/')):
            os.makedirs(os.path.expanduser('~/.aws/'))
        aws_credentials_file_path = '~/.aws/credentials'
        aws_credentials_file_string = None
        if os.path.exists(os.path.expanduser(aws_credentials_file_path)):
            with open(os.path.expanduser(aws_credentials_file_path), 'r') as aws_credentials_file:
                aws_credentials_file_string = aws_credentials_file.read()
        with open(os.path.expanduser(aws_credentials_file_path), 'a+') as aws_credentials_file:
            if aws_credentials_file_string:
                if aws_credentials_file_string.endswith("\n\n"):
                    pass
                elif aws_credentials_file_string.endswith("\n"):
                    aws_credentials_file.write("\n")
                else:
                    aws_credentials_file.write("\n\n")
            aws_credentials_file.write("["+aws_profile_name+"]\n")
            aws_credentials_file.write("aws_access_key_id="+aws_access_key_id+"\n")
            aws_credentials_file.write("aws_secret_access_key="+aws_secret_access_key+"\n")
        print("AWS credentials successfully saved in "+aws_credentials_file_path+" file.\n")
    os.environ["AWS_PROFILE"] = aws_profile_name

def calculate_mturk_cost(payment_opt):
    """MTurk Pricing: https://requester.mturk.com/pricing
    20% fee on the reward and bonus amount (if any) you pay Workers.
    HITs with 10 or more assignments will be charged an additional 20% fee on the reward you pay Workers.

    Example payment_opt format for paying reward:
    {
        'type': 'reward',
        'num_total_assignments': 1,
        'reward': 0.05  # in dollars
    }

    Example payment_opt format for paying bonus:
    {
        'type': 'bonus',
        'amount': 1000  # in dollars
    }
    """
    total_cost = 0
    if payment_opt['type'] == 'reward':
        total_cost = payment_opt['num_total_assignments'] * payment_opt['reward'] * 1.2
        if payment_opt['num_total_assignments'] >= 10:
            total_cost = total_cost * 1.2
    elif payment_opt['type'] == 'bonus':
        total_cost = payment_opt['amount'] * 1.2
    return total_cost

def check_mturk_balance(balance_needed, is_sandbox):
    client = boto3.client(
        service_name = 'mturk',
        region_name = 'us-east-1',
        endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )

    # Region is always us-east-1
    if not is_sandbox:
        client = boto3.client(service_name = 'mturk', region_name='us-east-1')

    # Test that you can connect to the API by checking your account balance
    # In Sandbox this always returns $10,000
    try:
        user_balance = float(client.get_account_balance()['AvailableBalance'])
    except ClientError as e:
        if e.response['Error']['Code'] == 'RequestError':
            print('ERROR: To use the MTurk API, you will need an Amazon Web Services (AWS) Account. Your AWS account must be linked to your Amazon Mechanical Turk Account. Visit https://requestersandbox.mturk.com/developer to get started. (Note: if you have recently linked your account, please wait for a couple minutes before trying again.)\n')
            quit()
        else:
            raise

    balance_needed = balance_needed * 1.2 # AWS charges 20% fee for both reward and bonus payment

    if user_balance < balance_needed:
        print("You might not have enough money in your MTurk account. Please go to https://requester.mturk.com/account and increase your balance to at least $"+f'{balance_needed:.2f}'+", and then try again.")
        return False
    else:
        return True

def create_hit_config(task_description, unique_worker, is_sandbox):
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
    client = boto3.client(
        service_name = 'mturk',
        region_name = 'us-east-1',
        endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )
    # Region is always us-east-1
    if not is_sandbox:
        client = boto3.client(service_name = 'mturk', region_name='us-east-1')
    return client

def create_hit_type(hit_title, hit_description, hit_keywords, hit_reward, assignment_duration_in_seconds, is_sandbox):
    client = boto3.client(
        service_name = 'mturk',
        region_name = 'us-east-1',
        endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )

    # Region is always us-east-1
    if not is_sandbox:
        client = boto3.client(service_name = 'mturk', region_name='us-east-1')

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

    # Create the HIT type
    response = client.create_hit_type(
        AutoApprovalDelayInSeconds=4*7*24*3600, # auto-approve after 4 weeks
        AssignmentDurationInSeconds=assignment_duration_in_seconds,
        Reward=str(hit_reward),
        Title=hit_title,
        Keywords=hit_keywords,
        Description=hit_description,
        QualificationRequirements=localRequirements
    )
    hit_type_id = response['HITTypeId']
    return hit_type_id

def create_hit_with_hit_type(page_url, hit_type_id, num_assignments, is_sandbox):
    page_url = page_url.replace('&', '&amp;')

    question_data_struture = '''<ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
      <ExternalURL>'''+page_url+'''</ExternalURL>
      <FrameHeight>'''+str(mturk_hit_frame_height)+'''</FrameHeight>
    </ExternalQuestion>
    '''

    client = boto3.client(
        service_name = 'mturk',
        region_name = 'us-east-1',
        endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    )

    # Region is always us-east-1
    if not is_sandbox:
        client = boto3.client(service_name = 'mturk', region_name='us-east-1')

    # Create the HIT
    response = client.create_hit_with_hit_type(
        HITTypeId=hit_type_id,
        MaxAssignments=num_assignments,
        LifetimeInSeconds=31536000,
        Question=question_data_struture,
        # AssignmentReviewPolicy={
        #     'PolicyName': 'string',
        #     'Parameters': [
        #         {
        #             'Key': 'string',
        #             'Values': [
        #                 'string',
        #             ],
        #             'MapEntries': [
        #                 {
        #                     'Key': 'string',
        #                     'Values': [
        #                         'string',
        #                     ]
        #                 },
        #             ]
        #         },
        #     ]
        # },
        # HITReviewPolicy={
        #     'PolicyName': 'string',
        #     'Parameters': [
        #         {
        #             'Key': 'string',
        #             'Values': [
        #                 'string',
        #             ],
        #             'MapEntries': [
        #                 {
        #                     'Key': 'string',
        #                     'Values': [
        #                         'string',
        #                     ]
        #                 },
        #             ]
        #         },
        #     ]
        # },
    )

    # The response included several fields that will be helpful later
    hit_type_id = response['HIT']['HITTypeId']
    hit_id = response['HIT']['HITId']
    hit_link = "https://workersandbox.mturk.com/mturk/preview?groupId=" + hit_type_id
    if not is_sandbox:
        hit_link = "https://www.mturk.com/mturk/preview?groupId=" + hit_type_id
    return hit_link, hit_id
