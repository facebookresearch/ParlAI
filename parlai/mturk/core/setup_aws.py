# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
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
from parlai.mturk.core.data_model import setup_database_engine, init_database, check_database_health

aws_profile_name = 'parlai_mturk'
region_name = 'us-east-1'
user_name = getpass.getuser()

iam_role_name = 'parlai_relay_server'
lambda_function_name = 'parlai_relay_server_' + user_name
lambda_permission_statement_id = 'lambda-permission-statement-id'
api_gateway_name = 'ParlaiRelayServer_' + user_name
endpoint_api_name_html = 'html'  # For GET-ing HTML
endpoint_api_name_json = 'json'  # For GET-ing and POST-ing JSON

rds_db_instance_identifier = 'parlai-mturk-db-' + user_name
rds_db_name = 'parlai_mturk_db_' + user_name
rds_username = 'parlai_user'
rds_password = 'parlai_user_password'
rds_security_group_name = 'parlai-mturk-db-security-group'
rds_security_group_description = 'Security group for ParlAI MTurk DB'
rds_db_instance_class = 'db.t2.medium'

parent_dir = os.path.dirname(os.path.abspath(__file__))
generic_files_to_copy = [
    os.path.join(parent_dir, 'hit_config.json'),
    os.path.join(parent_dir, 'data_model.py'),
    os.path.join(parent_dir, 'html', 'core.html'), 
    os.path.join(parent_dir, 'html', 'cover_page.html'), 
    os.path.join(parent_dir, 'html', 'mturk_index.html')
]
lambda_server_directory_name = 'lambda_server'
lambda_server_zip_file_name = 'lambda_server.zip'
mturk_hit_frame_height = 650

def add_api_gateway_method(api_gateway_client, lambda_function_arn, rest_api_id, endpoint_resource, http_method_type, response_data_type):
    api_gateway_client.put_method(
        restApiId = rest_api_id,
        resourceId = endpoint_resource['id'],
        httpMethod = http_method_type,
        authorizationType = "NONE",
        apiKeyRequired = False,
    )

    response_parameters = { 'method.response.header.Access-Control-Allow-Origin': False }
    if response_data_type == 'html':
        response_parameters['method.response.header.Content-Type'] = False
    response_models = {}
    if response_data_type == 'json':
        response_models = { 'application/json': 'Empty' }
    api_gateway_client.put_method_response(
        restApiId = rest_api_id,
        resourceId = endpoint_resource['id'],
        httpMethod = http_method_type,
        statusCode = '200',
        responseParameters = response_parameters,
        responseModels = response_models
    )

    api_gateway_client.put_integration(
        restApiId = rest_api_id,
        resourceId = endpoint_resource['id'],
        httpMethod = http_method_type,
        type = 'AWS',
        integrationHttpMethod = 'POST', # this has to be POST
        uri = "arn:aws:apigateway:"+region_name+":lambda:path/2015-03-31/functions/"+lambda_function_arn+"/invocations",
        requestTemplates = {
            'application/json': \
'''{
  "body" : $input.json('$'),
  "headers": {
    #foreach($header in $input.params().header.keySet())
    "$header": "$util.escapeJavaScript($input.params().header.get($header))" #if($foreach.hasNext),#end

    #end
  },
  "method": "$context.httpMethod",
  "params": {
    #foreach($param in $input.params().path.keySet())
    "$param": "$util.escapeJavaScript($input.params().path.get($param))" #if($foreach.hasNext),#end

    #end
  },
  "query": {
    #foreach($queryParam in $input.params().querystring.keySet())
    "$queryParam": "$util.escapeJavaScript($input.params().querystring.get($queryParam))" #if($foreach.hasNext),#end

    #end
  }
}'''
        },
        passthroughBehavior = 'WHEN_NO_TEMPLATES'
    )

    response_parameters = { 'method.response.header.Access-Control-Allow-Origin': "'*'" }
    response_templates = { 'application/json': '' }
    if response_data_type == 'html':
        response_parameters['method.response.header.Content-Type'] = "'text/html'"
        response_templates = { "text/html": "$input.path('$')" }
    api_gateway_client.put_integration_response(
        restApiId = rest_api_id,
        resourceId = endpoint_resource['id'],
        httpMethod = http_method_type,
        statusCode = '200',
        responseParameters=response_parameters,
        responseTemplates=response_templates,
    )

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

def setup_rds():
    # Set up security group rules first
    ec2 = boto3.client('ec2', region_name=region_name)

    response = ec2.describe_vpcs()
    vpc_id = response.get('Vpcs', [{}])[0].get('VpcId', '')
    security_group_id = None

    try:
        response = ec2.create_security_group(GroupName=rds_security_group_name,
                                             Description=rds_security_group_description,
                                             VpcId=vpc_id)
        security_group_id = response['GroupId']
        print('RDS: Security group created.')

        data = ec2.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                {
                 'IpProtocol': 'tcp',
                 'FromPort': 5432,
                 'ToPort': 5432,
                 'IpRanges': [{'CidrIp': '0.0.0.0/0'}],
                 'Ipv6Ranges': [{'CidrIpv6': '::/0'}]
                },
            ])
        print('RDS: Security group ingress IP permissions set.')
    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidGroup.Duplicate':
            print('RDS: Security group already exists.')
            response = ec2.describe_security_groups(GroupNames=[rds_security_group_name])
            security_group_id = response['SecurityGroups'][0]['GroupId']

    rds_instance_is_ready = False
    while not rds_instance_is_ready:
        rds = boto3.client('rds', region_name=region_name)
        try:
            rds.create_db_instance(DBInstanceIdentifier=rds_db_instance_identifier,
                                   AllocatedStorage=20,
                                   DBName=rds_db_name,
                                   Engine='postgres',
                                   # General purpose SSD
                                   StorageType='gp2',
                                   StorageEncrypted=False,
                                   AutoMinorVersionUpgrade=True,
                                   MultiAZ=False,
                                   MasterUsername=rds_username,
                                   MasterUserPassword=rds_password,
                                   VpcSecurityGroupIds=[security_group_id],
                                   DBInstanceClass=rds_db_instance_class,
                                   Tags=[{'Key': 'Name', 'Value': rds_db_instance_identifier}])
            print('RDS: Starting RDS instance...')
        except ClientError as e:
            if e.response['Error']['Code'] == 'DBInstanceAlreadyExists':
                print('RDS: DB instance already exists.')
            else:
                raise

        response = rds.describe_db_instances(DBInstanceIdentifier=rds_db_instance_identifier)
        db_instances = response['DBInstances']
        db_instance = db_instances[0]

        if db_instance['DBInstanceClass'] != rds_db_instance_class: # If instance class doesn't match
            print('RDS: Instance class does not match.')
            remove_rds_database()
            rds_instance_is_ready = False
            continue

        status = db_instance['DBInstanceStatus']

        if status == 'deleting':
            print("RDS: Waiting for previous delete operation to complete. This might take a couple minutes...")
            try:
                while status == 'deleting':
                    time.sleep(5)
                    response = rds.describe_db_instances(DBInstanceIdentifier=rds_db_instance_identifier)
                    db_instances = response['DBInstances']
                    db_instance = db_instances[0]
                    status = db_instance['DBInstanceStatus']
            except ClientError as e:
                rds_instance_is_ready = False
                continue

        if status == 'creating':
            print("RDS: Waiting for newly created database to be available. This might take a couple minutes...")
            while status == 'creating':
                time.sleep(5)
                response = rds.describe_db_instances(DBInstanceIdentifier=rds_db_instance_identifier)
                db_instances = response['DBInstances']
                db_instance = db_instances[0]
                status = db_instance['DBInstanceStatus']

        endpoint = db_instance['Endpoint']
        host = endpoint['Address']

        setup_database_engine(host, rds_db_name, rds_username, rds_password)
        database_health_status = check_database_health()
        if database_health_status in ['missing_table', 'healthy']:
            print("Remote database health status: "+database_health_status)
            init_database()
        elif database_health_status in ['inconsistent_schema', 'unknown_error']:
            print("Remote database error: "+database_health_status+". Removing RDS database...")
            remove_rds_database()
            rds_instance_is_ready = False
            continue

        print('RDS: DB instance ready.')
        rds_instance_is_ready = True

    return host

def remove_rds_database():
    # Remove RDS database
    rds = boto3.client('rds', region_name=region_name)
    try:
        response = rds.describe_db_instances(DBInstanceIdentifier=rds_db_instance_identifier)
        db_instances = response['DBInstances']
        db_instance = db_instances[0]
        status = db_instance['DBInstanceStatus']

        if status == 'deleting':
            print("RDS: Waiting for previous delete operation to complete. This might take a couple minutes...")
        else:
            response = rds.delete_db_instance(
                DBInstanceIdentifier=rds_db_instance_identifier,
                SkipFinalSnapshot=True,
            )
            response = rds.describe_db_instances(DBInstanceIdentifier=rds_db_instance_identifier)
            db_instances = response['DBInstances']
            db_instance = db_instances[0]
            status = db_instance['DBInstanceStatus']

            if status == 'deleting':
                print("RDS: Deleting database. This might take a couple minutes...")

        try:
            while status == 'deleting':
                time.sleep(5)
                response = rds.describe_db_instances(DBInstanceIdentifier=rds_db_instance_identifier)
                db_instances = response['DBInstances']
                db_instance = db_instances[0]
                status = db_instance['DBInstanceStatus']
        except ClientError as e:
            print("RDS: Database deleted.")

    except ClientError as e:
        print("RDS: Database doesn't exist.")


def create_hit_config(task_description, num_hits, num_assignments, is_sandbox):
    mturk_submit_url = 'https://workersandbox.mturk.com/mturk/externalSubmit'
    if not is_sandbox:
        mturk_submit_url = 'https://www.mturk.com/mturk/externalSubmit'
    hit_config = {
        'task_description': task_description, 
        'num_hits': num_hits, 
        'num_assignments': num_assignments, 
        'is_sandbox': is_sandbox,
        'mturk_submit_url': mturk_submit_url,
    }
    hit_config_file_path = os.path.join(parent_dir, 'hit_config.json')
    if os.path.exists(hit_config_file_path):
        os.remove(hit_config_file_path)
    with open(hit_config_file_path, 'w') as hit_config_file:
        hit_config_file.write(json.dumps(hit_config))

def setup_relay_server_api(rds_host, task_files_to_copy, should_clean_up_after_upload=True):
    # Dynamically generate handler.py file, and then create zip file
    print("Lambda: Preparing relay server code...")

    # Create clean folder for lambda server code
    if os.path.exists(os.path.join(parent_dir, lambda_server_directory_name)):
        shutil.rmtree(os.path.join(parent_dir, lambda_server_directory_name))
    os.makedirs(os.path.join(parent_dir, lambda_server_directory_name))
    if os.path.exists(os.path.join(parent_dir, lambda_server_zip_file_name)):
        os.remove(os.path.join(parent_dir, lambda_server_zip_file_name))

    # Copying files
    with open(os.path.join(parent_dir, 'handler_template.py'), 'r') as handler_template_file:
        handler_file_string = handler_template_file.read()
    handler_file_string = handler_file_string.replace(
        '# {{block_task_config}}',
        "frame_height = " + str(mturk_hit_frame_height) + "\n" + \
        "rds_host = \'" + rds_host + "\'\n" + \
        "rds_db_name = \'" + rds_db_name + "\'\n" + \
        "rds_username = \'" + rds_username + "\'\n" + \
        "rds_password = \'" + rds_password + "\'")
    with open(os.path.join(parent_dir, lambda_server_directory_name, 'handler.py'), 'w') as handler_file:
        handler_file.write(handler_file_string)
    create_zip_file(
        lambda_server_directory_name=lambda_server_directory_name,
        lambda_server_zip_file_name=lambda_server_zip_file_name,
        files_to_copy=generic_files_to_copy + task_files_to_copy
    )
    with open(os.path.join(parent_dir, lambda_server_zip_file_name), mode='rb') as zip_file:
        zip_file_content = zip_file.read()

    # Create Lambda function
    lambda_client = boto3.client('lambda', region_name=region_name)
    lambda_function_arn = None
    try:
        # Case 1: if Lambda function exists
        lambda_function = lambda_client.get_function(FunctionName=lambda_function_name)
        print("Lambda: Function already exists. Uploading latest version of code...")
        lambda_function_arn = lambda_function['Configuration']['FunctionArn']
        # Upload latest code for Lambda function
        lambda_client.update_function_code(
            FunctionName = lambda_function_name,
            ZipFile = zip_file_content,
            Publish = True
        )
    except ClientError as e:
        # Case 2: if Lambda function does not exist
        print("Lambda: Function does not exist. Creating it...")
        iam_client = boto3.client('iam')
        try:
            iam_client.get_role(RoleName=iam_role_name)
        except ClientError as e:
            # Should create IAM role for Lambda server
            iam_client.create_role(
                RoleName = iam_role_name,
                AssumeRolePolicyDocument = '''{ "Version": "2012-10-17", "Statement": [ { "Effect": "Allow", "Principal": { "Service": "lambda.amazonaws.com" }, "Action": "sts:AssumeRole" } ]}'''
            )
            iam_client.attach_role_policy(
                RoleName = iam_role_name,
                PolicyArn = 'arn:aws:iam::aws:policy/AWSLambdaFullAccess'
            )
            iam_client.attach_role_policy(
                RoleName = iam_role_name,
                PolicyArn = 'arn:aws:iam::aws:policy/AmazonRDSFullAccess'
            )
            iam_client.attach_role_policy(
                RoleName = iam_role_name,
                PolicyArn = 'arn:aws:iam::aws:policy/AmazonMechanicalTurkFullAccess'
            )

        iam = boto3.resource('iam')
        iam_role = iam.Role(iam_role_name)
        lambda_function_arn = None

        # Create the Lambda function and upload latest code
        while True:
            try:
                response = lambda_client.create_function(
                    FunctionName = lambda_function_name,
                    Runtime = 'python2.7',
                    Role = iam_role.arn,
                    Handler='handler.lambda_handler',
                    Code={
                        'ZipFile': zip_file_content
                    },
                    Timeout = 300, # in seconds
                    MemorySize = 128, # in MB
                    Publish = True,
                )
                lambda_function_arn = response['FunctionArn']
                break
            except ClientError as e:
                print("Lambda: Waiting for IAM role creation to take effect...")
                time.sleep(10)

        # Add permission to endpoints for calling Lambda function
        response = lambda_client.add_permission(
            FunctionName = lambda_function_name,
            StatementId = lambda_permission_statement_id,
            Action = 'lambda:InvokeFunction',
            Principal = 'apigateway.amazonaws.com',
        )

    # Clean up if needed
    if should_clean_up_after_upload:
        shutil.rmtree(os.path.join(parent_dir, lambda_server_directory_name))
        os.remove(os.path.join(parent_dir, lambda_server_zip_file_name))
        os.remove(os.path.join(parent_dir, 'hit_config.json'))

    # Check API Gateway existence.
    # If doesn't exist, create the APIs, point them to Lambda function, and set correct configurations
    api_gateway_exists = False
    rest_api_id = None
    api_gateway_client = boto3.client('apigateway', region_name=region_name)
    response = api_gateway_client.get_rest_apis()
    if not 'items' in response:
        api_gateway_exists = False
    else:
        rest_apis = response['items']
        for api in rest_apis:
            if api['name'] == api_gateway_name:
                api_gateway_exists = True
                rest_api_id = api['id']
                break
    if not api_gateway_exists:
        rest_api = api_gateway_client.create_rest_api(
            name = api_gateway_name,
        )
        rest_api_id = rest_api['id']

    # Create endpoint resources if doesn't exist
    html_endpoint_exists = False
    json_endpoint_exists = False
    root_endpoint_id = None
    response = api_gateway_client.get_resources(restApiId=rest_api_id)
    resources = response['items']
    for resource in resources:
        if resource['path'] == '/':
            root_endpoint_id = resource['id']
        elif resource['path'] == '/' + endpoint_api_name_html:
            html_endpoint_exists = True
        elif resource['path'] == '/' + endpoint_api_name_json:
            json_endpoint_exists = True

    if not html_endpoint_exists:
        print("API Gateway: Creating endpoint for html...")
        resource_for_html_endpoint = api_gateway_client.create_resource(
            restApiId = rest_api_id,
            parentId = root_endpoint_id,
            pathPart = endpoint_api_name_html
        )

        # Set up GET method
        add_api_gateway_method(
            api_gateway_client = api_gateway_client,
            lambda_function_arn = lambda_function_arn,
            rest_api_id = rest_api_id,
            endpoint_resource = resource_for_html_endpoint,
            http_method_type = 'GET',
            response_data_type = 'html'
        )
    else:
        print("API Gateway: Endpoint for html already exists.")

    if not json_endpoint_exists:
        print("API Gateway: Creating endpoint for json...")
        resource_for_json_endpoint = api_gateway_client.create_resource(
            restApiId = rest_api_id,
            parentId = root_endpoint_id,
            pathPart = endpoint_api_name_json
        )

        # Set up GET method
        add_api_gateway_method(
            api_gateway_client = api_gateway_client,
            lambda_function_arn = lambda_function_arn,
            rest_api_id = rest_api_id,
            endpoint_resource = resource_for_json_endpoint,
            http_method_type = 'GET',
            response_data_type = 'json'
        )

        # Set up POST method
        add_api_gateway_method(
            api_gateway_client = api_gateway_client,
            lambda_function_arn = lambda_function_arn,
            rest_api_id = rest_api_id,
            endpoint_resource = resource_for_json_endpoint,
            http_method_type = 'POST',
            response_data_type = 'json'
        )
    else:
        print("API Gateway: Endpoint for json already exists.")

    if not (html_endpoint_exists and json_endpoint_exists):
        api_gateway_client.create_deployment(
            restApiId = rest_api_id,
            stageName = "prod",
            cacheClusterEnabled = False,
        )

    html_api_endpoint_url = 'https://' + rest_api_id + '.execute-api.' + region_name + '.amazonaws.com/prod/' + endpoint_api_name_html
    json_api_endpoint_url = 'https://' + rest_api_id + '.execute-api.' + region_name + '.amazonaws.com/prod/' + endpoint_api_name_json

    return html_api_endpoint_url, json_api_endpoint_url

def calculate_mturk_cost(payment_opt):
    """MTurk Pricing: https://requester.mturk.com/pricing
    20% fee on the reward and bonus amount (if any) you pay Workers.
    HITs with 10 or more assignments will be charged an additional 20% fee on the reward you pay Workers.

    Example payment_opt format for paying reward:
    {
        'type': 'reward',
        'num_hits': 1,
        'num_assignments': 1,
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
        total_cost = payment_opt['num_hits'] * payment_opt['num_assignments'] * payment_opt['reward'] * 1.2
        if payment_opt['num_assignments'] >= 10:
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
    return hit_link

def setup_all_dependencies(lambda_server_directory_name):
    devnull = open(os.devnull, 'w')
    parent_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if Anaconda is installed
    has_anaconda = False
    try:
        call("conda", stdout=devnull, stderr=devnull)
        has_anaconda = True
    except OSError:
        has_anaconda = False

    # Set up all other dependencies
    if has_anaconda:
        call(("pip install --target="+os.path.join(parent_dir, lambda_server_directory_name)+" -r "+os.path.join(parent_dir, "lambda_requirements.txt")).split(" "), stdout=devnull, stderr=devnull)
    else:
        shutil.rmtree(os.path.join(parent_dir, "venv"), ignore_errors=True)
        call("pip install virtualenv".split(" "), stdout=devnull, stderr=devnull)
        call(("virtualenv -p python2 "+os.path.join(parent_dir, "venv")).split(" "), stdout=devnull, stderr=devnull)
        call((os.path.join(parent_dir, 'venv', 'bin', 'pip')+" install --target="+os.path.join(parent_dir, lambda_server_directory_name)+" -r "+os.path.join(parent_dir, "lambda_requirements.txt")).split(" "), stdout=devnull, stderr=devnull)
        shutil.rmtree(os.path.join(parent_dir, "venv"), ignore_errors=True)

    # Set up psycopg2
    shutil.rmtree(os.path.join(parent_dir, 'awslambda-psycopg2'), ignore_errors=True)
    call(("git clone https://github.com/jkehler/awslambda-psycopg2.git " + os.path.join(parent_dir, "awslambda-psycopg2")).split(" "), stdout=devnull, stderr=devnull)
    shutil.copytree(os.path.join(parent_dir, 'awslambda-psycopg2', 'with_ssl_support', 'psycopg2'), os.path.join(parent_dir, lambda_server_directory_name, "psycopg2"))
    shutil.rmtree(os.path.join(parent_dir, 'awslambda-psycopg2'))

def create_zip_file(lambda_server_directory_name, lambda_server_zip_file_name, files_to_copy=None, verbose=False):
    setup_all_dependencies(lambda_server_directory_name)
    parent_dir = os.path.dirname(os.path.abspath(__file__))

    src = os.path.join(parent_dir, lambda_server_directory_name)
    dst = os.path.join(parent_dir, lambda_server_zip_file_name)

    if files_to_copy:
        for file_path in files_to_copy:
            try:
                shutil.copy2(file_path, src)
            except FileNotFoundError:
                pass

    zf = zipfile.ZipFile("%s" % (dst), "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            os.chmod(absname, 0o777)
            arcname = os.path.relpath(absname, abs_src)
            if verbose:
                print('zipping %s as %s' % (os.path.join(dirname, filename),
                                            arcname))
            zf.write(absname, arcname)
    zf.close()

    if verbose:
        print("Done!")

def setup_aws(task_files_to_copy):
    rds_host = setup_rds()
    html_api_endpoint_url, json_api_endpoint_url = setup_relay_server_api(rds_host=rds_host, task_files_to_copy=task_files_to_copy)

    return html_api_endpoint_url, json_api_endpoint_url

def clean_aws():
    # Remove RDS database
    try:
        rds = boto3.client('rds', region_name=region_name)
        response = rds.delete_db_instance(
            DBInstanceIdentifier=rds_db_instance_identifier,
            SkipFinalSnapshot=True,
        )
        response = rds.describe_db_instances(DBInstanceIdentifier=rds_db_instance_identifier)
        db_instances = response['DBInstances']
        db_instance = db_instances[0]
        status = db_instance['DBInstanceStatus']

        if status == 'deleting':
            print("RDS: Deleting database. This might take a couple minutes...")

        try:
            while status == 'deleting':
                time.sleep(5)
                response = rds.describe_db_instances(DBInstanceIdentifier=rds_db_instance_identifier)
                db_instances = response['DBInstances']
                db_instance = db_instances[0]
                status = db_instance['DBInstanceStatus']
        except ClientError as e:
            print("RDS: Database deleted.")

    except ClientError as e:
        print("RDS: Database doesn't exist.")

    # Remove RDS security group
    try:
        ec2 = boto3.client('ec2', region_name=region_name)

        response = ec2.describe_security_groups(GroupNames=[rds_security_group_name])
        security_group_id = response['SecurityGroups'][0]['GroupId']

        response = ec2.delete_security_group(
            DryRun=False,
            GroupName=rds_security_group_name,
            GroupId=security_group_id
        )
        print("RDS: Security group removed.")
    except ClientError as e:
        print("RDS: Security group doesn't exist.")

    # Remove API Gateway endpoints
    api_gateway_client = boto3.client('apigateway', region_name=region_name)
    api_gateway_exists = False
    rest_api_id = None
    response = api_gateway_client.get_rest_apis()
    if not 'items' in response:
        api_gateway_exists = False
    else:
        rest_apis = response['items']
        for api in rest_apis:
            if api['name'] == api_gateway_name:
                api_gateway_exists = True
                rest_api_id = api['id']
                break
    if api_gateway_exists:
        response = api_gateway_client.delete_rest_api(
            restApiId=rest_api_id
        )
        print("API Gateway: Endpoints are removed.")
    else:
        print("API Gateway: Endpoints don't exist.")

    # Remove permission for calling Lambda function
    try:
        lambda_client = boto3.client('lambda', region_name=region_name)
        response = lambda_client.remove_permission(
            FunctionName=lambda_function_name,
            StatementId=lambda_permission_statement_id
        )
        print("Lambda: Permission removed.")
    except ClientError as e:
        print("Lambda: Permission doesn't exist.")

    # Remove Lambda function
    try:
        lambda_client = boto3.client('lambda', region_name=region_name)
        response = lambda_client.delete_function(
            FunctionName=lambda_function_name
        )
        print("Lambda: Function removed.")
    except ClientError as e:
        print("Lambda: Function doesn't exist.")

    # Remove IAM role
    try:
        iam_client = boto3.client('iam')

        try:
            response = iam_client.detach_role_policy(
                RoleName=iam_role_name,
                PolicyArn='arn:aws:iam::aws:policy/AWSLambdaFullAccess'
            )
        except ClientError as e:
            pass

        try:
            response = iam_client.detach_role_policy(
                RoleName=iam_role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonRDSFullAccess'
            )
        except ClientError as e:
            pass

        try:
            response = iam_client.detach_role_policy(
                RoleName=iam_role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonMechanicalTurkFullAccess'
            )
        except ClientError as e:
            pass

        response = iam_client.delete_role(
            RoleName=iam_role_name
        )
        time.sleep(10)
        print("IAM: Role removed.")
    except ClientError as e:
        print("IAM: Role doesn't exist.")

if __name__ == "__main__":
    if sys.argv[1] == 'clean':
        setup_aws_credentials()
        clean_aws()
    elif sys.argv[1] == 'remove_rds':
        setup_aws_credentials()
        remove_rds_database()
