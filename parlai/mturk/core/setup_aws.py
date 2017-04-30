# Copyright 2004-present Facebook. All Rights Reserved.
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
import uuid
import hashlib
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound

aws_profile_name = 'parlai_mturk'
region_name = 'us-west-2'

iam_role_name = 'parlai_relay_server'
lambda_function_name = 'parlai_relay_server'
api_gateway_name = 'ParlaiRelayServer'
endpoint_api_name_index = 'parlai_relay_server'
endpoint_api_name_message = 'parlai_relay_server_message'
endpoint_api_name_approval = 'parlai_relay_server_approval'

rds_db_instance_identifier = 'parlai-mturk-db'
rds_db_name = 'parlai_mturk_db'
rds_username = 'parlai_user'
rds_password = 'parlai_user_password'
rds_security_group_name = 'parlai-mturk-db-security-group'
rds_security_group_description = 'Security group for ParlAI MTurk DB'

parent_dir = os.path.dirname(os.path.abspath(__file__))
files_to_copy = [parent_dir+'/'+'data_model.py', parent_dir+'/'+'mturk_index.html']
lambda_server_directory_name = 'lambda_server'
lambda_server_zip_file_name = 'lambda_server.zip'

def setup_aws_credentials():
    try:
        session = boto3.Session(profile_name=aws_profile_name)
    except ProfileNotFound as e:
        print("AWS credentials not found. Please create an IAM user with AdministratorAccess permission at https://console.aws.amazon.com/iam/, and then enter the user's security credentials below:")
        aws_access_key_id = input('Access Key ID: ')
        aws_secret_access_key = input('Secret Access Key: ')
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
        print("AWS credentials successfully saved in "+aws_credentials_file_path+" file.")
    os.environ["AWS_PROFILE"] = aws_profile_name

def get_requester_key():
    # Compute requester key
    session = boto3.Session(profile_name=aws_profile_name)
    hash_gen = hashlib.sha512()
    hash_gen.update(session.get_credentials().access_key.encode('utf-8')+session.get_credentials().secret_key.encode('utf-8'))
    requester_key_gt = hash_gen.hexdigest()

    return requester_key_gt

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
                               DBInstanceClass='db.t2.micro',
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
    status = db_instance['DBInstanceStatus']

    if status not in ['available', 'backing-up']:
        print("RDS: Waiting for newly created database to be available. This might take a couple minutes...")

    while status not in ['available', 'backing-up']:
        time.sleep(5)
        response = rds.describe_db_instances(DBInstanceIdentifier=rds_db_instance_identifier)
        db_instances = response['DBInstances']
        db_instance = db_instances[0]
        status = db_instance['DBInstanceStatus']
    
    endpoint = db_instance['Endpoint']
    host = endpoint['Address']

    print('RDS: DB instance ready.')

    return host

def setup_relay_server_api(mturk_submit_url, rds_host, task_config, is_sandbox, num_hits, requester_key_gt, should_clean_up_after_upload=True):
    # Dynamically generate handler.py file, and then create zip file
    print("Lambda: Preparing relay server code...")

    # Create clean folder for lambda server code
    if os.path.exists(parent_dir + '/' + lambda_server_directory_name):
        shutil.rmtree(parent_dir + '/' + lambda_server_directory_name)
    os.makedirs(parent_dir + '/' + lambda_server_directory_name)
    if os.path.exists(parent_dir + '/' + lambda_server_zip_file_name):
        os.remove(parent_dir + '/' + lambda_server_zip_file_name)

    # Copying files
    with open(parent_dir+'/handler_template.py', 'r') as handler_template_file:
        handler_file_string = handler_template_file.read()
    handler_file_string = handler_file_string.replace(
        '# {{block_task_config}}', 
        "mturk_submit_url = \'" + mturk_submit_url + "\'\n" + \
        "rds_host = \'" + rds_host + "\'\n" + \
        "rds_db_name = \'" + rds_db_name + "\'\n" + \
        "rds_username = \'" + rds_username + "\'\n" + \
        "rds_password = \'" + rds_password + "\'\n" + \
        "requester_key_gt = \'" + requester_key_gt + "\'\n" + \
        "num_hits = " + str(num_hits) + "\n" + \
        "is_sandbox = " + str(is_sandbox) + "\n" + \
        'task_description = ' + task_config['task_description'])
    with open(parent_dir + '/' + lambda_server_directory_name+'/handler.py', "w") as handler_file:
        handler_file.write(handler_file_string)
    create_zip_file(
        lambda_server_directory_name=lambda_server_directory_name, 
        lambda_server_zip_file_name=lambda_server_zip_file_name,
        files_to_copy=files_to_copy
    )
    with open(parent_dir + '/' + lambda_server_zip_file_name, mode='rb') as zip_file:
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
                    Timeout = 10, # in seconds
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
            StatementId = str(uuid.uuid1()),
            Action = 'lambda:InvokeFunction',
            Principal = 'apigateway.amazonaws.com',
        )

    # Clean up if needed
    if should_clean_up_after_upload:
        shutil.rmtree(parent_dir + '/' + lambda_server_directory_name)
        os.remove(parent_dir + '/' + lambda_server_zip_file_name)

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
    index_endpoint_exists = False
    message_endpoint_exists = False
    approval_endpoint_exists = False
    root_endpoint_id = None
    response = api_gateway_client.get_resources(restApiId=rest_api_id)
    resources = response['items']
    for resource in resources:
        if resource['path'] == '/':
            root_endpoint_id = resource['id']
        elif resource['path'] == '/' + endpoint_api_name_index:
            index_endpoint_exists = True
        elif resource['path'] == '/' + endpoint_api_name_message:
            message_endpoint_exists = True
        elif resource['path'] == '/' + endpoint_api_name_approval:
            approval_endpoint_exists = True

    if not index_endpoint_exists:
        print("API Gateway: Creating endpoint for index...")
        resource_for_index_endpoint = api_gateway_client.create_resource(
            restApiId = rest_api_id,
            parentId = root_endpoint_id,
            pathPart = endpoint_api_name_index
        )
        # Set up GET method
        api_gateway_client.put_method(
            restApiId = rest_api_id,
            resourceId = resource_for_index_endpoint['id'],
            httpMethod = "GET",
            authorizationType = "NONE",
            apiKeyRequired = False,
        )
        api_gateway_client.put_method_response(
            restApiId = rest_api_id,
            resourceId = resource_for_index_endpoint['id'],
            httpMethod = 'GET',
            statusCode = '200',
            responseParameters = {
                'method.response.header.Access-Control-Allow-Origin': False,
                'method.response.header.Content-Type': False
            }
        )
        api_gateway_client.put_integration(
            restApiId = rest_api_id,
            resourceId = resource_for_index_endpoint['id'],
            httpMethod = 'GET',
            type = 'AWS',
            integrationHttpMethod = 'POST', # this has to be POST
            uri = "arn:aws:apigateway:"+region_name+":lambda:path/2015-03-31/functions/"+lambda_function_arn+"/invocations",
            requestTemplates = {
                'application/json': \
'''
{
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
}
'''
            },
            passthroughBehavior = 'WHEN_NO_TEMPLATES'
        )
        api_gateway_client.put_integration_response(
            restApiId = rest_api_id,
            resourceId = resource_for_index_endpoint['id'],
            httpMethod = 'GET',
            statusCode = '200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Origin': "'*'",
                'method.response.header.Content-Type': "'text/html'"
            },
            responseTemplates={
                "text/html": "$input.path('$')"
            },
        )

        # Set up POST method
        api_gateway_client.put_method(
            restApiId = rest_api_id,
            resourceId = resource_for_index_endpoint['id'],
            httpMethod = "POST",
            authorizationType = "NONE",
            apiKeyRequired = False,
        )
        api_gateway_client.put_method_response(
            restApiId = rest_api_id,
            resourceId = resource_for_index_endpoint['id'],
            httpMethod = 'POST',
            statusCode = '200',
            responseParameters = {
                'method.response.header.Access-Control-Allow-Origin': False
            },
            responseModels = {
                'application/json': 'Empty'
            }
        )
        api_gateway_client.put_integration(
            restApiId = rest_api_id,
            resourceId = resource_for_index_endpoint['id'],
            httpMethod = 'POST',
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
        api_gateway_client.put_integration_response(
            restApiId = rest_api_id,
            resourceId = resource_for_index_endpoint['id'],
            httpMethod = 'POST',
            statusCode = '200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Origin': "'*'"
            },
            responseTemplates={
                'application/json': ''
            },
        )
    else:
        print("API Gateway: Endpoint for index already exists.")

    if not message_endpoint_exists:
        print("API Gateway: Creating endpoint for message...")
        resource_for_message_endpoint = api_gateway_client.create_resource(
            restApiId = rest_api_id,
            parentId = root_endpoint_id,
            pathPart = endpoint_api_name_message
        )
        # TODO: set up integration configs for this endpoint here
        # Set up GET method
        api_gateway_client.put_method(
            restApiId = rest_api_id,
            resourceId = resource_for_message_endpoint['id'],
            httpMethod = "GET",
            authorizationType = "NONE",
            apiKeyRequired = False,
        )
        api_gateway_client.put_method_response(
            restApiId = rest_api_id,
            resourceId = resource_for_message_endpoint['id'],
            httpMethod = 'GET',
            statusCode = '200',
            responseParameters = {
                'method.response.header.Access-Control-Allow-Origin': False
            },
            responseModels = {
                'application/json': 'Empty'
            }
        )
        api_gateway_client.put_integration(
            restApiId = rest_api_id,
            resourceId = resource_for_message_endpoint['id'],
            httpMethod = 'GET',
            type = 'AWS',
            integrationHttpMethod = 'POST', # this has to be POST
            uri = "arn:aws:apigateway:"+region_name+":lambda:path/2015-03-31/functions/"+lambda_function_arn+"/invocations",
            requestTemplates = {
                'application/json': \
'''
{
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
}
'''
            },
            passthroughBehavior = 'WHEN_NO_TEMPLATES'
        )
        api_gateway_client.put_integration_response(
            restApiId = rest_api_id,
            resourceId = resource_for_message_endpoint['id'],
            httpMethod = 'GET',
            statusCode = '200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Origin': "'*'"
            },
            responseTemplates={
                "application/json": ""
            },
        )

        # Set up POST method
        api_gateway_client.put_method(
            restApiId = rest_api_id,
            resourceId = resource_for_message_endpoint['id'],
            httpMethod = "POST",
            authorizationType = "NONE",
            apiKeyRequired = False,
        )
        api_gateway_client.put_method_response(
            restApiId = rest_api_id,
            resourceId = resource_for_message_endpoint['id'],
            httpMethod = 'POST',
            statusCode = '200',
            responseParameters = {
                'method.response.header.Access-Control-Allow-Origin': False
            },
            responseModels = {
                'application/json': 'Empty'
            }
        )
        api_gateway_client.put_integration(
            restApiId = rest_api_id,
            resourceId = resource_for_message_endpoint['id'],
            httpMethod = 'POST',
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
        api_gateway_client.put_integration_response(
            restApiId = rest_api_id,
            resourceId = resource_for_message_endpoint['id'],
            httpMethod = 'POST',
            statusCode = '200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Origin': "'*'"
            },
            responseTemplates={
                'application/json': ''
            },
        )
    else:
        print("API Gateway: Endpoint for message already exists.")

    if not approval_endpoint_exists:
        print("API Gateway: Creating endpoint for approval...")
        resource_for_approval_endpoint = api_gateway_client.create_resource(
            restApiId = rest_api_id,
            parentId = root_endpoint_id,
            pathPart = endpoint_api_name_approval
        )
        # Set up GET method
        api_gateway_client.put_method(
            restApiId = rest_api_id,
            resourceId = resource_for_approval_endpoint['id'],
            httpMethod = "GET",
            authorizationType = "NONE",
            apiKeyRequired = False,
        )
        api_gateway_client.put_method_response(
            restApiId = rest_api_id,
            resourceId = resource_for_approval_endpoint['id'],
            httpMethod = 'GET',
            statusCode = '200',
            responseParameters = {
                'method.response.header.Access-Control-Allow-Origin': False,
                'method.response.header.Content-Type': False
            }
        )
        api_gateway_client.put_integration(
            restApiId = rest_api_id,
            resourceId = resource_for_approval_endpoint['id'],
            httpMethod = 'GET',
            type = 'AWS',
            integrationHttpMethod = 'POST', # this has to be POST
            uri = "arn:aws:apigateway:"+region_name+":lambda:path/2015-03-31/functions/"+lambda_function_arn+"/invocations",
            requestTemplates = {
                'application/json': \
'''
{
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
}
'''
            },
            passthroughBehavior = 'WHEN_NO_TEMPLATES'
        )
        api_gateway_client.put_integration_response(
            restApiId = rest_api_id,
            resourceId = resource_for_approval_endpoint['id'],
            httpMethod = 'GET',
            statusCode = '200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Origin': "'*'",
                'method.response.header.Content-Type': "'text/html'"
            },
            responseTemplates={
                "text/html": "$input.path('$')"
            },
        )

        # Set up POST method
        api_gateway_client.put_method(
            restApiId = rest_api_id,
            resourceId = resource_for_approval_endpoint['id'],
            httpMethod = "POST",
            authorizationType = "NONE",
            apiKeyRequired = False,
        )
        api_gateway_client.put_method_response(
            restApiId = rest_api_id,
            resourceId = resource_for_approval_endpoint['id'],
            httpMethod = 'POST',
            statusCode = '200',
            responseParameters = {
                'method.response.header.Access-Control-Allow-Origin': False
            },
            responseModels = {
                'application/json': 'Empty'
            }
        )
        api_gateway_client.put_integration(
            restApiId = rest_api_id,
            resourceId = resource_for_approval_endpoint['id'],
            httpMethod = 'POST',
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
        api_gateway_client.put_integration_response(
            restApiId = rest_api_id,
            resourceId = resource_for_approval_endpoint['id'],
            httpMethod = 'POST',
            statusCode = '200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Origin': "'*'"
            },
            responseTemplates={
                'application/json': ''
            },
        )
    else:
        print("API Gateway: Endpoint for approval already exists.")

    if not (index_endpoint_exists and message_endpoint_exists and approval_endpoint_exists):
        api_gateway_client.create_deployment(
            restApiId = rest_api_id,
            stageName = "prod",
        )

    index_api_endpoint_url = 'https://' + rest_api_id + '.execute-api.' + region_name + '.amazonaws.com/prod/' + endpoint_api_name_index
    approval_api_endpoint_url = 'https://' + rest_api_id + '.execute-api.' + region_name + '.amazonaws.com/prod/' + endpoint_api_name_approval
    return index_api_endpoint_url, approval_api_endpoint_url

def check_mturk_balance(num_hits, hit_reward, is_sandbox):
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
    user_balance = float(client.get_account_balance()['AvailableBalance'])
    
    balance_needed = num_hits * hit_reward * 1.2

    if user_balance < balance_needed:
        print("You might not have enough money in your MTurk account. Please increase your balance to at least $"+f'{balance_needed:.2f}'+" and try again.")
        return False
    else:
        return True

def create_hit_type(hit_title, hit_description, hit_keywords, hit_reward, is_sandbox):
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
    user_balance = client.get_account_balance()

    # TODO: check balance to see if enough

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
        AssignmentDurationInSeconds=1800,
        Reward=str(hit_reward),
        Title=hit_title,
        Keywords=hit_keywords,
        Description=hit_description,
        QualificationRequirements=localRequirements
    )
    hit_type_id = response['HITTypeId']
    return hit_type_id

def create_hit_with_hit_type(page_url, hit_type_id, is_sandbox=True):
    page_url = page_url.replace('&', '&amp;')

    frame_height = 650
    
    question_data_struture = '''<ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
      <ExternalURL>{{external_url}}</ExternalURL>
      <FrameHeight>{{frame_height}}</FrameHeight>
    </ExternalQuestion>
    '''

    question_data_struture = question_data_struture.replace('{{external_url}}', page_url).replace('{{frame_height}}', str(frame_height))

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
    user_balance = client.get_account_balance()

    # Create the HIT 
    response = client.create_hit_with_hit_type(
        HITTypeId=hit_type_id,
        MaxAssignments=1,
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

    # response = client.create_hit(
    #     MaxAssignments = 1,
    #     LifetimeInSeconds = 31536000,
    #     AssignmentDurationInSeconds = 1800,
    #     Reward = str(hit_reward),
    #     Title = hit_title,
    #     Keywords = hit_keywords,
    #     Description = hit_description,
    #     Question = question_data_struture,
    #     #QualificationRequirements = localRequirements
    # )

    # The response included several fields that will be helpful later
    hit_type_id = response['HIT']['HITTypeId']
    hit_id = response['HIT']['HITId']
    hit_link = "https://workersandbox.mturk.com/mturk/preview?groupId=" + hit_type_id
    return hit_link

def setup_all_dependencies(lambda_server_directory_name):
    devnull = open(os.devnull, 'w')
    parent_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up all other dependencies
    command_str = "pip install --target="+parent_dir+'/'+lambda_server_directory_name+" -r "+parent_dir+"/lambda_requirements.txt"
    command = command_str.split(" ")
    call(command, stdout=devnull, stderr=devnull)

    # Set up psycopg2
    command = "git clone https://github.com/yf225/awslambda-psycopg2.git".split(" ")
    call(command, stdout=devnull, stderr=devnull)
    shutil.copytree("./awslambda-psycopg2/with_ssl_support/psycopg2", parent_dir+'/'+lambda_server_directory_name+"/psycopg2")
    shutil.rmtree("./awslambda-psycopg2")

def create_zip_file(lambda_server_directory_name, lambda_server_zip_file_name, files_to_copy=None, verbose=False):
    setup_all_dependencies(lambda_server_directory_name)
    parent_dir = os.path.dirname(os.path.abspath(__file__))

    src = parent_dir + '/' + lambda_server_directory_name
    dst = parent_dir + '/' + lambda_server_zip_file_name

    if files_to_copy:
        for file_path in files_to_copy:
            shutil.copy2(file_path, src)

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

def setup_aws(task_config, num_hits, is_sandbox=True):
    mturk_submit_url = 'https://workersandbox.mturk.com/mturk/externalSubmit'
    if not is_sandbox:
        mturk_submit_url = 'https://www.mturk.com/mturk/externalSubmit'
    requester_key_gt = get_requester_key()
    rds_host = setup_rds()
    index_api_endpoint_url, approval_api_endpoint_url = setup_relay_server_api(mturk_submit_url, rds_host, task_config, is_sandbox, num_hits, requester_key_gt)

    chat_interface_url = index_api_endpoint_url + "?endpoint=index&task_group_id={{task_group_id}}&conversation_id={{conversation_id}}&cur_agent_id={{cur_agent_id}}"
    approval_url = approval_api_endpoint_url + "?endpoint=approval&task_group_id={{task_group_id}}&conversation_id=1&cur_agent_id={{cur_agent_id}}&requester_key="+requester_key_gt
    
    # webbrowser.open(chat_interface_url)
    
    return rds_host, chat_interface_url, approval_url
