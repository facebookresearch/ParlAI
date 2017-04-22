# Copyright 2004-present Facebook. All Rights Reserved.
import os
import shutil
import boto3
import botocore
import time
import json
import webbrowser
import uuid
from mturk_task_config import *
import create_zip_file
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound

# bucket_name = 'parlai-relay-server'  # default name for ParlAI MTurk S3 bucket
aws_profile_name = 'parlai_mturk'
region_name = 'us-west-2'

iam_role_name = 'parlai_relay_server'
lambda_function_name = 'parlai_relay_server'
api_gateway_name = 'ParlaiRelayServer'
endpoint_api_name_index = 'parlai_relay_server'
endpoint_api_name_message = 'parlai_relay_server_message'
lambda_server_directory = 'lambda_server'

rds_db_instance_identifier = 'parlai-mturk-db'
rds_db_name = 'parlai_mturk_db'
rds_username = 'parlai_user'
rds_password = 'parlai_user_password'
rds_security_group_name = 'parlai_mturk_db_security_group'
rds_security_group_description = 'Security group for ParlAI MTurk DB'

parent_dir = os.path.dirname(os.path.abspath(__file__))
files_to_copy = [parent_dir+'/'+'data_management.py', parent_dir+'/'+'mturk_index.html']

# TODO: We should have a separate file that stores the user's access tokens, and ask them to enter it on initial use.
# or ask them to put it in ~/.aws/config file, following the correct format.
# ACCESS_KEY = 'AKIAJLD4IBIEDB2NEWEQ'  
# SECRET_KEY = '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'


def setup_aws_credentials():
    try:
        session = boto3.Session(profile_name=aws_profile_name)
    except ProfileNotFound as e:
        print("AWS credentials not found, please enter below:")
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
        print('RDS: Ingress Successfully Set ' + data)
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
        print('RDS: Starting RDS instance with ID: ' + rds_db_instance_identifier)
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
        print("RDS: Waiting for DB status to change to available. This might take a couple minutes...")

    while status not in ['available', 'backing-up']:
        time.sleep(5)
        response = rds.describe_db_instances(DBInstanceIdentifier=rds_db_instance_identifier)
        db_instances = response['DBInstances']
        db_instance = db_instances[0]
        status = db_instance['DBInstanceStatus']
    
    endpoint = db_instance['Endpoint']
    host = endpoint['Address']

    print('RDS: DB instance ready with host: ' + host)

    return host


def setup_relay_server_api(rds_host, should_clean_up_after_upload=True):
    # Dynamically generate handler.py file, and then create zip file
    print("Lambda: Preparing relay server code...")

    # Create folder for lambda server code
    if not os.path.exists(lambda_server_directory):
        os.makedirs(lambda_server_directory)

    # Copying files
    with open('handler_template.py', 'r') as handler_template_file:
        handler_file_string = handler_template_file.read()
    handler_file_string = handler_file_string.replace(
        '# {{block_task_config}}', 
        "rds_host = \'" + rds_host + "\'\n" + \
        "rds_db_name = \'" + rds_db_name + "\'\n" + \
        "rds_username = \'" + rds_username + "\'\n" + \
        "rds_password = \'" + rds_password + "\'\n" + \
        'agent_display_names = ' + str(agent_display_names) + '\n' + \
        'task_description = ' + task_description + '\n' + \
        'state_config = ' + str(state_config))
    with open(lambda_server_directory+'/handler.py', "w") as handler_file:
        handler_file.write(handler_file_string)
    create_zip_file.create_zip_file(files_to_copy=files_to_copy)
    with open('lambda_server.zip', mode='rb') as zip_file:
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
                PolicyArn = 'arn:aws:iam::aws:policy/AmazonS3FullAccess'
            )
            iam_client.attach_role_policy(
                RoleName = iam_role_name,
                PolicyArn = 'arn:aws:iam::aws:policy/AmazonRDSFullAccess'
            )

        iam = boto3.resource('iam')
        iam_role = iam.Role(iam_role_name)

        # Create the Lambda function and upload latest code
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

        # Add permission to endpoints for calling Lambda function
        response = lambda_client.add_permission(
            FunctionName = lambda_function_name,
            StatementId = str(uuid.uuid1()),
            Action = 'lambda:InvokeFunction',
            Principal = 'apigateway.amazonaws.com',
        )

    # Clean up if needed
    if should_clean_up_after_upload:
        shutil.rmtree(lambda_server_directory)
        os.remove('lambda_server.zip')

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

    if not (index_endpoint_exists and message_endpoint_exists):
        api_gateway_client.create_deployment(
            restApiId = rest_api_id,
            stageName = "prod",
        )

    index_api_endpoint_url = 'https://' + rest_api_id + '.execute-api.' + region_name + '.amazonaws.com/prod/' + endpoint_api_name_index
    return index_api_endpoint_url


def submit_to_mturk(mturk_chat_url):
    page_url = mturk_chat_url
    page_url = page_url.replace('&', '&amp;')

    title = "Test HIT (COMPLETE THIS TASK ONLY ONCE!)"
    description = "COMPLETE THIS TASK ONLY ONCE! All submissions after the first will be rejected"
    keywords = "easy"
    frame_height = 650
    amount = 0.05

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

    # Uncomment the below to connect to the live marketplace
    # Region is always us-east-1
    # client = boto3.client(service_name = 'mturk', region_name='us-east-1')

    # Test that you can connect to the API by checking your account balance
    user_balance = client.get_account_balance()

    # In Sandbox this always returns $10,000
    # print("Your account balance is " + user_balance['AvailableBalance'])

    # Create a qualification with Locale In('US', 'CA') requirement attached
    # localRequirements = [{
    #     'QualificationTypeId': '00000000000000000071',
    #     'Comparator': 'In',
    #     'LocaleValues': [{
    #         'Country': 'US'
    #     }, {
    #         'Country': 'CA'
    #     }],
    #     'RequiredToPreview': True
    # }]

    # Create the HIT 
    response = client.create_hit(
        MaxAssignments = 1,
        LifetimeInSeconds = 6000,
        AssignmentDurationInSeconds = 600,
        Reward ='0.05',
        Title = title,
        Keywords = keywords,
        Description = description,
        Question = question_data_struture,
        #QualificationRequirements = localRequirements
    )

    # The response included several fields that will be helpful later
    hit_type_id = response['HIT']['HITTypeId']
    hit_id = response['HIT']['HITId']
    hit_link = "https://workersandbox.mturk.com/mturk/preview?groupId=" + hit_type_id
    return hit_link


def setup_aws():
    setup_aws_credentials()
    rds_host = setup_rds()
    index_api_endpoint_url = setup_relay_server_api(rds_host)

    chat_interface_url = index_api_endpoint_url + "?endpoint=index&task_group_id={{task_group_id}}&conversation_id={{conversation_id}}&cur_agent_id={{cur_agent_id}}"
    
    # webbrowser.open(chat_interface_url)
    
    return rds_host, chat_interface_url


if __name__ == '__main__':
    setup_aws()