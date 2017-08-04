# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import os
import platform
import sh
import shlex
import hashlib
import netrc
import sys
import shutil
import subprocess
import zipfile
import boto3
import botocore
import time
import json
import webbrowser
import hashlib
import getpass
import re
import glob
from botocore.exceptions import ClientError
from botocore.exceptions import ProfileNotFound

region_name = 'us-east-1'
user_name = getpass.getuser()

rds_db_instance_identifier = 'parlai-mturk-db-' + user_name
rds_db_name = 'parlai_mturk_db_' + user_name
rds_username = 'parlai_user'
rds_password = 'parlai_user_password'
rds_security_group_name = 'parlai-mturk-db-security-group'
rds_security_group_description = 'Security group for ParlAI MTurk DB'
rds_db_instance_class = 'db.t2.medium'

parent_dir = os.path.dirname(os.path.abspath(__file__))
server_source_directory_name = 'server'
heroku_server_directory_name = 'heroku_server'
task_directory_name = 'task'

def create_hit_config(task_description, unique_worker, is_sandbox):
    mturk_submit_url = 'https://workersandbox.mturk.com/mturk/externalSubmit'
    if not is_sandbox:
        mturk_submit_url = 'https://www.mturk.com/mturk/externalSubmit'
    hit_config = {
        'task_description': task_description,
        'is_sandbox': is_sandbox,
        'mturk_submit_url': mturk_submit_url,
        'unique_worker': unique_worker,
    }
    hit_config_file_path = os.path.join(parent_dir, 'hit_config.json')
    if os.path.exists(hit_config_file_path):
        os.remove(hit_config_file_path)
    with open(hit_config_file_path, 'w') as hit_config_file:
        hit_config_file.write(json.dumps(hit_config))

def create_server_config(db_host, db_name, db_username, db_password):
    server_config = {
        'db_host': db_host,
        'db_name': db_name,
        'db_username': db_username,
        'db_password': db_password,
    }
    server_config_file_path = os.path.join(parent_dir, 'server_config.json')
    if os.path.exists(server_config_file_path):
        os.remove(server_config_file_path)
    with open(server_config_file_path, 'w') as server_config_file:
        server_config_file.write(json.dumps(server_config))

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

def setup_heroku_server(task_files_to_copy=None):
    # Install Heroku CLI
    os_name = None
    bit_architecture = None

    platform_info = platform.platform()
    if 'Darwin' in platform_info: # Mac OS X
        os_name = 'darwin'
    elif 'Linux' in platform_info: # Linux
        os_name = 'linux'
    else:
        os_name = 'windows'

    bit_architecture_info = platform.architecture()[0]
    if '64bit' in bit_architecture_info:
        bit_architecture = 'x64'
    else:
        bit_architecture = 'x86'

    existing_heroku_directory_names = glob.glob(os.path.join(parent_dir, 'heroku-cli-*'))
    if len(existing_heroku_directory_names):
        shutil.rmtree(os.path.join(parent_dir, existing_heroku_directory_names[0]))
    if os.path.exists(os.path.join(parent_dir, 'heroku.tar.gz')):
        os.remove(os.path.join(parent_dir, 'heroku.tar.gz'))

    os.chdir(parent_dir)
    sh.wget(shlex.split('https://cli-assets.heroku.com/heroku-cli/channels/stable/heroku-cli-{}-{}.tar.gz -O heroku.tar.gz'.format(os_name, bit_architecture)))
    sh.tar(shlex.split('-xvzf heroku.tar.gz'))

    heroku_directory_name = glob.glob(os.path.join(parent_dir, 'heroku-cli-*'))[0]
    heroku_directory_path = os.path.join(parent_dir, heroku_directory_name)
    heroku_executable_path = os.path.join(heroku_directory_path, 'bin', 'heroku')
    
    server_source_directory_path = os.path.join(parent_dir, server_source_directory_name)
    heroku_server_directory_path = os.path.join(parent_dir, heroku_server_directory_name)

    sh.rm(shlex.split('-rf '+heroku_server_directory_path))
    
    shutil.copytree(server_source_directory_path, heroku_server_directory_path)

    task_directory_path = os.path.join(heroku_server_directory_path, task_directory_name)
    sh.mv(os.path.join(heroku_server_directory_path, 'html'), task_directory_path)

    hit_config_file_path = os.path.join(parent_dir, 'hit_config.json')
    sh.mv(hit_config_file_path, task_directory_path)

    server_config_file_path = os.path.join(parent_dir, 'server_config.json')
    sh.mv(server_config_file_path, heroku_server_directory_path)
    
    for file_path in task_files_to_copy:
        try:
            shutil.copy2(file_path, task_directory_path)
        except FileNotFoundError:
            pass

    print("Heroku: Starting server...")

    os.chdir(heroku_server_directory_path)
    sh.git('init')
    heroku_user_identifier = None
    while not heroku_user_identifier:
        try:
            subprocess.check_output(shlex.split(heroku_executable_path+' auth:token'))
            heroku_user_identifier = netrc.netrc(os.path.join(os.path.expanduser("~"), '.netrc')).hosts['api.heroku.com'][0]
        except subprocess.CalledProcessError:
            raise SystemExit("A free Heroku account is required for launching MTurk tasks. Please register at https://signup.heroku.com/ and run `"+heroku_executable_path+" login` at the terminal to login to Heroku, and then run this program again.")

    heroku_app_name = (user_name + '-' + hashlib.md5(heroku_user_identifier.encode('utf-8')).hexdigest())[:30]
    try:
        subprocess.check_output(shlex.split(heroku_executable_path+' create ' + heroku_app_name))
    except subprocess.CalledProcessError: # Heroku app already exists
        subprocess.check_output(shlex.split(heroku_executable_path+' git:remote -a ' + heroku_app_name))

    # Enable WebSockets
    try:
        subprocess.check_output(shlex.split(heroku_executable_path+' features:enable http-session-affinity'))
    except subprocess.CalledProcessError: # Already enabled WebSockets
        pass

    os.chdir(heroku_server_directory_path)
    sh.git(shlex.split('add -A'))
    sh.git(shlex.split('commit -m "app"'))
    sh.git(shlex.split('push -f heroku master'))
    subprocess.check_output(shlex.split(heroku_executable_path+' ps:scale web=1'))
    os.chdir(parent_dir)
    
    if os.path.exists(heroku_directory_path):
        shutil.rmtree(heroku_directory_path)
    if os.path.exists(os.path.join(parent_dir, 'heroku.tar.gz')):
        os.remove(os.path.join(parent_dir, 'heroku.tar.gz'))
    
    sh.rm(shlex.split('-rf '+heroku_server_directory_path))

    return 'https://'+heroku_app_name+'.herokuapp.com'
    
def setup_server(task_files_to_copy):
    db_host = setup_rds()
    create_server_config(
        db_host=db_host,
        db_name=rds_db_name,
        db_username=rds_username,
        db_password=rds_password
    )
    server_url = setup_heroku_server(task_files_to_copy=task_files_to_copy)
    return server_url, db_host

if __name__ == "__main__":
    if sys.argv[1] == 'remove_rds':
        setup_aws_credentials()
        remove_rds_database()