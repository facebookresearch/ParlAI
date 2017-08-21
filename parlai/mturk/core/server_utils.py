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
        'unique_worker': unique_worker
    }
    hit_config_file_path = os.path.join(parent_dir, 'hit_config.json')
    if os.path.exists(hit_config_file_path):
        os.remove(hit_config_file_path)
    with open(hit_config_file_path, 'w') as hit_config_file:
        hit_config_file.write(json.dumps(hit_config))

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
    return setup_heroku_server(task_files_to_copy=task_files_to_copy)
