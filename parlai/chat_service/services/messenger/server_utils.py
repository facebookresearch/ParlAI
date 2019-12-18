#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import glob
import hashlib
import netrc
import os
import platform
import sh
import shlex
import shutil
import subprocess
import time

region_name = 'us-east-1'
user_name = getpass.getuser()

parent_dir = os.path.dirname(os.path.abspath(__file__))
server_source_directory_name = 'server'
heroku_server_directory_name = 'heroku_server'
local_server_directory_name = 'local_server'
task_directory_name = 'task'

server_process = None

heroku_url = 'https://cli-assets.heroku.com/heroku-cli/channels/stable/heroku-cli'


def setup_heroku_server(task_name):
    print("Heroku: Collecting files...")
    # Install Heroku CLI
    os_name = None
    bit_architecture = None

    # Get the platform we are working on
    platform_info = platform.platform()
    if 'Darwin' in platform_info:  # Mac OS X
        os_name = 'darwin'
    elif 'Linux' in platform_info:  # Linux
        os_name = 'linux'
    else:
        os_name = 'windows'

    # Find our architecture
    bit_architecture_info = platform.architecture()[0]
    if '64bit' in bit_architecture_info:
        bit_architecture = 'x64'
    else:
        bit_architecture = 'x86'

    # Remove existing heroku client files
    existing_heroku_directory_names = glob.glob(
        os.path.join(parent_dir, 'heroku-cli-*')
    )
    if len(existing_heroku_directory_names) == 0:
        if os.path.exists(os.path.join(parent_dir, 'heroku.tar.gz')):
            os.remove(os.path.join(parent_dir, 'heroku.tar.gz'))

        # Get the heroku client and unzip
        os.chdir(parent_dir)
        sh.wget(
            shlex.split(
                '{}-{}-{}.tar.gz -O heroku.tar.gz'.format(
                    heroku_url, os_name, bit_architecture
                )
            )
        )
        sh.tar(shlex.split('-xvzf heroku.tar.gz'))

    heroku_directory_name = glob.glob(os.path.join(parent_dir, 'heroku-cli-*'))[0]
    heroku_directory_path = os.path.join(parent_dir, heroku_directory_name)
    heroku_executable_path = os.path.join(heroku_directory_path, 'bin', 'heroku')

    server_source_directory_path = os.path.join(
        parent_dir, server_source_directory_name
    )
    heroku_server_directory_path = os.path.join(
        parent_dir, '{}_{}'.format(heroku_server_directory_name, task_name)
    )

    # Delete old server files
    sh.rm(shlex.split('-rf ' + heroku_server_directory_path))

    # Copy over a clean copy into the server directory
    shutil.copytree(server_source_directory_path, heroku_server_directory_path)

    print("Heroku: Starting server...")

    os.chdir(heroku_server_directory_path)
    sh.git('init')

    # get heroku credentials
    heroku_user_identifier = None
    while not heroku_user_identifier:
        try:
            subprocess.check_output(shlex.split(heroku_executable_path + ' auth:token'))
            heroku_user_identifier = netrc.netrc(
                os.path.join(os.path.expanduser("~"), '.netrc')
            ).hosts['api.heroku.com'][0]
        except subprocess.CalledProcessError:
            raise SystemExit(
                'A free Heroku account is required for launching MTurk tasks. '
                'Please register at https://signup.heroku.com/ and run `{} '
                'login` at the terminal to login to Heroku, and then run this '
                'program again.'.format(heroku_executable_path)
            )

    heroku_app_name = (
        '{}-{}-{}'.format(
            user_name,
            task_name,
            hashlib.md5(heroku_user_identifier.encode('utf-8')).hexdigest(),
        )
    )[:30]

    while heroku_app_name[-1] == '-':
        heroku_app_name = heroku_app_name[:-1]

    # Create or attach to the server
    try:
        subprocess.check_output(
            shlex.split('{} create {}'.format(heroku_executable_path, heroku_app_name)),
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        error_text = bytes.decode(e.output)
        if "Name is already taken" in error_text:  # already running this app
            do_continue = input(
                'An app is already running with that name, do you want to '
                'restart a new run with it? (y/N): '
            )
            if do_continue != 'y':
                raise SystemExit('User chose not to re-run the app.')
            else:
                delete_heroku_server(task_name)
                try:
                    subprocess.check_output(
                        shlex.split(
                            '{} create {}'.format(
                                heroku_executable_path, heroku_app_name
                            )
                        ),
                        stderr=subprocess.STDOUT,
                    )
                except subprocess.CalledProcessError as e:
                    error_text = bytes.decode(e.output)
                    sh.rm(shlex.split('-rf {}'.format(heroku_server_directory_path)))
                    print(error_text)
                    raise SystemExit(
                        'Something unexpected happened trying to set up the '
                        'heroku server - please use the above printed error '
                        'to debug the issue however necessary.'
                    )
        elif "Delete some apps" in error_text:  # too many apps running
            sh.rm(shlex.split('-rf {}'.format(heroku_server_directory_path)))
            raise SystemExit(
                'You have hit your limit on concurrent apps with heroku, '
                'which are required to run multiple concurrent tasks.\nPlease '
                'wait for some of your existing tasks to complete. If you '
                'have no tasks running, login to heroku.com and delete some '
                'of the running apps or verify your account to allow more '
                'concurrent apps.'
            )
        else:
            sh.rm(shlex.split('-rf {}'.format(heroku_server_directory_path)))
            print(error_text)
            raise SystemExit(
                'Something unexpected happened trying to set up the heroku '
                'server - please use the above printed error to debug the '
                'issue however necessary.'
            )

    # Enable WebSockets
    try:
        subprocess.check_output(
            shlex.split(
                '{} features:enable http-session-affinity'.format(
                    heroku_executable_path
                )
            )
        )
    except subprocess.CalledProcessError:  # Already enabled WebSockets
        pass

    # commit and push to the heroku server
    os.chdir(heroku_server_directory_path)
    sh.git(shlex.split('add -A'))
    sh.git(shlex.split('commit -m "app"'))
    sh.git(shlex.split('push -f heroku master'))
    subprocess.check_output(
        shlex.split('{} ps:scale web=1'.format(heroku_executable_path))
    )
    os.chdir(parent_dir)

    # Clean up heroku files
    if os.path.exists(os.path.join(parent_dir, 'heroku.tar.gz')):
        os.remove(os.path.join(parent_dir, 'heroku.tar.gz'))

    sh.rm(shlex.split('-rf {}'.format(heroku_server_directory_path)))

    return 'https://{}.herokuapp.com'.format(heroku_app_name)


def delete_heroku_server(task_name):
    heroku_directory_name = glob.glob(os.path.join(parent_dir, 'heroku-cli-*'))[0]
    heroku_directory_path = os.path.join(parent_dir, heroku_directory_name)
    heroku_executable_path = os.path.join(heroku_directory_path, 'bin', 'heroku')

    heroku_user_identifier = netrc.netrc(
        os.path.join(os.path.expanduser("~"), '.netrc')
    ).hosts['api.heroku.com'][0]

    heroku_app_name = (
        '{}-{}-{}'.format(
            user_name,
            task_name,
            hashlib.md5(heroku_user_identifier.encode('utf-8')).hexdigest(),
        )
    )[:30]
    while heroku_app_name[-1] == '-':
        heroku_app_name = heroku_app_name[:-1]
    print("Heroku: Deleting server: {}".format(heroku_app_name))
    subprocess.check_output(
        shlex.split(
            '{} destroy {} --confirm {}'.format(
                heroku_executable_path, heroku_app_name, heroku_app_name
            )
        )
    )


def setup_local_server(task_name):
    global server_process
    print("Local Server: Collecting files...")

    server_source_directory_path = os.path.join(
        parent_dir, server_source_directory_name
    )
    local_server_directory_path = os.path.join(
        parent_dir, '{}_{}'.format(local_server_directory_name, task_name)
    )

    # Delete old server files
    sh.rm(shlex.split('-rf ' + local_server_directory_path))

    # Copy over a clean copy into the server directory
    shutil.copytree(server_source_directory_path, local_server_directory_path)

    print("Local: Starting server...")

    os.chdir(local_server_directory_path)

    packages_installed = subprocess.call(['npm', 'install'])
    if packages_installed != 0:
        raise Exception(
            'please make sure npm is installed, otherwise view '
            'the above error for more info.'
        )

    server_process = subprocess.Popen(['node', 'server.js'])

    time.sleep(1)
    print('Server running locally with pid {}.'.format(server_process.pid))
    host = input('Please enter the public server address, like https://hostname.com: ')
    port = input('Please enter the port given above, likely 3000: ')
    return '{}:{}'.format(host, port)


def delete_local_server(task_name):
    global server_process
    print('Terminating server')
    server_process.terminate()
    server_process.wait()
    print('Cleaning temp directory')
    local_server_directory_path = os.path.join(
        parent_dir, '{}_{}'.format(local_server_directory_name, task_name)
    )
    sh.rm(shlex.split('-rf ' + local_server_directory_path))


def setup_server(task_name, local=False):
    if local:
        return setup_local_server(task_name)
    return setup_heroku_server(task_name)


def delete_server(task_name, local=False):
    if local:
        delete_local_server(task_name)
    else:
        delete_heroku_server(task_name)
