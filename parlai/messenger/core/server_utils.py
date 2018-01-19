# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import getpass
import glob
import hashlib
import netrc
import os
import platform
import re
import sh
import shlex
import shutil
import subprocess

region_name = 'us-east-1'
user_name = getpass.getuser()

parent_dir = os.path.dirname(os.path.abspath(__file__))
server_source_directory_name = 'server'
heroku_server_directory_name = 'heroku_server'
task_directory_name = 'task'

heroku_url = \
    'https://cli-assets.heroku.com/heroku-cli/channels/stable/heroku-cli'

# TODO randomize this
verify_token = 'TEMPORARY_TOKEN'

gcloud_url = (
    'https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/'
    'google-cloud-sdk-185.0.0')

def get_gcloud_client():
    print("GCloud: Getting client...")
    # Install google-cloud-sdk-185 CLI
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
        raise Exception(
            'ParlAI messenger is not yet supported on windows machines')

    # Find our architecture
    bit_architecture_info = platform.architecture()[0]
    if '64bit' in bit_architecture_info:
        bit_architecture = 'x86_64'
    else:
        bit_architecture = 'x86'

    # Remove existing client files
    existing_gcloud_directory_names = \
        glob.glob(os.path.join(parent_dir, 'google-cloud-sdk'))
    if len(existing_gcloud_directory_names) == 0:
        if os.path.exists(os.path.join(parent_dir, 'gcloud.tar.gz')):
            os.remove(os.path.join(parent_dir, 'gcloud.tar.gz'))

        # Get the gcloud client and unzip
        os.chdir(parent_dir)
        sh.wget(shlex.split('{}-{}-{}.tar.gz -O gcloud.tar.gz'.format(
            gcloud_url,
            os_name,
            bit_architecture
        )))
        sh.tar(shlex.split('-xvzf gcloud.tar.gz'))

    gcloud_directory_path = os.path.join(parent_dir, 'google-cloud-sdk')
    gcloud_executable_path = \
        os.path.join(gcloud_directory_path, 'bin', 'gcloud')

    # Ensure we have the beta features (required for gcloud functions)
    subprocess.check_output(
        shlex.split(gcloud_executable_path + ' components install beta -q')
    )

    return gcloud_executable_path


def setup_gcloud_function(heroku_server_url):
    gcloud_executable_path = get_gcloud_client()
    # prepare the cloud function file to take the proper endpoint
    sh.sed(
        shlex.split(
            '"s~##HEROKU_URL##~' + heroku_server_url +
            '~" forward_message_template.js'
        ),
        _out='forward_message/index_tmp.js'
    )
    sh.sed(
        shlex.split(
            '"s~##VERIFY_TOKEN##~' + verify_token + '~" '
            'forward_message/index_tmp.js'
        ),
        _out='forward_message/index.js'
    )
    sh.rm(shlex.split('forward_message/index_tmp.js'))

    try:
        print('Creating the google cloud function')
        subprocess.check_output(
            shlex.split(gcloud_executable_path + ' beta functions deploy ' +
                        'forward_message --source=forward_message ' +
                        '--trigger-http --verbosity debug ' +
                        '--entry-point=forward_message')
        )
    except Exception:
        # The stored setup was corrupted or something otherwise went wrong.
        print(
            'You need to be logged into google cloud to be able to recieve '
            'messages from the messenger API without hosting your own endpoint'
            '. When the process asks you to select a cloud project to use, '
            'either select one you\'re using specifically for parlai-messenger'
            ' or create a new one.')
        input('Please press Enter to login to your google cloud account... ')
        subprocess.check_output(
            shlex.split(gcloud_executable_path + ' init')
        )
        # Actually submit the function
        subprocess.check_output(
            shlex.split(gcloud_executable_path + ' beta functions deploy ' +
                        'forward_message --source=forward_message.js ' +
                        '--trigger-http')
        )

    gcloud_config = bytes.decode(subprocess.check_output(
        shlex.split(gcloud_executable_path + ' config list')
    ))
    project_extract_pat = re.compile(r'project = ([^\n]*)\n')
    project_name = project_extract_pat.search(gcloud_config).group(1)
    print('Extracted project name: ' + project_name)

    # Determine if billing is enabled, which is required to transmit the
    # messages to heroku from the function
    billing_enabled = False
    billing_pat = re.compile(r'\nbillingEnabled: true\n')
    while not billing_enabled:
        billing_details = bytes.decode(subprocess.check_output(
            shlex.split(gcloud_executable_path +
                        ' beta billing projects describe ' +
                        project_name)
        ))
        if billing_pat.search(billing_details):
            billing_enabled = True
        else:
            input('Please go to https://console.cloud.google.com/billing/ and '
                  'activate billing for the current project ({}) in order to '
                  'use ParlAI messenger. As long as your usage stays below '
                  '5GB a month this service should remain free according to '
                  'https://cloud.google.com/functions/pricing. Once finished '
                  'press enter here.')

    function_details = bytes.decode(subprocess.check_output(shlex.split(
        gcloud_executable_path + ' beta functions describe forward_message'
    )))
    url_pat = re.compile(r'httpsTrigger:\n  url: ([^\n]*)\n')
    webhook_url = url_pat.search(function_details).group(1)
    print('Your validation id is ' + verify_token)
    print('Your webhook url is ' + webhook_url)


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
    existing_heroku_directory_names = \
        glob.glob(os.path.join(parent_dir, 'heroku-cli-*'))
    if len(existing_heroku_directory_names) == 0:
        if os.path.exists(os.path.join(parent_dir, 'heroku.tar.gz')):
            os.remove(os.path.join(parent_dir, 'heroku.tar.gz'))

        # Get the heroku client and unzip
        os.chdir(parent_dir)
        sh.wget(shlex.split('{}-{}-{}.tar.gz -O heroku.tar.gz'.format(
            heroku_url,
            os_name,
            bit_architecture
        )))
        sh.tar(shlex.split('-xvzf heroku.tar.gz'))

    heroku_directory_name = \
        glob.glob(os.path.join(parent_dir, 'heroku-cli-*'))[0]
    heroku_directory_path = os.path.join(parent_dir, heroku_directory_name)
    heroku_executable_path = \
        os.path.join(heroku_directory_path, 'bin', 'heroku')

    server_source_directory_path = \
        os.path.join(parent_dir, server_source_directory_name)
    heroku_server_directory_path = os.path.join(parent_dir, '{}_{}'.format(
        heroku_server_directory_name,
        task_name
    ))

    # Delete old server files
    sh.rm(shlex.split('-rf '+heroku_server_directory_path))

    # Copy over a clean copy into the server directory
    shutil.copytree(server_source_directory_path, heroku_server_directory_path)

    print("Heroku: Starting server...")

    os.chdir(heroku_server_directory_path)
    sh.git('init')

    # get heroku credentials
    heroku_user_identifier = None
    while not heroku_user_identifier:
        try:
            subprocess.check_output(
                shlex.split(heroku_executable_path+' auth:token')
            )
            heroku_user_identifier = (
                netrc.netrc(os.path.join(os.path.expanduser("~"), '.netrc'))
                     .hosts['api.heroku.com'][0]
            )
        except subprocess.CalledProcessError:
            raise SystemExit(
                'A free Heroku account is required for launching MTurk tasks. '
                'Please register at https://signup.heroku.com/ and run `{} '
                'login` at the terminal to login to Heroku, and then run this '
                'program again.'.format(heroku_executable_path)
            )

    heroku_app_name = ('{}-{}-{}'.format(
        user_name,
        task_name,
        hashlib.md5(heroku_user_identifier.encode('utf-8')).hexdigest()
    ))[:30]

    while heroku_app_name[-1] == '-':
        heroku_app_name = heroku_app_name[:-1]

    # Create or attach to the server
    try:
        subprocess.check_output(shlex.split(
            '{} create {}'.format(heroku_executable_path, heroku_app_name)
        ))
    except subprocess.CalledProcessError:  # User has too many apps
        sh.rm(shlex.split('-rf {}'.format(heroku_server_directory_path)))
        raise SystemExit(
            'You have hit your limit on concurrent apps with heroku, which are'
            ' required to run multiple concurrent tasks.\nPlease wait for some'
            ' of your existing tasks to complete. If you have no tasks '
            'running, login to heroku and delete some of the running apps or '
            'verify your account to allow more concurrent apps'
        )

    # Enable WebSockets
    try:
        subprocess.check_output(shlex.split(
            '{} features:enable http-session-affinity'.format(
                heroku_executable_path
            )
        ))
    except subprocess.CalledProcessError:  # Already enabled WebSockets
        pass

    # commit and push to the heroku server
    os.chdir(heroku_server_directory_path)
    sh.git(shlex.split('add -A'))
    sh.git(shlex.split('commit -m "app"'))
    sh.git(shlex.split('push -f heroku master'))
    subprocess.check_output(shlex.split('{} ps:scale web=1'.format(
        heroku_executable_path)
    ))
    os.chdir(parent_dir)

    # Clean up heroku files
    if os.path.exists(os.path.join(parent_dir, 'heroku.tar.gz')):
        os.remove(os.path.join(parent_dir, 'heroku.tar.gz'))

    sh.rm(shlex.split('-rf {}'.format(heroku_server_directory_path)))

    return 'https://{}.herokuapp.com'.format(heroku_app_name)


def delete_heroku_server(task_name):
    heroku_directory_name = \
        glob.glob(os.path.join(parent_dir, 'heroku-cli-*'))[0]
    heroku_directory_path = os.path.join(parent_dir, heroku_directory_name)
    heroku_executable_path = \
        os.path.join(heroku_directory_path, 'bin', 'heroku')

    heroku_user_identifier = (
        netrc.netrc(os.path.join(os.path.expanduser("~"), '.netrc'))
             .hosts['api.heroku.com'][0]
    )

    heroku_app_name = ('{}-{}-{}'.format(
        user_name,
        task_name,
        hashlib.md5(heroku_user_identifier.encode('utf-8')).hexdigest()
    ))[:30]
    while heroku_app_name[-1] == '-':
        heroku_app_name = heroku_app_name[:-1]
    print("Heroku: Deleting server: {}".format(heroku_app_name))
    subprocess.check_output(shlex.split(
        '{} destroy {} --confirm {}'.format(
            heroku_executable_path,
            heroku_app_name,
            heroku_app_name
        )
    ))


def setup_server(task_name):
    return setup_heroku_server(task_name)


def delete_server(task_name):
    delete_heroku_server(task_name)
