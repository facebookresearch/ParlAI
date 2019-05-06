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
import parlai.mturk.core.shared_utils as shared_utils

region_name = 'us-east-1'
user_name = getpass.getuser()

parent_dir = shared_utils.get_core_dir()
legacy_server_source_directory_name = 'server_legacy'
server_source_directory_name = 'react_server'
heroku_server_directory_name = 'heroku_server'
local_server_directory_name = 'local_server'
task_directory_name = 'task'

server_process = None

heroku_url = \
    'https://cli-assets.heroku.com/heroku-cli/channels/stable/heroku-cli'


def setup_legacy_heroku_server(task_name, task_files_to_copy=None,
                               heroku_team=None, use_hobby=False,
                               tmp_dir=parent_dir):
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
        glob.glob(os.path.join(tmp_dir, 'heroku-cli-*'))
    if len(existing_heroku_directory_names) == 0:
        if os.path.exists(os.path.join(tmp_dir, 'heroku.tar.gz')):
            os.remove(os.path.join(tmp_dir, 'heroku.tar.gz'))

        # Get the heroku client and unzip
        os.chdir(tmp_dir)
        sh.wget(shlex.split('{}-{}-{}.tar.gz -O heroku.tar.gz'.format(
            heroku_url,
            os_name,
            bit_architecture
        )))
        sh.tar(shlex.split('-xvzf heroku.tar.gz'))

    heroku_directory_name = \
        glob.glob(os.path.join(tmp_dir, 'heroku-cli-*'))[0]
    heroku_directory_path = os.path.join(tmp_dir, heroku_directory_name)
    heroku_executable_path = \
        os.path.join(heroku_directory_path, 'bin', 'heroku')

    server_source_directory_path = \
        os.path.join(parent_dir, legacy_server_source_directory_name)
    heroku_server_directory_path = os.path.join(tmp_dir, '{}_{}'.format(
        heroku_server_directory_name,
        task_name
    ))

    # Delete old server files
    sh.rm(shlex.split('-rf ' + heroku_server_directory_path))

    # Copy over a clean copy into the server directory
    shutil.copytree(server_source_directory_path, heroku_server_directory_path)

    # Consolidate task files
    task_directory_path = \
        os.path.join(heroku_server_directory_path, task_directory_name)
    sh.mv(
        os.path.join(heroku_server_directory_path, 'html'),
        task_directory_path
    )

    hit_config_file_path = os.path.join(tmp_dir, 'hit_config.json')
    sh.mv(hit_config_file_path, task_directory_path)

    for file_path in task_files_to_copy:
        try:
            shutil.copy2(file_path, task_directory_path)
        except IsADirectoryError:  # noqa: F821 we don't support python2
            dir_name = os.path.basename(os.path.normpath(file_path))
            shutil.copytree(
                file_path, os.path.join(task_directory_path, dir_name))
        except FileNotFoundError:  # noqa: F821 we don't support python2
            pass

    print("Heroku: Starting server...")

    os.chdir(heroku_server_directory_path)
    sh.git('init')

    # get heroku credentials
    heroku_user_identifier = None
    while not heroku_user_identifier:
        try:
            subprocess.check_output(
                shlex.split(heroku_executable_path + ' auth:token')
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
        if heroku_team is not None:
            subprocess.check_output(shlex.split(
                '{} create {} --team {}'.format(
                    heroku_executable_path,
                    heroku_app_name,
                    heroku_team
                )
            ))
        else:
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

    if use_hobby:
        try:
            subprocess.check_output(shlex.split('{} dyno:type Hobby'.format(
                heroku_executable_path)
            ))
        except subprocess.CalledProcessError:  # User doesn't have hobby access
            delete_heroku_server(task_name)
            sh.rm(shlex.split('-rf {}'.format(heroku_server_directory_path)))
            raise SystemExit(
                'Server launched with hobby flag but account cannot create '
                'hobby servers.'
            )
    os.chdir(parent_dir)

    # Clean up heroku files
    if os.path.exists(os.path.join(parent_dir, 'heroku.tar.gz')):
        os.remove(os.path.join(parent_dir, 'heroku.tar.gz'))

    sh.rm(shlex.split('-rf {}'.format(heroku_server_directory_path)))

    return 'https://{}.herokuapp.com'.format(heroku_app_name)


def setup_heroku_server(task_name, task_files_to_copy=None,
                        heroku_team=None, use_hobby=False, tmp_dir=parent_dir):

    print("Heroku: Collecting files... for ", tmp_dir)
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
        glob.glob(os.path.join(tmp_dir, 'heroku-cli-*'))
    if len(existing_heroku_directory_names) == 0:
        if os.path.exists(os.path.join(tmp_dir, 'heroku.tar.gz')):
            os.remove(os.path.join(tmp_dir, 'heroku.tar.gz'))

        # Get the heroku client and unzip
        os.chdir(tmp_dir)
        sh.wget(shlex.split('{}-{}-{}.tar.gz -O heroku.tar.gz'.format(
            heroku_url,
            os_name,
            bit_architecture
        )))
        sh.tar(shlex.split('-xvzf heroku.tar.gz'))

    heroku_directory_name = \
        glob.glob(os.path.join(tmp_dir, 'heroku-cli-*'))[0]
    heroku_directory_path = os.path.join(tmp_dir, heroku_directory_name)
    heroku_executable_path = \
        os.path.join(heroku_directory_path, 'bin', 'heroku')

    server_source_directory_path = \
        os.path.join(parent_dir, server_source_directory_name)
    heroku_server_development_path = os.path.join(tmp_dir, '{}_{}'.format(
        heroku_server_directory_name,
        task_name
    ))

    # Delete old server files
    sh.rm(shlex.split('-rf ' + heroku_server_development_path))

    # Copy over a clean copy into the server directory
    shutil.copytree(server_source_directory_path, heroku_server_development_path)

    # Check to see if we need to build
    custom_component_dir = os.path.join(
        heroku_server_development_path, 'dev',
        'components', 'built_custom_components')
    if task_files_to_copy['needs_build'] is not None:
        # Build the directory, then pull the custom component out.
        print('Build: Detected custom package.json, prepping build')
        task_files_to_copy['components'] = []

        frontend_dir = task_files_to_copy['needs_build']

        sh.rm(shlex.split('-rf ' + custom_component_dir))
        shutil.copytree(frontend_dir, custom_component_dir)

        os.chdir(heroku_server_development_path)
        packages_installed = subprocess.call(
            ['npm', 'install', custom_component_dir])
        if packages_installed != 0:
            raise Exception('please make sure npm is installed, otherwise view'
                            ' the above error for more info.')

        os.chdir(custom_component_dir)

        webpack_complete = subprocess.call(['npm', 'run', 'dev'])
        if webpack_complete != 0:
            raise Exception('Webpack appears to have failed to build your '
                            'custom components. See the above for more info.')
    else:
        os.chdir(heroku_server_development_path)
        packages_installed = subprocess.call(
            ['npm', 'install', custom_component_dir])
        if packages_installed != 0:
            raise Exception('please make sure npm is installed, otherwise view'
                            ' the above error for more info.')

    # Move dev resource files to their correct places
    for resource_type in ['css', 'components']:
        target_resource_dir = os.path.join(
            heroku_server_development_path, 'dev', resource_type)
        for file_path in task_files_to_copy[resource_type]:
            try:
                file_name = os.path.basename(file_path)
                target_path = os.path.join(target_resource_dir, file_name)
                print('copying {} to {}'.format(file_path, target_path))
                shutil.copy2(file_path, target_path)
            except IsADirectoryError:  # noqa: F821
                dir_name = os.path.basename(os.path.normpath(file_path))
                shutil.copytree(
                    file_path, os.path.join(target_resource_dir, dir_name))
            except FileNotFoundError:  # noqa: F821
                pass

    # Compile the frontend
    os.chdir(heroku_server_development_path)

    packages_installed = subprocess.call(['npm', 'install'])
    if packages_installed != 0:
        raise Exception('please make sure npm is installed, otherwise view '
                        'the above error for more info.')

    webpack_complete = subprocess.call(['npm', 'run', 'dev'])
    if webpack_complete != 0:
        raise Exception('Webpack appears to have failed to build your '
                        'frontend. See the above error for more information.')

    # all the important files should've been moved to bundle.js in
    # server/static, now copy the rest into static
    target_resource_dir = os.path.join(
        heroku_server_development_path, 'server', 'static')
    for file_path in task_files_to_copy['static']:
        try:
            file_name = os.path.basename(file_path)
            target_path = os.path.join(target_resource_dir, file_name)
            shutil.copy2(file_path, target_path)
        except IsADirectoryError:  # noqa: F821 we don't support python2
            dir_name = os.path.basename(os.path.normpath(file_path))
            shutil.copytree(
                file_path, os.path.join(target_resource_dir, dir_name))
        except FileNotFoundError:  # noqa: F821 we don't support python2
            pass

    hit_config_file_path = os.path.join(tmp_dir, 'hit_config.json')
    sh.mv(hit_config_file_path, target_resource_dir)

    print("Heroku: Starting server...")

    heroku_server_directory_path = os.path.join(
        heroku_server_development_path, 'server')
    os.chdir(heroku_server_directory_path)
    sh.git('init')

    # get heroku credentials
    heroku_user_identifier = None
    while not heroku_user_identifier:
        try:
            subprocess.check_output(
                shlex.split(heroku_executable_path + ' auth:token')
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
        if heroku_team is not None:
            subprocess.check_output(shlex.split(
                '{} create {} --team {}'.format(
                    heroku_executable_path,
                    heroku_app_name,
                    heroku_team
                )
            ))
        else:
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

    if use_hobby:
        try:
            subprocess.check_output(shlex.split('{} dyno:type Hobby'.format(
                heroku_executable_path)
            ))
        except subprocess.CalledProcessError:  # User doesn't have hobby access
            delete_heroku_server(task_name)
            sh.rm(shlex.split('-rf {}'.format(heroku_server_directory_path)))
            raise SystemExit(
                'Server launched with hobby flag but account cannot create '
                'hobby servers.'
            )
    os.chdir(parent_dir)

    # Clean up heroku files
    if os.path.exists(os.path.join(tmp_dir, 'heroku.tar.gz')):
        os.remove(os.path.join(tmp_dir, 'heroku.tar.gz'))

    sh.rm(shlex.split('-rf {}'.format(heroku_server_development_path)))

    return 'https://{}.herokuapp.com'.format(heroku_app_name)


def delete_heroku_server(task_name, tmp_dir=parent_dir):
    heroku_directory_name = \
        glob.glob(os.path.join(tmp_dir, 'heroku-cli-*'))[0]
    heroku_directory_path = os.path.join(tmp_dir, heroku_directory_name)
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


def setup_local_server(task_name, task_files_to_copy=None):
    global server_process
    print("Local Server: Collecting files...")

    server_source_directory_path = \
        os.path.join(parent_dir, legacy_server_source_directory_name)
    local_server_directory_path = os.path.join(parent_dir, '{}_{}'.format(
        local_server_directory_name,
        task_name
    ))

    # Delete old server files
    sh.rm(shlex.split('-rf ' + local_server_directory_path))

    # Copy over a clean copy into the server directory
    shutil.copytree(server_source_directory_path, local_server_directory_path)

    # Consolidate task files
    task_directory_path = \
        os.path.join(local_server_directory_path, task_directory_name)
    sh.mv(
        os.path.join(local_server_directory_path, 'html'),
        task_directory_path
    )

    hit_config_file_path = os.path.join(parent_dir, 'hit_config.json')
    sh.mv(hit_config_file_path, task_directory_path)

    for file_path in task_files_to_copy:
        try:
            shutil.copy2(file_path, task_directory_path)
        except IsADirectoryError:  # noqa: F821 we don't support python2
            dir_name = os.path.basename(os.path.normpath(file_path))
            shutil.copytree(
                file_path, os.path.join(task_directory_path, dir_name))
        except FileNotFoundError:  # noqa: F821 we don't support python2
            pass

    print("Local: Starting server...")

    os.chdir(local_server_directory_path)

    packages_installed = subprocess.call(['npm', 'install'])
    if packages_installed != 0:
        raise Exception('please make sure npm is installed, otherwise view '
                        'the above error for more info.')

    server_process = subprocess.Popen(['node', 'server.js'])

    time.sleep(1)
    print('Server running locally with pid {}.'.format(server_process.pid))
    host = input(
        'Please enter the public server address, like https://hostname.com: ')
    port = input('Please enter the port given above, likely 3000: ')
    return '{}:{}'.format(host, port)


def delete_local_server(task_name):
    global server_process
    print('Terminating server')
    server_process.terminate()
    server_process.wait()
    print('Cleaning temp directory')

    local_server_directory_path = os.path.join(parent_dir, '{}_{}'.format(
        local_server_directory_name,
        task_name
    ))
    sh.rm(shlex.split('-rf {}'.format(local_server_directory_path)))


def setup_legacy_server(task_name, task_files_to_copy, local=False,
                        heroku_team=None, use_hobby=False, legacy=True,
                        tmp_dir=parent_dir):
    if local:
        return setup_local_server(
            task_name,
            task_files_to_copy=task_files_to_copy
        )
    return setup_legacy_heroku_server(
        task_name,
        task_files_to_copy=task_files_to_copy,
        heroku_team=heroku_team, use_hobby=use_hobby,
        tmp_dir=tmp_dir,
    )


def setup_server(task_name, task_files_to_copy, local=False, heroku_team=None,
                 use_hobby=False, legacy=True, tmp_dir=parent_dir):
    if local:
        raise Exception('Local server not yet supported for non-legacy tasks')
    return setup_heroku_server(
        task_name,
        task_files_to_copy=task_files_to_copy,
        heroku_team=heroku_team, use_hobby=use_hobby,
        tmp_dir=tmp_dir,
    )


def delete_server(task_name, local=False, tmp_dir=parent_dir):
    if local:
        delete_local_server(task_name)
    else:
        delete_heroku_server(task_name, tmp_dir)
