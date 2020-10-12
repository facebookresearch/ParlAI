#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shlex
import shutil
import subprocess
import random
import logging

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
from mephisto.core.operator import Operator

from importlib import import_module

# Have to import this even though it's not directly used
from parlai.crowdsourcing.tasks.turn_annotations_static import (
    turn_annotations_blueprint,
)

from mephisto.utils.scripts import MephistoRunScriptParser
from mephisto.providers.mturk.utils.script_utils import direct_soft_block_mturk_workers

LAUNCH_FILE = None

# Blueprint import required though not used; this satisfies linting
_ = turn_annotations_blueprint

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
FRONTEND_SOURCE_DIR = os.path.join(TASK_DIRECTORY, "webapp")
FRONTEND_BUILD_DIR = os.path.join(FRONTEND_SOURCE_DIR, "build")
STATIC_FILES_DIR = os.path.join(FRONTEND_SOURCE_DIR, "src", "static")


def setup_mephisto(launch_config):
    """
    Mephisto data is saved in:

    <datapath>/data/runs/NO_PROJECT/<project_id>/<task_run_id>/<assignment_id>/<agent_id>/data
    -- The NO_PROJECT and /data/ thing will fixed in later versions of Mephisto
    -- agent_id can be mapped to MTurk worker_id with workers table in sqlite3 db
    """
    parser = MephistoRunScriptParser()
    full_requester_name = launch_config.REQUESTER
    if 'sandbox' in launch_config.PROVIDER and launch_config.REQUESTER:
        full_requester_name = full_requester_name + '_sandbox'

    parser.set_defaults(
        **{
            'provider_type': launch_config.PROVIDER,
            'requester_name': full_requester_name,
            'datapath': launch_config.DATAPATH,
        }
    )

    architect_type, requester_name, db, args = parser.parse_launch_arguments()

    arg_string = (
        f"--blueprint-type {launch_config.BLUEPRINT_TYPE} "
        f"--architect-type {architect_type} "
        f"--requester-name {requester_name} "
        f'--task-title "\\"{launch_config.TASK_TITLE}\\"" '
        f'--task-description "\\"{launch_config.TASK_DESCRIPTION}\\"" '
        f'--task-name {launch_config.TASK_NAME} '
        f'--task-source "{TASK_DIRECTORY}/webapp/build/bundle.js" '
        f'--task-reward {launch_config.TASK_REWARD} '
        f'--subtasks-per-unit {launch_config.SUBTASKS_PER_UNIT} '
        f'--annotation-buckets {launch_config.ANNOTATION_BUCKETS} '
        f'--ask-reason {launch_config.ASK_REASON} '
        f'--task-tags chat,conversation,dialog,partner '
        # How many workers to do each assignment
        f"--units-per-assignment {launch_config.UNITS_PER_ASSIGNMENT} "
        # Maximum tasks a worker can do across all runs with task_name (0=infinite)
        f"--maximum-units-per-worker {launch_config.MAX_UNITS_PER_WORKER} "
        f"--allowed-concurrent 1 "
        f'--data-jsonl {launch_config.FILE_DATA_JSONL} '
        f'--extra-source-dir "{STATIC_FILES_DIR}" '
        f'--port 2222 '
        f'--onboarding-qualification turn-ann-s-onb '
        f"-use-onboarding True "
    )
    # Optional flags:
    try:
        arg_string += (
            f'--annotation-question "\\"{launch_config.ANNOTATION_QUESTION}\\"" '
        )
    except Exception:
        logging.info(f'Launch config {launch_config} had no ANNOTATION_QUESTION field')

    try:
        arg_string += (
            f'--annotation-indices-jsonl {launch_config.ANNOTATION_INDICES_JSONL} '
        )
    except Exception:
        logging.info(f'Launch config {launch_config} had no ANNOTATION_INDICES_JSONL')

    try:
        arg_string += f'--conversation-count {launch_config.CONVERSATION_COUNT} '
    except Exception:
        logging.info(f'Launch config {launch_config} had no CONVERSATION_COUNT')

    try:
        arg_string += f'--onboarding-data {launch_config.ONBOARDING_DATA} '
    except Exception:
        logging.info(f'Launch config {launch_config} had no ONBOARDING_DATA')

    print(arg_string)
    return db, arg_string


def build_task():
    """
    Build the task.
    """
    return_dir = os.getcwd()
    os.chdir(FRONTEND_SOURCE_DIR)
    if os.path.exists(FRONTEND_BUILD_DIR):
        shutil.rmtree(FRONTEND_BUILD_DIR)
    packages_installed = subprocess.call(["npm", "install"])
    if packages_installed != 0:
        raise Exception(
            "please make sure npm is installed, otherwise view "
            "the above error for more info."
        )

    webpack_complete = subprocess.call(["npm", "run", "dev"])
    if webpack_complete != 0:
        raise RuntimeError(
            "Webpack appears to have failed to build your "
            "frontend. See the above error for more information."
        )
    os.chdir(return_dir)


def block_workers(launch_config, opt, local_db, requester_name=None):
    """
    Use a block list to block undesired crowdsource workers (Mechanical Turkers for
    example).
    """
    if not requester_name:
        print(
            'Specified no requester_name, which is done when running locally. Skipping blocking...'
        )
        return

    block_list = launch_config.WORKER_BLOCK_LIST
    soft_block_qual_name = 'turn_annotations_static_no'
    print(
        f'About to soft block {len(block_list)} workers on {launch_config.PROVIDER} by giving qualification: {soft_block_qual_name}'
    )
    direct_soft_block_mturk_workers(
        local_db, block_list, soft_block_qual_name, requester_name
    )


def run_task(opt):
    """
    Note: launch_config opt should be something like:
    parlai.crowdsourcing.tasks.turn_annotations_static.launch_config.LaunchConfig
    """

    launch_config_file = opt.get('launch_config')
    launch_module = import_module(launch_config_file)
    launch_config = launch_module.LaunchConfig
    db, arg_string = setup_mephisto(launch_config)
    if 'sandbox' not in launch_config.PROVIDER:
        block_workers(opt, db, launch_config.REQUESTER)

    build_task()
    operator = Operator(db)
    operator.parse_and_launch_run_wrapper(shlex.split(arg_string), extra_args={})
    operator.wait_for_runs_then_shutdown()


def setup_args(parser=None):
    """
    This task takes a single argument which is a Python class specifying the launch
    config.
    """
    if parser is None:
        parser = ParlaiParser(True, False, 'Static Turn Annotations Task')
    parser.add_argument(
        '--launch-config',
        type=str,
        required=True,
        help='A Python config file with variables used in this script.',
    )
    return parser


class TurnAnnotationsStaticRunner(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return run_task(self.opt)


if __name__ == '__main__':
    random.seed(42)
    TurnAnnotationsStaticRunner.main()
