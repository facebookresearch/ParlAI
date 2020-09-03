#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shlex
import shutil
import subprocess

from mephisto.core.operator import Operator

# Have to import this even though it's not directly used
from parlai.crowdsourcing.tasks.turn_annotations_static import turn_annotations_blueprint

from mephisto.utils.scripts import MephistoRunScriptParser
from mephisto.providers.mturk.utils.script_utils import direct_soft_block_mturk_workers
from launch_config import (
    REQUESTER,
    PROVIDER,
    DATAPATH,
    TASK_DESCRIPTION,
    TASK_TITLE,
    FILE_DATA_JSONL,
    WORKER_BLOCK_LIST,
    TASK_REWARD,
    SUBTASKS_PER_UNIT,
    UNITS_PER_ASSIGNMENT,
    MAX_UNITS_PER_WORKER
)

# Blueprint import required though not used; this satisfies linting
_ = turn_annotations_blueprint

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
FRONTEND_SOURCE_DIR = os.path.join(TASK_DIRECTORY, "webapp")
FRONTEND_BUILD_DIR = os.path.join(FRONTEND_SOURCE_DIR, "build")
STATIC_FILES_DIR = os.path.join(FRONTEND_SOURCE_DIR, "src", "static")

"""
Data is saved in:
<datapath>/data/runs/NO_PROJECT/<project_id>/<task_run_id>/<assignment_id>/<agent_id>/data
-- The NO_PROJECT and /data/ thing will fixed in later versions of Mephisto
-- agent_id can be mapped to MTurk worker_id with workers table in sqlite3 db
"""

parser = MephistoRunScriptParser()
parser.set_defaults(
    **{
        'provider_type': PROVIDER,
        'requester_name': REQUESTER + ('_sandbox' if 'sandbox' in PROVIDER else ''),
        'datapath': DATAPATH,
    }
)

architect_type, requester_name, db, args = parser.parse_launch_arguments()

ARG_STRING = (
    "--blueprint-type turn_annotations_static_inflight_qa_blueprint "
    f"--architect-type {architect_type} "
    f"--requester-name {requester_name} "
    f'--task-title "\\"{TASK_TITLE}\\"" '
    f'--task-description "\\"{TASK_DESCRIPTION}\\"" '
    f"--task-name ta-static "
    f'--task-source "{TASK_DIRECTORY}/webapp/build/bundle.js" '
    f'--task-reward {TASK_REWARD} '
    f'--subtasks-per-unit {SUBTASKS_PER_UNIT} '
    f'--annotate-last-utterance-only True '
    f'--task-tags chat,conversation,dialog '
    # How many workers to do each assignment
    f"--units-per-assignment {UNITS_PER_ASSIGNMENT} "
    # Maximum tasks a worker can do across all runs with task_name (0=infinite)
    f"--maximum-units-per-worker {MAX_UNITS_PER_WORKER} "
    f"--allowed-concurrent 1 "
    f'--data-jsonl {FILE_DATA_JSONL} '
    f'--extra-source-dir "{STATIC_FILES_DIR}" '
    f'--port 8888 '
    f'--onboarding-qualification turn-annotation-static-onb '
    f'--use-onboarding True '
)


def build_task():
    """
    Build the task
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
        raise Exception(
            "Webpack appears to have failed to build your "
            "frontend. See the above error for more information."
        )
    os.chdir(return_dir)


def block_workers(local_db, requester_name):
    """Use a block list to block undesired crowdsource workers (Mechanical Turkers for example)."""
    block_list = WORKER_BLOCK_LIST
    soft_block_qual_name = 'turn_annotations_static_no'
    print(
        f'About to soft block {len(block_list)} workers on {PROVIDER} by giving qualification: {soft_block_qual_name}'
    )
    direct_soft_block_mturk_workers(
        local_db, block_list, soft_block_qual_name, requester_name
    )


if __name__ == '__main__':
    if 'sandbox' not in PROVIDER:
        block_workers(db, REQUESTER)

    build_task()

    operator = Operator(db)
    operator.parse_and_launch_run_wrapper(shlex.split(ARG_STRING), extra_args={})
    operator.wait_for_runs_then_shutdown()
