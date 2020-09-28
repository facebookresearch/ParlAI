#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, List

import hydra
from mephisto.core.hydra_config import RunScriptConfig, register_script_config
from mephisto.core.operator import Operator
from mephisto.providers.mturk.utils.script_utils import direct_soft_block_mturk_workers
from mephisto.utils.scripts import load_db_and_process_config
from omegaconf import DictConfig

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
from parlai.crowdsourcing.tasks.turn_annotations_static.turn_annotations_blueprint import (
    STATIC_BLUEPRINT_TYPE,
)

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
FRONTEND_SOURCE_DIR = os.path.join(TASK_DIRECTORY, "webapp")
FRONTEND_BUILD_DIR = os.path.join(FRONTEND_SOURCE_DIR, "build")
STATIC_FILES_DIR = os.path.join(FRONTEND_SOURCE_DIR, "src", "static")

defaults = [
    {"mephisto/blueprint": STATIC_BLUEPRINT_TYPE},
    {"mephisto/architect": "local"},
    {"mephisto/provider": "mock"},
    {"conf": "example"},
]


@dataclass
class ScriptConfig(RunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY
    current_time: int = int(time.time())


register_script_config(name='scriptconfig', module=ScriptConfig)


@hydra.main(config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    db, cfg = load_db_and_process_config(cfg)
    operator = Operator(db)
    operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=None)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)


if __name__ == "__main__":
    main()


def block_workers(block_list: List[str], provider: str, local_db, requester_name=None):
    """
    Use a block list to block undesired crowdsource workers (Mechanical Turkers for
    example).
    """
    if not requester_name:
        print(
            'Specified no requester_name, which is done when running locally. Skipping blocking...'
        )
        return

    soft_block_qual_name = 'turn_annotations_static_no'
    print(
        f'About to soft block {len(block_list)} workers on {provider} by giving qualification: {soft_block_qual_name}'
    )
    direct_soft_block_mturk_workers(
        db=local_db,
        worker_list=block_list,
        soft_block_qual_name=soft_block_qual_name,
        requester_name=requester_name,
    )


def run_task(opt):

    random.seed(42)

    # Set up Mephisto
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

    arg_string = (
        "--blueprint-type turn_annotations_static_inflight_qa_blueprint "
        f"--architect-type {architect_type} "
        f"--requester-name {requester_name} "
        f'--task-title "\\"{launch_config.TASK_TITLE}\\"" '
        f'--task-description "\\"{launch_config.TASK_DESCRIPTION}\\"" '
        f"--task-name turn-ann-s "
        f'--task-source "{TASK_DIRECTORY}/webapp/build/bundle.js" '
        f'--task-reward {launch_config.TASK_REWARD} '
        f'--subtasks-per-unit {launch_config.SUBTASKS_PER_UNIT} '
        f'--annotate-last-utterance-only {launch_config.ANNOTATE_LAST_UTTERANCE_ONLY} '
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

    if 'sandbox' not in launch_config.PROVIDER:
        block_workers(
            block_list=launch_config.WORKER_BLOCK_LIST,
            provider=launch_config.PROVIDER,
            local_db=db,
            requester_name=launch_config.REQUESTER,
        )

    # Build the task
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
