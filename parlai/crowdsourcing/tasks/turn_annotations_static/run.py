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
from typing import Any, List

import hydra
from mephisto.core.hydra_config import RunScriptConfig, register_script_config
from mephisto.core.operator import Operator
from mephisto.providers.mturk.utils.script_utils import direct_soft_block_mturk_workers
from mephisto.utils.scripts import load_db_and_process_config
from omegaconf import DictConfig

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

    random.seed(42)

    # TODO: revise below
    arg_string = (
        f"--task-name turn-ann-s "
        f'--task-source "{TASK_DIRECTORY}/webapp/build/bundle.js" '
        f'--task-tags chat,conversation,dialog,partner '
        f"--allowed-concurrent 1 "
        f'--extra-source-dir "{STATIC_FILES_DIR}" '
        f'--port 2222 '
        f'--onboarding-qualification turn-ann-s-onb '
        f"-use-onboarding True "
    )

    if cfg.mephisto.provider.get('_provider_type', 'mock') == 'mturk':
        soft_block_qual_name = 'turn_annotations_static_no'
        print(
            f'About to soft block {len(launch_config.WORKER_BLOCK_LIST)} workers by giving qualification: {soft_block_qual_name}'
        )
        direct_soft_block_mturk_workers(
            db=db,
            worker_list=launch_config.WORKER_BLOCK_LIST,
            soft_block_qual_name=soft_block_qual_name,
            requester_name=cfg.mephisto.provider.get("requester_name", None),
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

    operator = Operator(db)
    operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=None)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)


if __name__ == "__main__":
    main()
