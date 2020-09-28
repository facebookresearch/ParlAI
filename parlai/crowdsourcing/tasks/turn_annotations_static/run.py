#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, List

import hydra
from mephisto.core.hydra_config import register_script_config
from mephisto.core.operator import Operator
from mephisto.utils.scripts import load_db_and_process_config
from omegaconf import DictConfig

# Have to import this even though it's not directly used, in order to register the
# blueprints
from parlai.crowdsourcing.tasks.turn_annotations_static import (
    turn_annotations_blueprint,
)
from parlai.crowdsourcing.utils.mturk import (
    MTurkRunScriptConfig,
    soft_block_mturk_workers,
)

# Satisfy linting
_ = turn_annotations_blueprint

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
FRONTEND_SOURCE_DIR = os.path.join(TASK_DIRECTORY, "webapp")
FRONTEND_BUILD_DIR = os.path.join(FRONTEND_SOURCE_DIR, "build")

defaults = [
    {"mephisto/architect": "local"},
    {"mephisto/provider": "mock"},
    {"conf": "example"},
]
# NOTE: here, mephisto/blueprint should be specified in the YAML file


@dataclass
class ScriptConfig(MTurkRunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY


register_script_config(name='scriptconfig', module=ScriptConfig)


@hydra.main(config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    db, cfg = load_db_and_process_config(cfg)

    random.seed(42)

    soft_block_qual_name = cfg.mephisto.task.get('task_name', 'turn_annotations_static')
    # Default to a task-specific name to avoid soft-block collisions
    soft_block_mturk_workers(cfg=cfg, db=db, soft_block_qual_name=soft_block_qual_name)

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
