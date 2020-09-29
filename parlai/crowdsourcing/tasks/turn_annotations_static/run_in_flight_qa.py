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
from mephisto.core.hydra_config import config
from omegaconf import DictConfig

from parlai.crowdsourcing.tasks.turn_annotations_static.turn_annotations_blueprint import (
    STATIC_IN_FLIGHT_QA_BLUEPRINT_TYPE,
)
from parlai.crowdsourcing.tasks.turn_annotations_static.util import run_static_task
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig


TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ScriptConfig(MTurkRunScriptConfig):
    task_dir: str = TASK_DIRECTORY


config.store(group='script', name='scriptconfig', node=ScriptConfig)


@hydra.main(config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    run_static_task(cfg=cfg, task_directory=TASK_DIRECTORY)


if __name__ == "__main__":
    main()
