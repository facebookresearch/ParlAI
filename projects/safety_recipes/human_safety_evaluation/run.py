#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Any, List

import hydra
from mephisto.core.hydra_config import register_script_config
from omegaconf import DictConfig

from parlai.crowdsourcing.tasks.turn_annotations_static.util import run_static_task
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig
from parlai.crowdsourcing.tasks.turn_annotations_static.run import (
    TASK_DIRECTORY as BASE_TASK_DIRECTORY,
)
from parlai.crowdsourcing.tasks.turn_annotations_static.run import defaults

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ScriptConfig(MTurkRunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    base_task_dir: str = BASE_TASK_DIRECTORY
    task_dir: str = TASK_DIRECTORY
    monitoring_log_rate: int = field(
        default=30,
        metadata={
            'help': 'Frequency in seconds of logging the monitoring of the crowdsourcing task'
        },
    )


register_script_config(name='scriptconfig', module=ScriptConfig)


@hydra.main(config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    run_static_task(cfg=cfg, task_directory=BASE_TASK_DIRECTORY)


if __name__ == "__main__":
    main()
