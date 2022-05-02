#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Any, List

import hydra
from mephisto.operations.hydra_config import register_script_config
from omegaconf import DictConfig

from parlai.crowdsourcing.tasks.dialcrowd.dialcrowd_blueprint import (
    STATIC_BLUEPRINT_TYPE,
)
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig, run_static_task

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

_ = STATIC_BLUEPRINT_TYPE

defaults = ["_self_", {"conf": "example"}]


@dataclass
class ScriptConfig(MTurkRunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY
    monitoring_log_rate: int = field(
        default=30,
        metadata={
            'help': 'Frequency in seconds of logging the monitoring of the crowdsourcing task'
        },
    )


register_script_config(name='scriptconfig', module=ScriptConfig)


@hydra.main(config_path="hydra_configs", config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    run_static_task(cfg=cfg, task_directory=TASK_DIRECTORY, task_id='dialcrowd')


if __name__ == "__main__":
    main()
