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

from parlai.crowdsourcing.tasks.model_chat.model_chat_blueprint import (
    IMAGE_CHAT_BLUEPRINT_TYPE,
)
from parlai.crowdsourcing.tasks.model_chat.impl import run_task
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig


TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


defaults = [
    {'mephisto/blueprint': IMAGE_CHAT_BLUEPRINT_TYPE},
    {"mephisto/architect": "local"},
    {"mephisto/provider": "mock"},
    {"conf": "example_image_chat"},
]


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


@hydra.main(config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    run_task(cfg=cfg, task_directory=TASK_DIRECTORY)


if __name__ == "__main__":
    main()
