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

from parlai.crowdsourcing.tasks.model_chat.model_chat_blueprint import BLUEPRINT_TYPE
from parlai.crowdsourcing.tasks.model_chat.impl import run_task
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig
import json

"""
Read parlai/crowdsourcing/README.md to learn how to launch
crowdsourcing tasks with this script.
"""

_ = BLUEPRINT_TYPE

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


defaults = ["_self_", {"conf": "example"}]


def check_override_opt(args):
    with open(args.blueprint.override_opt.completed_run_stats) as f:
        override_opt = json.load(f)
    if (
        override_opt['bot_model_name']
        != args.blueprint.conversations_needed_string.split(":")[0]
    ):
        raise AssertionError(
            f"YOU bot model name in {args.blueprint.override_opt.completed_run_stats} doesnt match with {args.blueprint.conversations_needed_string}"
        )


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
    check_override_opt(cfg.mephisto)
    run_task(cfg=cfg, task_directory=TASK_DIRECTORY, world_module=None)


if __name__ == "__main__":
    main()
