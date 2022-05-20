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

from parlai.crowdsourcing.tasks.pairwise_per_turn_eval.per_turn_eval_blueprint import (
    BLUEPRINT_TYPE,
)
from parlai.crowdsourcing.tasks.pairwise_per_turn_eval.impl import run_task
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig

"""
Read parlai/crowdsourcing/README.md to learn how to launch
crowdsourcing tasks with this script.
"""

_ = BLUEPRINT_TYPE


TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


defaults = ['_self_', {"conf": "example_model_comparison"}]


@dataclass
class PerTurnEvalConfig(MTurkRunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY
    monitoring_log_rate: int = field(
        default=30,
        metadata={
            'help': 'Frequency in seconds of logging the monitoring of the crowdsourcing task'
        },
    )


register_script_config(name='perturnevalconfig', module=PerTurnEvalConfig)


@hydra.main(config_path="hydra_configs", config_name="perturnevalconfig")
def main(cfg: DictConfig) -> None:
    run_task(cfg=cfg, task_directory=TASK_DIRECTORY)


if __name__ == "__main__":
    main()
