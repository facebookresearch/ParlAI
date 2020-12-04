#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# python run.py mephisto/architect=heroku mephisto.provider.requester_name=$REQUESTER_NAME conf=mturk_sandbox # fewer tasks
# python run.py mephisto/architect=heroku mephisto.provider.requester_name=$REQUESTER_NAME conf=mturk

import argparse
from dataclasses import dataclass, field
import os
from typing import Any, List

import hydra
from mephisto.operations.hydra_config import register_script_config
from omegaconf import DictConfig

from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig
from parlai_internal.mturk.tasks.empathy_labeled_utterances.scripts.util import (
    run_static_task,
)
from parlai_internal.mturk.tasks.empathy_labeled_utterances.scripts.turn_annotations_blueprint import (
    STATIC_BLUEPRINT_TYPE,
)

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

defaults = [
    {'mephisto/blueprint': STATIC_BLUEPRINT_TYPE},
    {'mephisto/architect': 'local'},
    {'mephisto/provider': 'mock'},
    {'conf': 'sample'},
]


@dataclass
class ScriptConfig(MTurkRunScriptConfig):
    defaults_key: str = 'mock'
    defaults: List[Any] = field(default_factory=lambda: defaults)
    base_task_dir: str = TASK_DIRECTORY
    task_dir: str = TASK_DIRECTORY
    monitoring_log_rate: int = field(
        default=30,
        metadata={
            'help': 'Frequency in seconds of logging the monitoring of the crowdsourcing task'
        },
    )


register_script_config(name='scriptconfig', module=ScriptConfig)


@hydra.main(config_name='scriptconfig')
def main(cfg: DictConfig) -> None:
    run_static_task(cfg=cfg, task_directory=TASK_DIRECTORY)


if __name__ == "__main__":
    main()
