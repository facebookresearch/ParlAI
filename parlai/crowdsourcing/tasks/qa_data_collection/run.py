#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import List, Any

import hydra
from omegaconf import DictConfig
from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
    SharedParlAITaskState,
)
from mephisto.operations.hydra_config import register_script_config
from mephisto.operations.operator import Operator
from mephisto.tools.scripts import load_db_and_process_config

from parlai.crowdsourcing.tasks.qa_data_collection.util import get_teacher
from parlai.crowdsourcing.utils.frontend import build_task
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig

"""
Read parlai/crowdsourcing/README.md to learn how to launch
crowdsourcing tasks with this script.
"""

TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

defaults = ["_self_", {"conf": "example"}]


@dataclass
class TeacherConfig:
    task: str = field(default="squad:SquadQATeacher", metadata={"help": ""})
    datatype: str = field(default="train", metadata={"help": ""})


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
    turn_timeout: int = field(
        default=300,
        metadata={
            "help": "Maximum response time before kicking "
            "a worker out, default 300 seconds"
        },
    )
    teacher: TeacherConfig = TeacherConfig()


register_script_config(name="scriptconfig", module=ScriptConfig)


@hydra.main(config_path="hydra_configs", config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    db, cfg = load_db_and_process_config(cfg)

    teacher = get_teacher(cfg)
    world_opt = {"turn_timeout": cfg.turn_timeout, "teacher": teacher}

    custom_bundle_path = cfg.mephisto.blueprint.get("custom_source_bundle", None)
    if custom_bundle_path is not None:
        if not os.path.exists(custom_bundle_path):
            build_task(TASK_DIRECTORY)

    shared_state = SharedParlAITaskState(
        world_opt=world_opt, onboarding_world_opt=world_opt
    )

    operator = Operator(db)
    operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=shared_state)
    operator.wait_for_runs_then_shutdown(
        skip_input=True, log_rate=cfg.monitoring_log_rate
    )


if __name__ == "__main__":
    main()
