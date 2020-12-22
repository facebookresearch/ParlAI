#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from dataclasses import dataclass, field
from itertools import chain
from typing import List, Any

import hydra
from omegaconf import DictConfig
from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
    BLUEPRINT_TYPE,
    SharedParlAITaskState,
)
from mephisto.operations.hydra_config import RunScriptConfig, register_script_config
from mephisto.operations.operator import Operator
from mephisto.tools.scripts import load_db_and_process_config

from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task


TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

defaults = [
    {"mephisto/blueprint": BLUEPRINT_TYPE},
    {"mephisto/architect": "local"},
    {"mephisto/provider": "mock"},
    {"conf": "example"},
]


@dataclass
class TeacherConfig:
    task: str = field(default="squad:SquadQATeacher", metadata={"help": ""})
    datatype: str = field(default="train", metadata={"help": ""})


@dataclass
class TestScriptConfig(RunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY
    turn_timeout: int = field(
        default=300,
        metadata={
            "help": "Maximum response time before kicking "
            "a worker out, default 300 seconds"
        },
    )
    teacher: TeacherConfig = TeacherConfig()


register_script_config(name="scriptconfig", module=TestScriptConfig)


@hydra.main(config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    db, cfg = load_db_and_process_config(cfg)

    parser = ParlaiParser(True, False)
    opt = parser.parse_args(
        list(chain.from_iterable(('--' + k, v) for k, v in cfg.teacher.items()))
    )
    agent = RepeatLabelAgent(opt)
    teacher = create_task(opt, agent).get_task_agent()

    world_opt = {"turn_timeout": cfg.turn_timeout, "teacher": teacher}

    custom_bundle_path = cfg.mephisto.blueprint.get("custom_source_bundle", None)
    if custom_bundle_path is not None:
        assert os.path.exists(custom_bundle_path), (
            "Must build the custom bundle with `npm install; npm run dev` from within "
            f"the {TASK_DIRECTORY}/webapp directory in order to demo a custom bundle "
        )
        world_opt["send_task_data"] = True

    shared_state = SharedParlAITaskState(
        world_opt=world_opt, onboarding_world_opt=world_opt
    )

    operator = Operator(db)

    operator.validate_and_run_config(cfg.mephisto, shared_state)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)


if __name__ == "__main__":
    main()
