#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Any, List

import hydra
from mephisto.operations.hydra_config import register_script_config
from mephisto.operations.operator import Operator
from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
    BLUEPRINT_TYPE,
    SharedParlAITaskState,
)
from mephisto.tools.scripts import load_db_and_process_config
from omegaconf import DictConfig

from parlai.crowdsourcing.tasks.chat_demo.util import TASK_DIRECTORY
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig


defaults = [
    {"mephisto/blueprint": BLUEPRINT_TYPE},
    {"mephisto/architect": "local"},
    {"mephisto/provider": "mock"},
    "conf/base",
    {"conf": "example"},
]


@dataclass
class ScriptConfig(MTurkRunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY
    turn_timeout: int = field(
        default=300,
        metadata={
            "help": "Maximum response time before kicking "
            "a worker out, default 300 seconds"
        },
    )
    monitoring_log_rate: int = field(
        default=30,
        metadata={
            'help': 'Frequency in seconds of logging the monitoring of the crowdsourcing task'
        },
    )


register_script_config(name="scriptconfig", module=ScriptConfig)
relative_task_directory = os.path.relpath(TASK_DIRECTORY, os.path.dirname(__file__))


@hydra.main(config_path=relative_task_directory, config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    db, cfg = load_db_and_process_config(cfg)

    world_opt = {"turn_timeout": cfg.turn_timeout}

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
    operator.wait_for_runs_then_shutdown(
        skip_input=True, log_rate=cfg.monitoring_log_rate
    )


if __name__ == "__main__":
    main()
