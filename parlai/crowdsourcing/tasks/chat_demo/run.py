#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field

import hydra
from omegaconf import DictConfig

# From the Mephisto repo
from examples.parlai_chat_task_demo.parlai_test_script import (
    TestScriptConfig,
    TASK_DIRECTORY,
)
from mephisto.operations.hydra_config import register_script_config
from mephisto.operations.operator import Operator
from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
    SharedParlAITaskState,
)
from mephisto.tools.scripts import load_db_and_process_config

from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfigMixin

"""
Read parlai/crowdsourcing/README.md to learn how to launch
crowdsourcing tasks with this script.
"""


@dataclass
class ScriptConfig(MTurkRunScriptConfigMixin, TestScriptConfig):
    """
    Use the mixin's soft-blocking with the defaults from the Mephisto version.
    """

    monitoring_log_rate: int = field(
        default=30,
        metadata={
            'help': 'Frequency in seconds of logging the monitoring of the crowdsourcing task'
        },
    )


register_script_config(name="scriptconfig", module=ScriptConfig)
relative_task_directory = os.path.relpath(TASK_DIRECTORY, os.path.dirname(__file__))
config_path = os.path.join(relative_task_directory, 'hydra_configs')


@hydra.main(config_path=config_path, config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    db, cfg = load_db_and_process_config(cfg)

    world_opt = {"num_turns": cfg.num_turns, "turn_timeout": cfg.turn_timeout}

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
