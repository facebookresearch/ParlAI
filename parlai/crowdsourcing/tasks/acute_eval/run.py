#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, List

import hydra
from mephisto.operations.hydra_config import register_script_config
from mephisto.operations.operator import Operator
from mephisto.tools.scripts import load_db_and_process_config
from omegaconf import DictConfig

from parlai.crowdsourcing.tasks.acute_eval.acute_eval_blueprint import BLUEPRINT_TYPE
from parlai.crowdsourcing.tasks.acute_eval.util import TASK_DIRECTORY
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig


"""
Read parlai/crowdsourcing/README.md to learn how to launch
crowdsourcing tasks with this script.

Script for running ACUTE-Evals.
The only argument that *must* be set for this to be run is:
``pairings_filepath``:  Path to pairings file in the format specified in the README.md

The following args are useful to tweak to fit your specific needs;
- ``annotations_per_pair``: A useful arg if you'd like to evaluate a given conversation pair more than once.
- ``num_matchup_pairs``: Essentially, how many pairs of conversations you would like to evaluate
- ``subtasks_per_unit``: How many comparisons you'd like a turker to complete in one HIT

"""

_ = BLUEPRINT_TYPE

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
    db, cfg = load_db_and_process_config(cfg)
    print(f'*** RUN ID: {cfg.mephisto.task.task_name} ***')
    operator = Operator(db)
    operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=None)
    operator.wait_for_runs_then_shutdown(
        skip_input=True, log_rate=cfg.monitoring_log_rate
    )


if __name__ == "__main__":
    main()
