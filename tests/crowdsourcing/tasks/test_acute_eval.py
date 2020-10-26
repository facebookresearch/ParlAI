#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test the ACUTE-Eval crowdsourcing task.
"""

import os
import unittest
from dataclasses import dataclass, field
from typing import Any, List

import hydra
from hydra.experimental import compose, initialize_config_dir
from mephisto.core.hydra_config import register_script_config
from mephisto.core.operator import Operator
from mephisto.utils.scripts import load_db_and_process_config
from omegaconf import DictConfig, OmegaConf

from parlai.crowdsourcing.tasks.acute_eval.acute_eval_blueprint import BLUEPRINT_TYPE
from parlai.crowdsourcing.tasks.acute_eval.run import (
    ScriptConfig,
    defaults,
    TASK_DIRECTORY,
)


test_defaults = defaults + [
    {'mephisto/architect': 'mock'},
    {'mephisto/provider': 'mock'},
]

# common_prefix = os.path.commonpath([__file__, TASK_DIRECTORY])
relative_task_directory = os.path.relpath(TASK_DIRECTORY, os.path.dirname(__file__))


@dataclass
class TestScriptConfig(ScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: test_defaults)


@hydra.main(config_path=relative_task_directory, config_name="test_script_config")
def main(cfg: DictConfig) -> None:
    db, cfg = load_db_and_process_config(cfg)
    operator = Operator(db)
    operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=None)
    print(OmegaConf.to_yaml(cfg))
    print(operator.supervisor.channels[-1].job.architect.server)
    # TODO: remove 2
    # {{{TODO: do all per-turn testing and end-state testing}}}


class TestAcuteEval(unittest.TestCase):
    """
    Test the ACUTE-Eval crowdsourcing task.
    """

    def test_base_task(self):
        # with initialize_config_dir(config_dir=os.path.join(TASK_DIRECTORY, 'conf')):
        # overrides = [f'+mephisto/blueprint={BLUEPRINT_TYPE}', '+mephisto/architect=mock', '+mephisto/provider=mock', f'+task_dir={TASK_DIRECTORY}']
        # cfg = compose(config_name='example', overrides=overrides)
        register_script_config(name='test_script_config', module=TestScriptConfig)
        main()


if __name__ == "__main__":
    unittest.main()
