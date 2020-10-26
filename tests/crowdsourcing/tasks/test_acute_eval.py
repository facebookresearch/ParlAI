#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test the ACUTE-Eval crowdsourcing task.
"""

import unittest
from dataclasses import field
from typing import Any, List

import hydra
from mephisto.core.hydra_config import register_script_config
from mephisto.core.operator import Operator
from mephisto.utils.scripts import load_db_and_process_config
from omegaconf import DictConfig, OmegaConf

from parlai.crowdsourcing.tasks.acute_eval.run import ScriptConfig, defaults


test_defaults = {**defaults, 'mephisto/architect': 'mock', 'mephisto/provider': 'mock'}


class TestScriptConfig(ScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: test_defaults)


class TestAcuteEval(unittest.TestCase):
    """
    Test the ACUTE-Eval crowdsourcing task.
    """

    @hydra.main(config_name="test_script_config")
    def main(self, cfg: DictConfig) -> None:
        db, cfg = load_db_and_process_config(cfg)
        operator = Operator(db)
        operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=None)
        print(OmegaConf.to_yaml(cfg))
        print(operator.supervisor.channels[-1].job.architect.server)
        # TODO: remove 2
        # {{{TODO: do all per-turn testing and end-state testing}}}

    def test_base_task(self):
        register_script_config(name='test_script_config', module=TestScriptConfig)
        self.main()


if __name__ == "__main__":
    unittest.main()
