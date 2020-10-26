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
from mephisto.core.hydra_config import register_script_config
from mephisto.core.operator import Operator
from mephisto.utils.scripts import load_db_and_process_config
from omegaconf import DictConfig, OmegaConf

from parlai.crowdsourcing.tasks.acute_eval.run import (
    ScriptConfig,
    defaults,
    TASK_DIRECTORY,
)


test_defaults = defaults + [
    {'mephisto/architect': 'mock'},
    {'mephisto/provider': 'mock'},
]

relative_task_directory = os.path.relpath(TASK_DIRECTORY, os.path.dirname(__file__))


@dataclass
class TestScriptConfig(ScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: test_defaults)


@hydra.main(config_path=relative_task_directory, config_name="test_script_config")
def main(cfg: DictConfig) -> None:

    # Set up the mock server
    db, cfg = load_db_and_process_config(cfg)
    cfg.mephisto.architect.should_run_server = True
    operator = Operator(db)
    operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=None)
    sup = operator.supervisor
    assert len(sup.channels) == 1
    channel = list(sup.channels.keys())[0]
    server = sup.channels[channel].job.architect.server
    print(OmegaConf.to_yaml(cfg))
    # TODO: remove print statement

    # # Handle baseline setup
    # task_runner = TaskRunnerClass(self.task_run, config, EMPTY_STATE)
    # sup.register_job(self.architect, task_runner, self.provider)
    # self.assertEqual(len(sup.channels), 1)
    # channel_info = list(sup.channels.values())[0]
    # self.assertIsNotNone(channel_info)
    # self.assertTrue(channel_info.channel.is_alive())
    # channel_id = channel_info.channel_id
    # task_runner = channel_info.job.task_runner
    # self.assertIsNotNone(channel_id)
    # self.assertEqual(
    #     len(self.architect.server.subs),
    #     1,
    #     "MockServer doesn't see registered channel",
    # )
    # self.assertIsNotNone(
    #     self.architect.server.last_alive_packet,
    #     "No alive packet received by server",
    # )
    # sup.launch_sending_thread()
    # self.assertIsNotNone(sup.sending_thread)

    # Create a mock worker agent
    mock_worker_name = "MOCK_WORKER"
    server.register_mock_worker(mock_worker_name)
    workers = db.find_workers(worker_name=mock_worker_name)
    worker_id = workers[0].db_id
    mock_agent_details = "FAKE_ASSIGNMENT"
    server.register_mock_agent(worker_id, mock_agent_details)
    agent_id = db.find_agents()[0].db_id

    print("SENDING_THREAD: ", sup.sending_thread)

    # {{{TODO: do all per-turn testing and end-state testing}}}

    sup.shutdown()


class TestAcuteEval(unittest.TestCase):
    """
    Test the ACUTE-Eval crowdsourcing task.
    """

    def test_base_task(self):
        register_script_config(name='test_script_config', module=TestScriptConfig)
        main()


if __name__ == "__main__":
    unittest.main()
