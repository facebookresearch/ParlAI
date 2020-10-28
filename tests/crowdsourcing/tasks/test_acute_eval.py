#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test the ACUTE-Eval crowdsourcing task.
"""

import os
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import Any, List

import hydra
from mephisto.core.hydra_config import register_script_config
from mephisto.core.local_database import LocalMephistoDB
from mephisto.core.operator import Operator
from mephisto.utils.scripts import augment_config_from_db
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
    data_dir = tempfile.mkdtemp()
    database_path = os.path.join(data_dir, "mephisto.db")
    db = LocalMephistoDB(database_path)
    cfg = augment_config_from_db(cfg, db)
    cfg.mephisto.architect.should_run_server = True
    operator = Operator(db)
    operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=None)
    sup = operator.supervisor
    assert len(sup.channels) == 1
    channel = list(sup.channels.keys())[0]
    server = sup.channels[channel].job.architect.server
    print(OmegaConf.to_yaml(cfg))
    # TODO: remove print statement

    # Create a mock worker agent
    mock_worker_name = "MOCK_WORKER"
    server.register_mock_worker(mock_worker_name)
    workers = db.find_workers(worker_name=mock_worker_name)
    worker_id = workers[0].db_id
    mock_agent_details = "FAKE_ASSIGNMENT"
    server.register_mock_agent(worker_id, mock_agent_details)
    agent_id_1 = db.find_agents()[0].db_id

    print("SENDING_THREAD: ", sup.sending_thread)
    print('RECEIVED MESSAGES: ', server.received_messages)
    # TODO: remove block

    # {{{TODO: just try calling task_runner.get_init_data_for_agent(agent_info.agent); for this static task, this will trigger `agent.state.set_init_state(assignment_data.shared)`}}}

    # server.send_agent_act(agent_id_1, {"text": "message1"})
    # message = server.get_agent_message()
    # TODO: remove

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
