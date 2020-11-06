#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities for running tests.
"""

import os
import tempfile
import time
from typing import List, Optional

from hydra.experimental import compose, initialize
from mephisto.core.local_database import LocalMephistoDB
from mephisto.core.operator import Operator
from mephisto.data_model.blueprint import SharedTaskState
from mephisto.utils.scripts import augment_config_from_db


class CrowdsourcingTestMixin:
    """
    Mixin for end-to-end tests of Mephisto-based crowdsourcing tasks.

    Allows for setup and teardown of the operator, as well as for config specification
    and agent registration.
    """

    def setUp(self):
        self.operator = None

    def tearDown(self):
        if self.operator is not None:
            self.operator.shutdown()

    def _set_up_config(
        self,
        blueprint_type: str,
        task_directory: str,
        overrides: Optional[List[str]] = None,
    ):
        """
        Set up the config and database.

        Uses the Hydra compose() API for unit testing and a temporary directory to store
        the test database.
        :param blueprint_type: string uniquely specifying Blueprint class
        :param task_directory: directory containing the `conf/` configuration folder.
          Will be injected as `${task_dir}` in YAML files.
        :param overrides: additional config overrides
        """

        # Define the configuration settings
        relative_task_directory = os.path.relpath(
            task_directory, os.path.dirname(__file__)
        )
        relative_config_path = os.path.join(relative_task_directory, 'conf')
        if overrides is None:
            overrides = []
        with initialize(config_path=relative_config_path):
            self.config = compose(
                config_name="example",
                overrides=[
                    f'+mephisto.blueprint._blueprint_type={blueprint_type}',
                    f'+mephisto/architect=mock',
                    f'+mephisto/provider=mock',
                    f'+task_dir={task_directory}',
                    f'+current_time={int(time.time())}',
                ]
                + overrides,
            )
            # TODO: when Hydra 1.1 is released with support for recursive defaults,
            #  don't manually specify all missing blueprint args anymore, but
            #  instead define the blueprint in the defaults list directly.
            #  Currently, the blueprint can't be set in the defaults list without
            #  overriding params in the YAML file, as documented at
            #  https://github.com/facebookresearch/hydra/issues/326 and as fixed in
            #  https://github.com/facebookresearch/hydra/pull/1044.

        self.data_dir = tempfile.mkdtemp()
        database_path = os.path.join(self.data_dir, "mephisto.db")
        self.db = LocalMephistoDB(database_path)
        self.config = augment_config_from_db(self.config, self.db)
        self.config.mephisto.architect.should_run_server = True

    def _set_up_server(self, shared_state: Optional[SharedTaskState] = None):
        """
        Set up the operator and server.
        """
        self.operator = Operator(self.db)
        self.operator.validate_and_run_config(
            self.config.mephisto, shared_state=shared_state
        )
        channel_info = list(self.operator.supervisor.channels.values())[0]
        self.server = channel_info.job.architect.server

    def _register_mock_agents(self, num_agents: int = 1) -> List[str]:
        """
        Register mock agents for testing, taking the place of crowdsourcing workers.

        Specify the number of agents to register. Return the agents' IDs after creation.
        """

        for idx in range(num_agents):

            # Register the worker
            mock_worker_name = f"MOCK_WORKER_{idx:d}"
            self.server.register_mock_worker(mock_worker_name)
            workers = self.db.find_workers(worker_name=mock_worker_name)
            worker_id = workers[0].db_id

            # Register the agent
            mock_agent_details = f"FAKE_ASSIGNMENT_{idx:d}"
            self.server.register_mock_agent(worker_id, mock_agent_details)

        # Get all agents' IDs
        agents = self.db.find_agents()
        agent_ids = [agent.db_id for agent in agents]

        return agent_ids
