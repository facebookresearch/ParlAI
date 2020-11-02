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

    def set_up_test(
        self,
        blueprint_type: str,
        task_directory: str,
        overrides: Optional[List[str]] = None,
    ):
        """
        Set up the config, database, operator, and server.

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
        self.operator = Operator(self.db)
        self.operator.validate_and_run_config(self.config.mephisto, shared_state=None)
        channel_info = list(self.operator.supervisor.channels.values())[0]
        self.server = channel_info.job.architect.server
