#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities for running tests.
"""

import os
import random
import tempfile
import time
import unittest
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from hydra.experimental import compose, initialize
from mephisto.abstractions.blueprint import SharedTaskState
from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.operations.operator import Operator
from mephisto.tools.scripts import augment_config_from_db
from pytest_regressions.data_regression import DataRegressionFixture


class AbstractCrowdsourcingTest:
    """
    Abstract class for end-to-end tests of Mephisto-based crowdsourcing tasks.

    Allows for setup and teardown of the operator, as well as for config specification
    and agent registration.
    """

    def _setup(self):
        """
        To be run before a test.

        Should be called in a pytest setup/teardown fixture.
        """

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        self.operator = None
        self.server = None

    def _teardown(self):
        """
        To be run after a test.

        Should be called in a pytest setup/teardown fixture.
        """

        if self.operator is not None:
            self.operator.force_shutdown()

        if self.server is not None:
            self.server.shutdown_mock()

    def _set_up_config(
        self,
        task_directory: str,
        overrides: Optional[List[str]] = None,
        config_name: str = "example",
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
        relative_config_path = os.path.join(
            relative_task_directory, 'hydra_configs', 'conf'
        )
        if overrides is None:
            overrides = []
        with initialize(config_path=relative_config_path):
            self.config = compose(
                config_name=config_name,
                overrides=[
                    f'mephisto/architect=mock',
                    f'mephisto/provider=mock',
                    f'+task_dir={task_directory}',
                    f'+current_time={int(time.time())}',
                ]
                + overrides,
            )

        self.data_dir = tempfile.mkdtemp()
        self.database_path = os.path.join(self.data_dir, "mephisto.db")
        self.db = LocalMephistoDB(self.database_path)
        self.config = augment_config_from_db(self.config, self.db)
        self.config.mephisto.architect.should_run_server = True

    def _set_up_server(self, shared_state: Optional[SharedTaskState] = None):
        """
        Set up the operator and server.
        """
        self.operator = Operator(self.db)
        self.operator.launch_task_run(self.config.mephisto, shared_state=shared_state)
        self.server = self._get_server()

    def _get_live_run(self):
        """
        Get the LiveTaskRun from this job.
        """
        live_runs = list(self.operator.get_running_task_runs().values())
        if len(live_runs) == 0:
            raise ValueError('No live runs present')
        return live_runs[0]

    def _get_server(self):
        """
        Return the MockArchitect's server associated with this run.
        """
        live_run = self._get_live_run()
        return live_run.architect.server

    def _get_channel_info(self):
        """
        Return channel info for the currently running job.
        """
        live_run = self._get_live_run()
        channels = list(live_run.client_io.channels.values())
        if len(channels) > 0:
            return channels[0]
        else:
            raise ValueError('No channel could be detected!')

    def _register_mock_agents(
        self, num_agents: int = 1, assume_onboarding: bool = False
    ) -> List[str]:
        """
        Register mock agents for testing and onboard them if needed, taking the place of
        crowdsourcing workers.

        Specify the number of agents to register. Return the agents' IDs after creation.
        """

        for idx in range(num_agents):

            mock_worker_registration_name = f"MOCK_WORKER_{idx:d}"
            mock_worker_name = f"{mock_worker_registration_name}_sandbox"

            # Register the agent
            mock_agent_details = f"FAKE_ASSIGNMENT_{idx:d}"
            self.server.register_mock_agent(
                mock_worker_registration_name, mock_agent_details
            )
            self.assert_sandbox_worker_created(mock_worker_name)
            workers = self.db.find_workers(worker_name=mock_worker_name)
            print(
                "Workers:",
                workers,
                self.db.find_workers(),
                [w.worker_name for w in self.db.find_workers()],
            )
            worker_id = workers[0].db_id

            if assume_onboarding:
                # Submit onboarding from the agent
                onboard_agents = self.db.find_onboarding_agents()
                onboard_data = {"onboarding_data": {"success": True}}
                self.server.register_mock_agent_after_onboarding(
                    worker_id, onboard_agents[0].get_agent_id(), onboard_data
                )
                self.await_channel_requests()
            self.await_channel_requests()
            print("Agents:", self.db.find_agents())

            # Make sure the agent can be found
            try:
                _ = self.db.find_agents()[idx]
            except IndexError:
                raise ValueError('The agent could not be registered!')

        # Get all agents' IDs
        agents = self.db.find_agents()
        if len(agents) != num_agents:
            raise ValueError(
                f'The actual number of agents is {len(agents):d} instead of the '
                f'desired {num_agents:d}!'
            )
        agent_ids = [agent.db_id for agent in agents]

        return agent_ids

    def await_channel_requests(self, timeout=2) -> None:
        time.sleep(0.1)
        tracked_run = self._get_live_run()
        assert self.operator._run_loop_until(
            lambda: len(tracked_run.client_io.request_id_to_channel_id) == 0, timeout
        ), f"Channeled requests not processed in time!"

    def assert_sandbox_worker_created(self, worker_name, timeout=2) -> None:
        assert self.operator._run_loop_until(
            lambda: len(self.db.find_workers(worker_name=worker_name)) > 0, timeout
        ), f"Worker {worker_name} not created in time!"


class AbstractOneTurnCrowdsourcingTest(AbstractCrowdsourcingTest):
    """
    Abstract class for end-to-end tests of one-turn crowdsourcing tasks.

    Useful for Blueprints such as AcuteEvalBlueprint and StaticReactBlueprint for which
    all of the worker's responses are sent to the backend code at once.
    """

    def _test_agent_state(
        self, task_data: Dict[str, Any], data_regression: DataRegressionFixture
    ):
        """
        Test that the actual agent state matches the expected state.

        Get the final agent state given the input task data and check that it is as
        expected.
        """
        state = self._get_agent_state(task_data=task_data)
        self._check_agent_state(state=state, data_regression=data_regression)

    def _get_agent_state(self, task_data: Dict[str, Any]):
        """
        Submit user task data and return the final agent state.

        Register a mock human agent, request initial data to define the 'inputs' field
        of the agent state, make the agent act to define the 'outputs' field of the
        agent state, and return the agent state.
        """

        # Set up the mock human agent
        if self.config.mephisto.blueprint.get("onboarding_qualification", None):
            agent_id = self._register_mock_agents(num_agents=1, assume_onboarding=True)[
                0
            ]
        else:
            agent_id = self._register_mock_agents(num_agents=1)[0]

        # Set initial data
        self.await_channel_requests()

        # Make agent act
        self.server.submit_mock_unit(agent_id, task_data)
        self.await_channel_requests()

        return self.db.find_agents()[0].state.get_data()

    def _check_agent_state(
        self, state: Dict[str, Any], data_regression: DataRegressionFixture
    ):
        """
        Given an agent state, test that it is as expected.
        """
        data_regression.check(state)


class AbstractParlAIChatTest(AbstractCrowdsourcingTest):
    """
    Abstract class for end-to-end tests of one-turn ParlAIChatBlueprint tasks.
    """

    def _setup(self):
        super()._setup()
        self.message_sleep_time = 0  # Time to wait for any late messages to arrive

    def _test_agent_states(
        self,
        num_agents: int,
        agent_display_ids: Sequence[str],
        agent_messages: List[Sequence[str]],
        form_messages: Sequence[str],
        form_task_data: Sequence[Dict[str, Any]],
        expected_states: Sequence[Dict[str, Any]],
        agent_task_data: Optional[List[Sequence[Dict[str, Any]]]] = None,
    ):
        """
        Test that the actual agent states match the expected states.

        Register mock human agents, request initial data to define the 'inputs' fields
        of the agent states, make the agents have a conversation to define the 'outputs'
        fields of the agent states, and then check that the agent states all match the
        desired agent states.
        """

        # If no task data was supplied, create empty task data
        if agent_task_data is None:
            agent_task_data = []
            for message_round in agent_messages:
                agent_task_data.append([{}] * len(message_round))

        # Set up the mock human agents
        agent_ids = self._register_mock_agents(num_agents=num_agents)

        # # Feed messages to the agents

        # Have agents talk to each other
        assert len(agent_messages) == len(agent_task_data)
        for message_round, task_data_round in zip(agent_messages, agent_task_data):
            assert len(message_round) == len(task_data_round) == len(agent_ids)
            for agent_id, agent_display_id, message, task_data in zip(
                agent_ids, agent_display_ids, message_round, task_data_round
            ):
                self._send_agent_message(
                    agent_id=agent_id,
                    agent_display_id=agent_display_id,
                    text=message,
                    task_data=task_data,
                )

        # Have agents fill out the form
        for agent_idx, agent_id in enumerate(agent_ids):
            self.server.send_agent_act(
                agent_id=agent_id,
                act_content={
                    'text': form_messages[agent_idx],
                    'task_data': form_task_data[agent_idx],
                    'id': agent_display_ids[agent_idx],
                    'episode_done': False,
                },
            )
            self.await_channel_requests()

        # Submit the HIT
        for agent_id in agent_ids:
            self.server.submit_mock_unit(agent_id, {'final_data': {}})
            self.await_channel_requests()

        # # Check that the inputs and outputs are as expected

        # Get and filter actual messages
        time.sleep(self.message_sleep_time)
        actual_states = [agent.state.get_data() for agent in self.db.find_agents()]
        if len(actual_states) != len(expected_states):
            raise ValueError(
                f'There are {len(actual_states):d} agent states, instead of {len(expected_states):d} as expected!'
            )
        filtered_actual_states = []
        for actual_state in actual_states:
            filtered_actual_states.append(self._filter_agent_state_data(actual_state))

        # Check the contents of each message
        for actual_state, expected_state in zip(
            filtered_actual_states, expected_states
        ):
            clean_actual_state = self._remove_non_deterministic_keys(actual_state)
            assert clean_actual_state['inputs'] == expected_state['inputs']
            actual_num_messages = len(clean_actual_state['outputs']['messages'])
            expected_num_messages = len(expected_state['outputs']['messages'])
            if actual_num_messages != expected_num_messages:
                raise ValueError(
                    f'The actual number of messages is {actual_num_messages:d}, instead of {expected_num_messages:d} as expected!'
                )
            for actual_message, expected_message in zip(
                clean_actual_state['outputs']['messages'],
                expected_state['outputs']['messages'],
            ):
                for key, expected_value in expected_message.items():
                    self._check_output_key(
                        key=key,
                        actual_value=actual_message[key],
                        expected_value=expected_value,
                    )

    def _remove_non_deterministic_keys(self, actual_state: dict) -> dict:
        """
        Allow for subclasses to delete certain keys in the actual state that will change
        on each run.
        """
        return actual_state

    def _filter_agent_state_data(self, agent_state: dict) -> dict:
        """
        Remove agent state messages that do not contain text and are thus not useful for
        testing the crowdsourcing task.
        """
        filtered_messages = [
            m for m in agent_state['outputs']['messages'] if 'text' in m
        ]
        filtered_agent_state = {
            'inputs': agent_state['inputs'],
            'outputs': {**agent_state['outputs'], 'messages': filtered_messages},
        }
        return filtered_agent_state

    def _check_output_key(
        self: Union['AbstractParlAIChatTest', unittest.TestCase],
        key: str,
        actual_value: Any,
        expected_value: Any,
    ):
        # TODO: remove typing of self after switching to pytest regressions, in which we
        #  no longer inherit from TestCase
        """
        Check the actual and expected values, given that they come from the specified
        key of the output message dictionary.

        This function can be extended to handle special cases for subclassed Mephisto
        tasks.
        """
        if actual_value != expected_value:
            raise ValueError(
                f'The value of ["{key}"] is supposed to be {expected_value} but is '
                f'actually {actual_value}!'
            )

    def _send_agent_message(
        self, agent_id: str, agent_display_id: str, text: str, task_data: Dict[str, Any]
    ):
        """
        Have the agent specified by agent_id send the specified text and task data with
        the given display ID string.
        """
        act_content = {
            "text": text,
            "task_data": task_data,
            "id": agent_display_id,
            "episode_done": False,
        }
        self.server.send_agent_act(agent_id=agent_id, act_content=act_content)
        self.await_channel_requests()


def collapse_whitespace(in_string: str):
    """
    Helper function to remove extra whitespace that may make table outputs direct checks
    fail.
    """
    while "  " in in_string:
        in_string = in_string.replace("  ", " ")
    return in_string


def check_stdout(actual_stdout: str, expected_stdout_path: str):
    """
    Check that actual and expected stdouts match.

    Given a string of the actual stdout and a path to the expected stdout, check that
    both stdouts match, keeping in mind that the actual stdout may have additional
    strings relating to progress updates that are not found in the expected output
    strings.

    TODO: this can probably be moved to a method of an abstract test class once all
     analysis code relies on pytest regressions for some of its tests.
    """
    actual_stdout_lines = actual_stdout.split('\n')
    with open(expected_stdout_path) as f:
        expected_stdout = f.read()
    for expected_line in expected_stdout.split('\n'):
        if not any(
            collapse_whitespace(expected_line) in collapse_whitespace(actual_line)
            for actual_line in actual_stdout_lines
        ):
            raise ValueError(
                f'\n\tThe following line:\n\n{expected_line}\n\n\twas not found '
                f'in the actual stdout:\n\n{actual_stdout}'
            )
