#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the model chat crowdsourcing task.
"""

import os

import parlai.utils.testing as testing_utils
import pytest
from pytest_regressions.data_regression import DataRegressionFixture


# Inputs
AGENT_DISPLAY_IDS = ('Speaker 1',)
AGENT_MESSAGES = [
    ("What are you nervous about?",),
    ("Do you have any plans for the weekend?",),
    ("Yeah that sounds great! I like to bike and try new restaurants.",),
    ("Oh, Italian food is great. I also love Thai and Indian.",),
    (
        "Hmmm - anything with peanuts? Or I like when they have spicy licorice-like herbs.",
    ),
]
AGENT_TASK_DATA = [
    (
        {
            'problem_data_for_prior_message': {
                "bucket_0": False,
                "bucket_1": False,
                "bucket_2": True,
                "bucket_3": False,
                "bucket_4": True,
                "none_all_good": False,
            }
        },
    )
] * len(AGENT_MESSAGES)
FORM_MESSAGES = ("",)
# No info is sent through the 'text' field when submitting the form
FORM_TASK_DATA = (
    {
        "final_rating": 4,
        "problem_data_for_prior_message": {
            "bucket_0": False,
            "bucket_1": False,
            "bucket_2": True,
            "bucket_3": False,
            "bucket_4": True,
            "none_all_good": False,
        },
    },
)


try:

    import parlai.crowdsourcing.tasks.model_chat.worlds as world_module
    from parlai.crowdsourcing.tasks.model_chat.run import TASK_DIRECTORY
    from parlai.crowdsourcing.tasks.model_chat.model_chat_blueprint import (
        SharedModelChatTaskState,
    )
    from parlai.crowdsourcing.tasks.model_chat.utils import AbstractModelChatTest

    class TestModelChat(AbstractModelChatTest):
        """
        Test the model chat crowdsourcing task.
        """

        @pytest.fixture(scope="function")
        def setup_teardown(self):
            """
            Call code to set up and tear down tests.
            """
            self._setup()
            yield self.operator
            # All code after this will be run upon teardown
            self._teardown()

        @testing_utils.retry(ntries=3)
        def test_base_task(
            self, setup_teardown, data_regression: DataRegressionFixture
        ):

            self.operator = setup_teardown
            with testing_utils.tempdir() as tmpdir:

                # Paths
                model_opt_path = os.path.join(tmpdir, 'model_opts.yaml')
                chat_data_folder = os.path.join(tmpdir, 'final_chat_data')

                # Create a model opt file for the fixed-response model
                with open(model_opt_path, 'w') as f:
                    model_opt_contents = f"""\
fixed_response: >
    --model fixed_response
"""
                    f.write(model_opt_contents)

                # Set up the config and database
                num_convos = 10
                overrides = [
                    f'mephisto.blueprint.conversations_needed_string=\"fixed_response:{num_convos:d}\"',
                    f'mephisto.blueprint.chat_data_folder={chat_data_folder}',
                    f'mephisto.blueprint.model_opt_path={model_opt_path}',
                ]
                self._set_up_config(task_directory=TASK_DIRECTORY, overrides=overrides)

                # Set up the operator and server
                shared_state = SharedModelChatTaskState(world_module=world_module)
                self._set_up_server(shared_state=shared_state)

                # Check that the agent states are as they should be
                self._get_live_run().task_runner.task_run.get_blueprint().use_onboarding = (
                    False
                )
                self._test_agent_states(
                    num_agents=1,
                    agent_display_ids=AGENT_DISPLAY_IDS,
                    agent_messages=AGENT_MESSAGES,
                    form_messages=FORM_MESSAGES,
                    form_task_data=FORM_TASK_DATA,
                    agent_task_data=AGENT_TASK_DATA,
                    data_regression=data_regression,
                )

        def _remove_non_deterministic_keys(self, actual_state: dict) -> dict:
            actual_state = super()._remove_non_deterministic_keys(actual_state)

            # This chat task additionally includes a non-deterministic key in the first message
            custom_data = self._get_custom_data(actual_state)
            del custom_data['dialog'][0]['update_id']

            return actual_state

except ImportError:
    pass
