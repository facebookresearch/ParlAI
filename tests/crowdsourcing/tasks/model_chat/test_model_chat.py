#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the model chat crowdsourcing task.
"""

import glob
import json
import os
import unittest

import parlai.utils.testing as testing_utils
from parlai.zoo.blender.blender_90M import download as download_blender


# Inputs
AGENT_DISPLAY_IDS = ('Worker',)
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
        },
    },
)


try:

    import parlai.crowdsourcing.tasks.model_chat.worlds as world_module
    from parlai.crowdsourcing.tasks.model_chat.run import TASK_DIRECTORY
    from parlai.crowdsourcing.tasks.model_chat.model_chat_blueprint import (
        SharedModelChatTaskState,
        ModelChatBlueprintArgs,
        BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.model_chat.utils import AbstractModelChatTest

    class TestModelChat(AbstractModelChatTest):
        """
        Test the model chat crowdsourcing task.
        """

        # TODO: remove the inheritance from unittest.TestCase once this test uses pytest
        #  regressions. Also use a pytest.fixture to call self._setup() and
        #  self._teardown(), like the other tests use, instead of calling them with
        #  self.setUp() and self.tearDown()

        def setUp(self) -> None:
            self._setup()

        def tearDown(self) -> None:
            self._teardown()

        def test_base_task(self):

            with testing_utils.tempdir() as tmpdir:

                # Paths
                expected_states_folder = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'expected_states'
                )
                expected_chat_data_path = os.path.join(
                    expected_states_folder, 'final_chat_data.json'
                )
                expected_state_path = os.path.join(expected_states_folder, 'state.json')
                parlai_data_folder = os.path.join(tmpdir, 'parlai_data')
                model_folder = os.path.join(parlai_data_folder, 'models')
                chat_data_folder = os.path.join(tmpdir, 'final_chat_data')

                # Download the Blender 90M model
                download_blender(parlai_data_folder)

                # Set up the config and database
                num_blender_convos = 10
                args = ModelChatBlueprintArgs()
                overrides = [
                    f'+mephisto.blueprint.{key}={val}'
                    for key, val in args.__dict__.items()
                    if key
                    in [
                        'max_onboard_time',
                        'max_resp_time',
                        'override_opt',
                        'random_seed',
                        'world_file',
                    ]
                ] + [
                    'mephisto.blueprint.annotations_config_path=${task_dir}/task_config/annotations_config.json',
                    f'mephisto.blueprint.base_model_folder={model_folder}',
                    f'mephisto.blueprint.conversations_needed_string=\"blender/blender_90M:{num_blender_convos:d}\"',
                    f'mephisto.blueprint.chat_data_folder={chat_data_folder}',
                    '+mephisto.blueprint.left_pane_text_path=${task_dir}/task_config/left_pane_text.html',
                    f'+mephisto.blueprint.num_conversations={num_blender_convos:d}',
                    '+mephisto.blueprint.onboard_task_data_path=${task_dir}/task_config/onboard_task_data.json',
                    '+mephisto.blueprint.task_description_file=${task_dir}/task_config/task_description.html',
                ]
                # TODO: remove all of these params once Hydra 1.1 is released with
                #  support for recursive defaults
                self._set_up_config(
                    blueprint_type=BLUEPRINT_TYPE,
                    task_directory=TASK_DIRECTORY,
                    overrides=overrides,
                )

                # Set up the operator and server
                shared_state = SharedModelChatTaskState(world_module=world_module)
                self._set_up_server(shared_state=shared_state)

                # Check that the agent states are as they should be
                self._get_channel_info().job.task_runner.task_run.get_blueprint().use_onboarding = (
                    False
                )
                # Don't require onboarding for this test agent
                with open(expected_state_path) as f:
                    expected_state = json.load(f)
                self._test_agent_states(
                    num_agents=1,
                    agent_display_ids=AGENT_DISPLAY_IDS,
                    agent_messages=AGENT_MESSAGES,
                    form_messages=FORM_MESSAGES,
                    form_task_data=FORM_TASK_DATA,
                    expected_states=(expected_state,),
                    agent_task_data=AGENT_TASK_DATA,
                )

                # Check that the contents of the chat data file are as expected
                with open(expected_chat_data_path) as f:
                    expected_chat_data = json.load(f)
                results_path = list(
                    glob.glob(os.path.join(chat_data_folder, '*_*_*_sandbox.json'))
                )[0]
                with open(results_path) as f:
                    actual_chat_data = json.load(f)
                self._check_final_chat_data(
                    actual_value=actual_chat_data, expected_value=expected_chat_data
                )


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
