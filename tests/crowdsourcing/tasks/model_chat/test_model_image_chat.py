#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the model image chat crowdsourcing task.
"""

import glob
import json
import os
import pickle
import unittest

from PIL import Image

import parlai.utils.testing as testing_utils
from parlai.core.message import Message
from parlai.zoo.image_chat.transresnet_multimodal import (
    download as download_transresnet,
)


# Inputs
AGENT_DISPLAY_IDS = ('Worker',)
AGENT_MESSAGES = [
    ("Response 1",),
    ("Response 2",),
    ("Response 3",),
    ("Response 4",),
    ("Response 5",),
    ("Response 6",),
]
FORM_MESSAGES = ("",)
# No info is sent through the 'text' field when submitting the form
FORM_TASK_DATA = ({"final_rating": 0},)


try:

    import parlai.crowdsourcing.tasks.model_chat.worlds_image_chat as world_module
    from parlai.crowdsourcing.tasks.model_chat.run_image_chat import TASK_DIRECTORY
    from parlai.crowdsourcing.tasks.model_chat.model_chat_blueprint import (
        SharedModelImageChatTaskState,
        ModelImageChatBlueprintArgs,
        IMAGE_CHAT_BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.model_chat.utils import AbstractModelChatTest

    class TestModelImageChat(AbstractModelChatTest):
        """
        Test the model image chat crowdsourcing task.
        """

        # TODO: remove the inheritance from unittest.TestCase once this test uses pytest
        #  regressions. Also use a pytest.fixture to call self._setup() and
        #  self._teardown(), like the other tests use, instead of calling them with
        #  self.setUp() and self.tearDown()

        def setUp(self) -> None:
            self._setup()

        def tearDown(self) -> None:
            self._teardown()

        @testing_utils.retry(ntries=3)
        def test_base_task(self):

            with testing_utils.tempdir() as tmpdir:

                # Paths
                expected_states_folder = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'expected_states'
                )
                expected_chat_data_path = os.path.join(
                    expected_states_folder, 'final_chat_data__image_chat.json'
                )
                expected_state_path = os.path.join(
                    expected_states_folder, 'state__image_chat.json'
                )
                parlai_data_folder = os.path.join(tmpdir, 'parlai_data')
                chat_data_folder = os.path.join(tmpdir, 'final_chat_data')
                sample_image_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'test_image_stack',
                    'sample_image.jpg',
                )
                image_context_path = os.path.join(tmpdir, 'image_contexts')
                stack_folder = os.path.join(tmpdir, 'image_stack')

                # Save image context: instead of downloading images, just save a pickle
                # file with all of the image act
                image_context = [
                    {
                        'image_act': Message(
                            {
                                'text': 'Obsessive',
                                'image_id': '2923e28b6f588aff2d469ab2cccfac57',
                                'episode_done': False,
                                'label_candidates': [
                                    "I must learn that bird's name!",
                                    "My, aren't you a pretty bird?",
                                ],
                                'image': Image.open(sample_image_path),
                                'id': 'image_chat',
                                'eval_labels': ["I must learn that bird's name!"],
                            }
                        )
                    }
                ]
                with open(image_context_path, 'wb') as f:
                    pickle.dump(image_context, f)

                # Download the Transresnet Multimodal model
                download_transresnet(parlai_data_folder)

                # Set up the config and database
                num_convos = 1
                args = ModelImageChatBlueprintArgs()
                overrides = [
                    f'+mephisto.blueprint.{key}={val}'
                    for key, val in args.__dict__.items()
                    if key
                    in [
                        'evals_per_image_model_combo',
                        'max_resp_time',
                        'override_opt',
                        'random_seed',
                        'world_file',
                    ]
                ] + [
                    'mephisto.blueprint.annotations_config_path=""',
                    f'mephisto.blueprint.chat_data_folder={chat_data_folder}',
                    f'+mephisto.blueprint.image_context_path={image_context_path}',
                    '+mephisto.blueprint.left_pane_text_path=${task_dir}/task_config/left_pane_text.html',
                    '+mephisto.blueprint.max_concurrent_responses=1',
                    'mephisto.blueprint.model_opt_path=${task_dir}/task_config/image_model_opts.yaml',
                    f'+mephisto.blueprint.num_conversations={num_convos:d}',
                    f'+mephisto.blueprint.stack_folder={stack_folder}',
                    '+mephisto.blueprint.task_description_file=${task_dir}/task_config/task_description.html',
                    'mephisto.blueprint.task_model_parallel=False',
                ]
                # TODO: remove all of these params once Hydra 1.1 is released with
                #  support for recursive defaults
                self._set_up_config(
                    blueprint_type=IMAGE_CHAT_BLUEPRINT_TYPE,
                    task_directory=TASK_DIRECTORY,
                    overrides=overrides,
                )

                # Set up the operator and server
                shared_state = SharedModelImageChatTaskState(world_module=world_module)
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
                )

                # Check that the contents of the chat data file are as expected
                with open(expected_chat_data_path) as f:
                    expected_chat_data = json.load(f)
                results_path = list(
                    glob.glob(os.path.join(chat_data_folder, '*/*_*_*_sandbox.json'))
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
