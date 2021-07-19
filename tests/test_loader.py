#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
import unittest
import pytest

from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
import parlai.tasks.convai2.agents as c2agents
from parlai.tasks.convai2.worlds import (
    InteractiveWorld as c2interactive,
    SelfChatWorld as c2selfchat,
)
from parlai.core.loader import (
    load_agent_module,
    load_task_module,
    load_teacher_module,
    load_world_module,
)
from parlai.core.worlds import DialogPartnerWorld
import parlai.utils.testing as testing_utils

OPTION_CASES = [
    {'task': 'convai2:selfRevised', 'agent': 'repeat_label'},
    {'task': 'parlai.tasks.convai2.agents:selfRevisedTeacher', 'agent': 'repeat_label'},
]


class TestLoader(unittest.TestCase):
    """
    Make sure we can load various modules (agents, teachers, worlds).
    """

    def test_load_agent(self):
        for options in OPTION_CASES:
            agent_module = load_agent_module(options['agent'])
            self.assertEqual(agent_module, RepeatLabelAgent)

    def test_load_teacher(self):
        for options in OPTION_CASES:
            teacher_module = load_teacher_module(options['task'])
            self.assertEqual(teacher_module, c2agents.SelfRevisedTeacher)

    def test_load_task(self):
        for options in OPTION_CASES:
            task_module = load_task_module(options['task'])
            self.assertEqual(task_module, c2agents)

    def test_load_interactive_world(self):
        for options in OPTION_CASES:
            world_module = load_world_module(
                options['task'].split(':')[0], interactive_task=True
            )
            self.assertEqual(world_module, c2interactive)

    def test_load_self_chat_world(self):
        for options in OPTION_CASES:
            world_module = load_world_module(
                options['task'].split(':')[0], selfchat_task=True
            )
            self.assertEqual(world_module, c2selfchat)

    def test_load_dialog_partner_world(self):
        for options in OPTION_CASES:
            world_module = load_world_module(
                options['task'].split(':')[0], interactive_task=False, num_agents=2
            )
            self.assertEqual(world_module, DialogPartnerWorld)


class TestZoo(unittest.TestCase):
    """
    Test that zoo files import as expected.
    """

    def test_zoo_no_exists(self):
        with self.assertRaises(ImportError):
            testing_utils.display_model(
                {'model_file': 'zoo:unittests/fake', 'task': 'integration_tests'}
            )
        with self.assertRaises(ImportError):
            testing_utils.display_model(
                {'model_file': 'zoo:fakemodel', 'task': 'integration_tests'}
            )


@pytest.mark.nofbcode
class TestLoadParlAIInternal(unittest.TestCase):
    """
    Make sure we can load an agent from internal.
    """

    def setUp(self):
        # create a parlai_internal folder if it does not exist
        self.parlai_internal_exists = os.path.exists('parlai_internal')
        if not self.parlai_internal_exists:
            os.mkdir('parlai_internal')

        self.agent_folder_exists = os.path.exists('parlai_internal/agents')
        if not self.agent_folder_exists:
            os.mkdir('parlai_internal/agents')

        # copy over the example agent from example_parlai_internal
        shutil.copytree(
            'example_parlai_internal/agents/parrot', 'parlai_internal/agents/parrot'
        )

    def test_load_internal_agent(self):
        agent_module = load_agent_module('internal:parrot')
        assert agent_module, 'Could not load internal agent'

    def tearDown(self):
        # clean up: remove all folders that we created over the course
        # of this test
        if not self.parlai_internal_exists:
            shutil.rmtree('parlai_internal/')
        elif not self.agent_folder_exists:
            shutil.rmtree('parlai_internal/agents/')
        else:
            shutil.rmtree('parlai_internal/agents/parrot/')


if __name__ == '__main__':
    unittest.main()
