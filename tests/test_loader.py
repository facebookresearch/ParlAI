#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest

from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
import parlai.tasks.convai2.agents as c2agents
from parlai.tasks.convai2.worlds import InteractiveWorld as c2interactive
from parlai.core.loader import (
    load_agent_module,
    load_task_module,
    load_teacher_module,
    load_world_module,
)
from parlai.core.worlds import DialogPartnerWorld


OPTIONS = {
    'task': 'convai2:selfRevised',
    'agent': 'repeat_label',
}


class TestLoader(unittest.TestCase):
    """
    Make sure the package is alive.
    """

    def test_load_agent(self):
        agent_module = load_agent_module(OPTIONS['agent'])
        self.assertEqual(agent_module, RepeatLabelAgent)

    def test_load_teacher(self):
        teacher_module = load_teacher_module(OPTIONS['task'])
        self.assertEqual(teacher_module, c2agents.SelfRevisedTeacher)

    def test_load_task(self):
        task_module = load_task_module(OPTIONS['task'])
        self.assertEqual(task_module, c2agents)

    def test_load_interactive_world(self):
        world_module = load_world_module(
            OPTIONS['task'].split(':')[0], interactive_task=True,
        )
        print(world_module)
        self.assertEqual(world_module, c2interactive)

    def test_load_dialog_partner_world(self):
        world_module = load_world_module(
            OPTIONS['task'].split(':')[0], interactive_task=False, num_agents=2
        )
        self.assertEqual(world_module, DialogPartnerWorld)


if __name__ == '__main__':
    unittest.main()
