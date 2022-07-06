#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import parlai.utils.testing as testing_utils
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.scripts.display_data import setup_args
from parlai.tasks.convai2.worlds import InteractiveWorld


class TestConvAI2InteractiveWorld(unittest.TestCase):
    @patch("parlai.tasks.convai2.worlds._load_personas")
    def test_share(self, mock_load_personas):
        test_personas = ['your persona:I live on a pirate\'s shoulder']
        with testing_utils.tempdir() as data_path:
            mock_load_personas.return_value = test_personas
            kwargs = {
                'task': 'convai2',
                'datapath': data_path,
                'interactive_task': True,
                'interactive_mode': True,
            }
            parser = setup_args()
            parser.set_defaults(**kwargs)
            opt = parser.parse_args([])
            agent = RepeatLabelAgent(opt)
            agent2 = agent.clone()
            world = InteractiveWorld(opt=opt, agents=[agent, agent2])
            # We should not reload personas on share
            mock_load_personas.return_value = None
            new_world = world.clone()

            self.assertEqual(new_world.personas_list, test_personas)


if __name__ == '__main__':
    unittest.main()
