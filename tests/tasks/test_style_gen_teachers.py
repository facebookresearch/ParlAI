#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import parlai.utils.testing as testing_utils
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.scripts.display_data import setup_args


class TestStyleGenTeachers(unittest.TestCase):
    """
    Basic tests to count the number of examples/episodes and to check a few utterances.
    """

    def test_counts(self):

        with testing_utils.tempdir() as tmpdir:
            data_path = tmpdir
            cases = [
                ('LabeledBlendedSkillTalk', 'train', -1, -1),
                ('LabeledBlendedSkillTalk', 'valid', -1, -1),
                ('LabeledBlendedSkillTalk', 'test', -1, -1),
                ('LabeledConvAI2PT', 'train', -1, -1),
                ('LabeledConvAI2PT', 'valid', -1, -1),
                ('LabeledConvAI2PT', 'test', -1, -1),
                ('LabeledEmpatheticDialoguesPT', 'train', -1, -1),
                ('LabeledEmpatheticDialoguesPT', 'valid', -1, -1),
                ('LabeledEmpatheticDialoguesPT', 'test', -1, -1),
                ('LabeledWizardOfWikipediaPT', 'train', -1, -1),
                ('LabeledWizardOfWikipediaPT', 'valid', -1, -1),
                ('LabeledWizardOfWikipediaPT', 'test', -1, -1),
            ]
            for teacher_name, datatype, num_episodes, num_examples in cases:
                all_kwargs = {
                    'task': f'style_gen:{teacher_name}',
                    'datatype': datatype,
                    'datapath': data_path,
                }
                parser = setup_args()
                parser.set_defaults(**all_kwargs)
                opt = parser.parse_args([])
                agent = RepeatLabelAgent(opt)
                teacher = create_task(opt, agent).get_task_agent()
                self.assertEqual(teacher.num_episodes(), num_episodes)
                self.assertEqual(teacher.num_examples(), num_examples)

    def test_check_examples(self):

        with testing_utils.tempdir() as tmpdir:
            data_path = tmpdir

            # Check the first entry for 3 sample files
            teachers_datatypes_and_examples = [
                ('LabeledBlendedSkillTalk', 'train', {}),
                (
                    'LabeledConvAI2PT',
                    'valid',
                    {
                        'label_candidates': {
                            'num_cands': 20,
                            'first': 'FOO',
                            'last': 'FOO',
                        }
                    },
                ),
                (
                    'LabeledEmpatheticDialoguesPT',
                    'test',
                    {
                        'label_candidates': {
                            'num_cands': 100,
                            'first': 'FOO',
                            'last': 'FOO',
                        }
                    },
                ),
            ]
            for teacher_name, datatype, example in teachers_datatypes_and_examples:
                all_kwargs = {
                    'task': f'style_gen:{teacher_name}',
                    'datatype': datatype,
                    'datapath': data_path,
                }
                parser = setup_args()
                parser.set_defaults(**all_kwargs)
                opt = parser.parse_args([])
                agent = RepeatLabelAgent(opt)
                teacher = create_task(opt, agent).get_task_agent()
                actual_message = teacher.get(episode_idx=0, entry_idx=0)

                # Check for field equality
                self.assertEqual(set(actual_message.keys()), set(example.keys()))

                # Check label candidates
                if 'label_candidates' in example:
                    params = example['label_candidates']
                    self.assertEqual(
                        len(actual_message['label_candidates']), params['num_cands']
                    )
                    self.assertEqual(
                        actual_message['label_candidates'][0], params['first']
                    )
                    self.assertEqual(
                        actual_message['label_candidates'][-1], params['last']
                    )

                # Check other fields
                for key in [k for k in example.keys() if k != 'label_candidates']:
                    self.assertEqual(example[key], actual_message[key])


if __name__ == '__main__':
    unittest.main()
