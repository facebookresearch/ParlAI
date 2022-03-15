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
                ('LabeledBlendedSkillTalk', 'train', 4819, 27018),
                ('LabeledBlendedSkillTalk', 'valid', 1009, 5651),
                ('LabeledBlendedSkillTalk', 'test', 980, 5482),
                ('LabeledConvAI2PersonaTopicifier', 'train', 17878, 131438),
                ('LabeledConvAI2PersonaTopicifier', 'valid', 1000, 7801),
                ('LabeledConvAI2PersonaTopicifier', 'test', 1000, 7801),
                ('LabeledEDPersonaTopicifier', 'train', 39057, 64636),
                ('LabeledEDPersonaTopicifier', 'valid', 2769, 5738),
                ('LabeledEDPersonaTopicifier', 'test', 2547, 5259),
                ('LabeledWoWPersonaTopicifier', 'train', 18430, 74092),
                ('LabeledWoWPersonaTopicifier', 'valid', 981, 3939),
                ('LabeledWoWPersonaTopicifier', 'test', 965, 3865),
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
                (
                    'LabeledBlendedSkillTalk',
                    'train',
                    {
                        'id': 'internal:blended_skill_talk',
                        'text': "your persona: i've 2 kids.\nyour persona: i love flowers.\nI love live music, that's why I try to go to concerts\nI do too. Wat do you like?\nI like acting, I hope to be an actor, what about you?",
                        'labels': ['that is ok.  have any kids?'],
                        'context_dataset': 'empathetic_dialogues',
                        'free_turker_message': 'I like acting, I hope to be an actor, what about you?',
                        'guided_turker_chosen_suggestion': ' ',
                        'personality': 'Maternal (Mother-like)',
                        'episode_done': False,
                    },
                ),
                (
                    'LabeledConvAI2PersonaTopicifier',
                    'valid',
                    {
                        'id': 'internal:blended_skill_talk:ConvAI2PersonaTopicifierTeacher',
                        'text': "your persona: i read twenty books a year.\nyour persona: i'm a stunt double as my second job.\nyour persona: i only eat kosher.\nyour persona: i was raised in a single parent household.\nAlabama\nhello what are doing today ?",
                        'labels': [
                            'i am good , i just got off work and tired , i have two jobs .'
                        ],
                        'personality': 'Lazy',
                        'episode_done': False,
                        'label_candidates': {
                            'num_cands': 20,
                            'first': 'oh really ? i am actually in high school and i am graduating as class of 2019 !',
                            'last': 'i am good , i just got off work and tired , i have two jobs .',
                        },
                    },
                ),
                (
                    'LabeledEDPersonaTopicifier',
                    'test',
                    {
                        'id': 'internal:blended_skill_talk:EDPersonaTopicifierTeacher',
                        'text': "your persona: my mom raised me by herself and taught me to play baseball.\nyour persona: i blog about salt water aquarium ownership.\nyour persona: i still love to line dry my clothes.\nyour persona: i am allergic to peanuts.\nyour persona: i'll one day own a ferret.\nMarine aquarium\nYeah about 10 years ago I had a horrifying experience. It was 100% their fault but they hit the water barrels and survived. They had no injuries but they almost ran me off the road.",
                        'labels': ['Did you suffer any injuries?'],
                        'situation': "I felt guilty when I was driving home one night and a person tried to fly into my lane, and didn't see me. I honked and they swerved back into their lane, slammed on their brakes, and hit the water cones.",
                        'emotion': 'guilty',
                        'prepend_ctx': 'None',
                        'prepend_cand': 'None',
                        'deepmoji_ctx': 'None',
                        'deepmoji_cand': 'None',
                        'personality': 'Curious',
                        'episode_done': False,
                        'label_candidates': {
                            'num_cands': 100,
                            'first': 'I hope it goes well! If it makes you feel any better, most of them are probably just as nervous and are looking for any excuse to relax and let their guard down, too. Good luck',
                            'last': "I know how you feel.  I moved away from my family and friends this summer.  Do you have family nearby at all? I often feel lonely when I'm watching a movie by myself but then I remind myself that I'm loved by a lot of people.",
                        },
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
