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

            # TODO: revise below
            opts_episodes_and_examples = [
                ({'datatype': 'train'}, 4819, 27018),
                ({'datatype': 'valid'}, 1009, 5651),
                ({'datatype': 'test'}, 980, 5482),
            ]
            for kwargs, num_episodes, num_examples in opts_episodes_and_examples:
                all_kwargs = {
                    **kwargs,
                    'task': 'blended_skill_talk',
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

            # Check the first entry (entry_idx==0) of the second episode for the train
            # set, in order to check the context for an episode that has a WoW topic
            # string
            train_opt_and_example = (
                {'datatype': 'train'},
                {
                    'text': "your persona: i just bought a new house with my partner.\nyour persona: i like to make my own coffee.\nLasagne\nOh, I love lasagne. I make my own noodles as well as the sauce. \nWow.  That's amazing.  I read where lasagne originated in Italy during the Middle Ages.  \nOh really!? That is interesting. I am actually italian myself.",
                    'labels': [
                        "Awesome. Me and my partner just bought a house. I can't wait to cook in my kitchen."
                    ],
                    'context_dataset': 'wizard_of_wikipedia',
                    'free_message': 'Oh really!? That is interesting. I am actually italian myself.',
                    'convai2': 'yum . i like to make lasagna and it s so good',
                    'empathetic_dialogues': 'Cool. I love italian. Real italian.',
                    'wizard_of_wikipedia': "Wow.  That's amazing.  I read where lasagne originated in Italy during the Middle Ages.",
                    'guided_chosen_suggestion': ' ',
                    'episode_done': False,
                },
            )
            all_kwargs = {
                **train_opt_and_example[0],
                'task': 'blended_skill_talk',
                'datapath': data_path,
            }
            parser = setup_args()
            parser.set_defaults(**all_kwargs)
            opt = parser.parse_args([])
            agent = RepeatLabelAgent(opt)
            teacher = create_task(opt, agent).get_task_agent()
            self.assertEqual(
                teacher.get(episode_idx=1, entry_idx=0), train_opt_and_example[1]
            )

            # Check the second entry (entry_idx==1) of the second episode for each dataset
            opts_and_examples = [
                (
                    {'datatype': 'train'},
                    {
                        'text': 'Moving in a new place can be a lot of fun. Are you a good cook?',
                        'labels': [
                            'I like to think so. I love to make coffee for an after dinner treat too.'
                        ],
                        'context_dataset': 'wizard_of_wikipedia',
                        'free_message': 'Moving in a new place can be a lot of fun. Are you a good cook?',
                        'convai2': 'yes ! trying to master lasagna .',
                        'empathetic_dialogues': "See. I'm not a great cook.",
                        'wizard_of_wikipedia': 'With the training and skills I have, I can cook pretty much anything.',
                        'guided_chosen_suggestion': ' ',
                        'episode_done': False,
                    },
                ),
                (
                    {'datatype': 'valid'},
                    {
                        'text': 'I like to go mountain biking with my friends.',
                        'labels': [
                            "I have never done that.  Not really the physical activity type, but I'd be willing to give it a try, I guess"
                        ],
                        'context_dataset': 'empathetic_dialogues',
                        'free_message': 'I like to go mountain biking with my friends.',
                        'convai2': "that's so cool , i love biking",
                        'empathetic_dialogues': "Ive never been on any but I'll try it out",
                        'wizard_of_wikipedia': "That's interesting!  Most mountain biking is in the categories of Trail and Cross Country riding styles",
                        'guided_chosen_suggestion': '',
                        'label_candidates': {
                            'num_cands': 100,
                            'first': 'i work as a vet so no days off over here!',
                            'last': 'And what else? ',
                        },
                        'episode_done': False,
                    },
                ),
                (
                    {'datatype': 'test'},
                    {
                        'text': "He eats insects, leaves and sun flower seeds. It's easy. They don't need walking and cleanup is simple. Do you have any pets?",
                        'labels': [
                            'No, not at the moment.  I have 3 girls and they are enough trouble! LOL'
                        ],
                        'context_dataset': 'empathetic_dialogues',
                        'free_message': "He eats insects, leaves and sun flower seeds. It's easy. They don't need walking and cleanup is simple. Do you have any pets?",
                        'convai2': "no , i don't have any pets either .",
                        'empathetic_dialogues': 'I do not just a cat',
                        'wizard_of_wikipedia': "I actually do.  He is ten years old and loves to be outside.  He's fat and furry.",
                        'guided_chosen_suggestion': '',
                        'label_candidates': {
                            'num_cands': 100,
                            'first': "Wow, engineering, sounds impressive.  I'm sure the income will be awesome.",
                            'last': 'but the worst part is you have to clean every day and keep the flat tidy all the time.  ',
                        },
                        'episode_done': False,
                    },
                ),
            ]
            for kwargs, example in opts_and_examples:
                all_kwargs = {
                    **kwargs,
                    'task': 'blended_skill_talk',
                    'datapath': data_path,
                }
                parser = setup_args()
                parser.set_defaults(**all_kwargs)
                opt = parser.parse_args([])
                agent = RepeatLabelAgent(opt)
                teacher = create_task(opt, agent).get_task_agent()
                actual_message = teacher.get(episode_idx=1, entry_idx=1)

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
