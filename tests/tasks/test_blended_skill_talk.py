#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import parlai.utils.testing as testing_utils
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from parlai.scripts.display_data import setup_args
from parlai.tasks.blended_skill_talk.agents import ContextGenerator
from parlai.tasks.blended_skill_talk.worlds import InteractiveWorld, _load_personas


class TestBlendedSkillTalkTeacher(unittest.TestCase):
    """
    Basic tests to count the number of examples/episodes and to check a few utterances.

    Manual check of the number of episodes and examples:
    >>> import json
    >>> for datatype in ['train', 'valid', 'test']:
    >>>    with open(f'data/blended_skill_talk/{datatype}.json') as f:
    >>>        data = json.load(f)
    >>>    num_episodes = len(data)
    >>>    num_examples = sum([len(d['dialog']) // 2 for d in data])
    >>>    print(f'Number of episodes: {num_episodes:d}')
    >>>    print(f'Number of examples: {num_examples:d}')

    Output of manual check:
    Number of episodes: 4819
    Number of examples: 27018
    Number of episodes: 1009
    Number of examples: 5651
    Number of episodes: 980
    Number of examples: 5482
    """

    def test_counts(self):

        with testing_utils.tempdir() as tmpdir:
            data_path = tmpdir

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


class TestPersonaTopicifierTeachers(unittest.TestCase):
    """
    Test PersonaTopicifier teachers.

    Check the contents of the first example of teachers in which ConvAI2 personas and
    WoW topics are added to contexts.
    """

    def test_check_examples(self):

        # Define all pairs of task strings and examples
        tasks_and_messages = [
            (
                "blended_skill_talk:ConvAI2PersonaTopicifier",
                {
                    'text': "your persona: i like to remodel homes.\nyour persona: i like to go hunting.\nyour persona: i like to shoot a bow.\nyour persona: my favorite holiday is halloween.\nNicholas Sparks\nhi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape .",
                    'labels': [
                        'you must be very fast . hunting is one of my favorite hobbies .'
                    ],
                    'reward': 0,
                    'label_candidates': (
                        'my mom was single with 3 boys , so we never left the projects .',
                        'i try to wear all black every day . it makes me feel comfortable .',
                        'well nursing stresses you out so i wish luck with sister',
                        'yeah just want to pick up nba nfl getting old',
                        'i really like celine dion . what about you ?',
                        'no . i live near farms .',
                        "i wish i had a daughter , i'm a boy mom . they're beautiful boys though still lucky",
                        'yeah when i get bored i play gone with the wind my favorite movie .',
                        "hi how are you ? i'm eatingdinner with my hubby and 2 kids .",
                        'were you married to your high school sweetheart ? i was .',
                        'that is great to hear ! are you a competitive rider ?',
                        "hi , i'm doing ok . i'm abanker . how about you ?",
                        "i'm 5 years old",
                        'hi there . how are you today ?',
                        'i totally understand how stressful that can be .',
                        'yeah sometimes you do not know what you are actually watching',
                        'mother taught me to cook ! we are looking for an exterminator .',
                        'i enjoy romantic movie . what is your favorite season ? mine is summer .',
                        'editing photos takesa lot of work .',
                        'you must be very fast . hunting is one of my favorite hobbies .',
                    ),
                    'episode_done': False,
                },
            ),
            (
                "blended_skill_talk:EDPersonaTopicifier",
                {
                    'situation': 'I remember going to the fireworks with my best friend. There was a lot of people, but it only felt like us in the world.',
                    'emotion': 'sentimental',
                    'prepend_ctx': None,
                    'prepend_cand': None,
                    'deepmoji_ctx': None,
                    'deepmoji_cand': None,
                    'text': 'your persona: people hate that i obsess about the poor.\nyour persona: i like to make cellphone apps that would help heal our world.\nyour persona: i like to watch people pray together.\nyour persona: people don t like me too much but i like them anyways.\nAndroid (operating system)#Applications\nI remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people, we felt like the only people in the world.',
                    'labels': [
                        'Was this a friend you were in love with, or just a best friend?'
                    ],
                    'episode_done': False,
                },
            ),
            (
                "blended_skill_talk:WoWPersonaTopicifier",
                {
                    'id': 'WizardDialogKnowledgeTeacher',
                    'text': "your persona: not a day goes by that i don't drink four mountain dews.\nyour persona: i enjoy movies about aliens invading the earth.\nyour persona: my favorite hobby is chess.\nyour persona: i just dyed my hair hot pink with purple highlights.\nScience fiction\n",
                    'labels': [
                        "I think science fiction is an amazing genre for anything. Future science, technology, time travel, FTL travel, they're all such interesting concepts."
                    ],
                    'chosen_topic': 'Science fiction',
                    'episode_done': False,
                    'label_candidates': [],
                    'knowledge': 'Science fiction Science fiction (often shortened to SF or sci-fi) is a genre of speculative fiction, typically dealing with imaginative concepts such as futuristic science and technology, space travel, time travel, faster than light travel, parallel universes, and extraterrestrial life.\nScience fiction Science fiction often explores the potential consequences of scientific and other innovations, and has been called a "literature of ideas".\nScience fiction It usually avoids the supernatural, unlike the related genre of fantasy.\nScience fiction Historically, science-fiction stories have had a grounding in actual science, but now this is only expected of hard science fiction.\nScience fiction Science fiction is difficult to define, as it includes a wide range of subgenres and themes.\nScience fiction Hugo Gernsback, who suggested the term "scientifiction" for his "Amazing Stories" magazine, wrote: "By \'scientifiction\' I mean the Jules Verne, H. G. Wells and Edgar Allan Poe type of story—a charming romance intermingled with scientific fact and prophetic vision... Not only do these amazing tales make tremendously interesting reading—they are always instructive.\nScience fiction They supply knowledge... in a very palatable form... New adventures pictured for us in the scientifiction of today are not at all impossible of realization tomorrow...\n',
                    'title': 'Science fiction',
                    'checked_sentence': 'Science fiction (often shortened to SF or sci-fi) is a genre of speculative fiction, typically dealing with imaginative concepts such as futuristic science and technology, space travel, time travel, faster than light travel, parallel universes, and extraterrestrial life.',
                },
            ),
        ]
        for task_string, desired_message in tasks_and_messages:

            # Get message
            kwargs = {'task': task_string, 'datatype': 'train:ordered'}
            parser = setup_args()
            parser.set_defaults(**kwargs)
            opt = parser.parse_args([])
            agent = RepeatLabelAgent(opt)
            teacher = create_task(opt, agent).get_task_agent()
            actual_message = teacher.get(episode_idx=0, entry_idx=0)

            print(f'\nChecking {task_string}:')
            for key in desired_message.keys():
                if key in ['label_candidates']:
                    # These are often created randomly and thus will vary
                    continue
                print(key)
                self.assertEqual(desired_message[key], actual_message[key])
            print('')


class TestContextGenerator(unittest.TestCase):
    def test_generated_context(self):
        datatypes_seeds_and_desired_contexts = [
            (
                'train',
                0,
                {
                    'context_dataset': 'wizard_of_wikipedia',
                    'persona_1_strings': [
                        'i am a vegetarian.',
                        'i live on a pig farm.',
                    ],
                    'persona_2_strings': [
                        "my wife hates me , she thinks i'm lazy and poor.",
                        'i won a lottery 6 years ago but nobody knows.',
                    ],
                    'additional_context': 'Vegetarianism',
                    'person1_seed_utterance': 'What reasons would a person become a vegetarian?',
                    'person2_seed_utterance': 'religion is one, it is strongly linked with a number of religions that originated in ancient India ',
                },
            ),
            (
                'valid',
                1,
                {
                    'context_dataset': 'convai2',
                    'persona_1_strings': [
                        'my parents were also teachers.',
                        'for vacation i enjoy time at the beach.',
                    ],
                    'persona_2_strings': [
                        'i love to go to disney world every year.',
                        'i am in the third grade.',
                    ],
                    'additional_context': None,
                    'person1_seed_utterance': "you have your whole life in front of you i am sure you'll",
                    'person2_seed_utterance': 'maybe i will meet him at disney world ! 3',
                },
            ),
            (
                'test',
                2,
                {
                    'context_dataset': 'wizard_of_wikipedia',
                    'persona_1_strings': [
                        'i like to dance.',
                        'my aunt helped me escape when i was of.',
                    ],
                    'persona_2_strings': [
                        'my bedroom is purple and lime green.',
                        'i am a vegan.',
                    ],
                    'additional_context': 'Dance',
                    'person1_seed_utterance': 'I love to dance too!  What kind do you do?',
                    'person2_seed_utterance': 'I love choreography dance, Dance can be categorized and described by its choreography, by its repertoire of movements',
                },
            ),
        ]
        for datatype, seed, desired_context in datatypes_seeds_and_desired_contexts:
            parser = ParlaiParser(False, False)
            parser.add_parlai_data_path()
            context_opt = parser.parse_args([])
            context_generator = ContextGenerator(
                context_opt, datatype=datatype, seed=seed
            )
            actual_context = context_generator.get_context()
            self.assertEqual(desired_context, actual_context)


class TestBlendedSkillTalkInteractiveWorld(unittest.TestCase):
    @patch("parlai.tasks.blended_skill_talk.worlds._load_personas")
    def test_share(self, mock_load_personas):
        test_personas = ['your persona:I live on a pirate\'s shoulder']
        with testing_utils.tempdir() as data_path:
            mock_load_personas.return_value = test_personas
            kwargs = {
                'task': 'blended_skill_talk',
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

            self.assertEqual(new_world.contexts_data, test_personas)

    def test_safe_personas(self):

        base_kwargs = Opt({'datatype': 'train', 'task': 'blended_skill_talk'})
        safe_personas_only_to_count = {False: 4819, True: 3890}
        for safe_personas_only, count in safe_personas_only_to_count.items():
            full_kwargs = {**base_kwargs, 'safe_personas_only': safe_personas_only}
            parser = setup_args()
            parser.set_defaults(**full_kwargs)
            opt = parser.parse_args([])
            personas = _load_personas(opt)
            self.assertEqual(len(personas), count)


if __name__ == '__main__':
    unittest.main()
