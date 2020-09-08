#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test components of specific crowdsourcing tasks.
"""

import glob
import json
import os
import threading
import unittest
from typing import Dict, List

import parlai.utils.testing as testing_utils
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.turn_annotations import run
from parlai.mturk.tasks.turn_annotations.bot_agent import TurkLikeAgent
from parlai.mturk.tasks.turn_annotations.worlds import TurnAnnotationsChatWorld


class TestTurnAnnotations(unittest.TestCase):
    """
    Test the turn annotations task.
    """

    def test_chat_world(self):

        with testing_utils.tempdir() as tmpdir:
            save_folder = tmpdir

            # Params
            model_name = 'multi_task_bst_tuned'
            zoo_model_file = 'zoo:blended_skill_talk/multi_task_bst_tuned/model'
            num_turns = 6
            config_folder = os.path.join(
                os.path.dirname(os.path.realpath(run.__file__)), 'config'
            )
            argparser = ParlaiParser(False, False)
            argparser.add_parlai_data_path()
            datapath = os.path.join(argparser.parlai_home, 'data')
            with open(os.path.join(config_folder, 'left_pane_text.html')) as f:
                left_pane_text = f.readlines()
            with open(os.path.join(config_folder, 'annotations_config.json')) as f:
                annotations_config = json.load(f)
            opt = Opt(
                {
                    'annotations_config': annotations_config,
                    'annotations_intro': 'Does this comment from your partner have any of the following attributes? (Check all that apply)',
                    'base_model_folder': os.path.join(
                        datapath, 'models', 'blended_skill_talk', model_name
                    ),
                    'check_acceptability': False,
                    'conversation_start_mode': 'hi',
                    'final_rating_question': 'Please rate your partner on a scale of 1-5.',
                    'include_persona': False,
                    'is_sandbox': True,
                    'left_pane_text': left_pane_text,
                    'save_folder': save_folder,
                    'task': 'turn_annotations',
                    'task_model_parallel': False,
                }
            )

            # Desired output
            bucket_assignments = [
                {
                    'bucket_0': False,
                    'bucket_1': False,
                    'bucket_2': True,
                    'bucket_3': False,
                    'bucket_4': True,
                }
            ] * num_turns  # Arbitrary choose buckets
            desired_results = {
                "personas": None,
                "context_dataset": None,
                "person1_seed_utterance": None,
                "person2_seed_utterance": None,
                "additional_context": None,
                "dialog": [
                    {
                        "left_pane_text": left_pane_text,
                        "episode_done": False,
                        "id": "Person1",
                        "text": "Hi!",
                        "fake_start": True,
                        "agent_idx": 0,
                        "config": {
                            "min_num_turns": num_turns,
                            "annotations_config": annotations_config,
                        },
                    },
                    {
                        "agent_idx": 1,
                        "text": "Hey!",
                        "id": "Polyencoder",
                        "problem_data": {
                            "turn_idx": 1,
                            "bucket_0": true,
                            "bucket_1": false,
                            "bucket_2": false,
                            "bucket_3": false,
                            "bucket_4": false,
                        },
                    },
                    {
                        "agent_idx": 0,
                        "text": "Hi there! How has your day been?",
                        "id": "Person1",
                    },
                    {
                        "agent_idx": 1,
                        "text": "My day has gone well how is yours?",
                        "id": "Polyencoder",
                        "problem_data": {
                            "turn_idx": 3,
                            "bucket_0": false,
                            "bucket_1": true,
                            "bucket_2": false,
                            "bucket_3": false,
                            "bucket_4": false,
                        },
                    },
                    {
                        "agent_idx": 0,
                        "text": "Mine has been pretty good. Do you have any plans for the weekend?",
                        "id": "Person1",
                    },
                    {
                        "agent_idx": 1,
                        "text": "No solid plans but it's friday so may spoil myself and pick up a takeaway and have a few drinks!",
                        "id": "Polyencoder",
                        "problem_data": {
                            "turn_idx": 5,
                            "bucket_0": false,
                            "bucket_1": false,
                            "bucket_2": true,
                            "bucket_3": false,
                            "bucket_4": false,
                        },
                    },
                    {
                        "agent_idx": 0,
                        "text": "Yeah that sounds great! What kind of food?",
                        "id": "Person1",
                    },
                    {
                        "agent_idx": 1,
                        "text": "Probably seafood or dessert!",
                        "id": "Polyencoder",
                        "problem_data": {
                            "turn_idx": 7,
                            "bucket_0": false,
                            "bucket_1": false,
                            "bucket_2": false,
                            "bucket_3": true,
                            "bucket_4": false,
                        },
                    },
                    {
                        "agent_idx": 0,
                        "text": "I like both of those things! What's your favorite dessert?",
                        "id": "Person1",
                    },
                    {
                        "agent_idx": 1,
                        "text": "My favourite is chocolate sundae! Do you have a favourite food?",
                        "id": "Polyencoder",
                        "problem_data": {
                            "turn_idx": 9,
                            "bucket_0": false,
                            "bucket_1": false,
                            "bucket_2": false,
                            "bucket_3": false,
                            "bucket_4": true,
                        },
                    },
                    {
                        "agent_idx": 0,
                        "text": "Hmm, good question - maybe ice cream?",
                        "id": "Person1",
                    },
                    {
                        "agent_idx": 1,
                        "text": "Ice cream! Ice cream is good. What kind do you like? I like vanilla!",
                        "id": "Polyencoder",
                        "problem_data": {
                            "turn_idx": 11,
                            "bucket_0": true,
                            "bucket_1": false,
                            "bucket_2": false,
                            "bucket_3": false,
                            "bucket_4": false,
                        },
                    },
                    {
                        "agent_idx": 0,
                        "text": "Oh you can't go wrong with vanilla! I like mint choco chip",
                        "id": "Person1",
                    },
                    {
                        "agent_idx": 1,
                        "text": "That sounds delicious! My favorite is m m chocolate chip.",
                        "id": "Polyencoder",
                        "problem_data": {
                            "turn_idx": 13,
                            "bucket_0": false,
                            "bucket_1": true,
                            "bucket_2": false,
                            "bucket_3": false,
                            "bucket_4": false,
                            "final_rating": "3",
                        },
                    },
                ],
                "workers": ["A1MAWO5M8TN1SN", "multi_task__bst_tuned"],
                "bad_workers": [],
                "acceptability_violations": [null],
                "hit_ids": ["3TUOHPJXYJ04GKD29QAW3HAL70JWXW", "none"],
                "assignment_ids": ["3NGMS9VZTNLORWYG99ZCI6JU9GTFF0", "none"],
                "task_description": {
                    "annotations_config": [
                        {
                            "value": "bucket_0",
                            "name": "Bucket 0",
                            "description": "this response implies something...0",
                        },
                        {
                            "value": "bucket_1",
                            "name": "Bucket 1",
                            "description": "this response implies something...1",
                        },
                        {
                            "value": "bucket_2",
                            "name": "Bucket 2",
                            "description": "this response implies something...2",
                        },
                        {
                            "value": "bucket_3",
                            "name": "Bucket 3",
                            "description": "this response implies something...3",
                        },
                        {
                            "value": "bucket_4",
                            "name": "Bucket 4",
                            "description": "this response implies something...4",
                        },
                    ],
                    "had_onboarding": false,
                    "model_nickname": "multi_task__bst_tuned",
                    "model_file": "/checkpoint/parlai/zoo/q_function/multi_task__bst_tuned/model",
                },
            }

            # Set up semaphore
            max_concurrent_responses = 1
            semaphore = threading.Semaphore(max_concurrent_responses)

            # Set up human agent
            human_worker = HumanLikeAgent(
                human_utterances=human_utterances, bucket_assignments=bucket_assignments
            )

            # Set up bot agent
            download_model_opt = Opt({'model_file': zoo_model_file})
            _ = create_agent(download_model_opt, requireModelExists=True)
            # First, trigger downloading multi_task_bst_tuned so that the model file
            # will exist
            shared_bot_agents = TurkLikeAgent.get_bot_agents(
                opt=opt, active_models=[model_name], datapath=datapath
            )

            # Get a bot and add it to the "human" worker
            print(f'Choosing the "{model_name}" model for the bot.')
            agent = create_agent_from_shared(shared_bot_agents[model_name])
            bot_worker = TurkLikeAgent(
                opt,
                model_name=model_name,
                model_agent=agent,
                num_turns=num_turns,
                semaphore=semaphore,
            )
            workers_including_bot = [human_worker, bot_worker]

            # Define world
            conv_idx = 0
            world = TurnAnnotationsChatWorld(
                opt=opt,
                agents=workers_including_bot,
                num_turns=num_turns,
                max_resp_time=180,
                tag='conversation t_{}'.format(conv_idx),
                context_info=None,
            )

            # Check each turn of the world
            while not world.episode_done():
                print('About to parley')
                world.parley()

            # Check the output data
            model_nickname, worker_is_unacceptable, convo_finished = world.save_data()
            self.assertEqual(model_nickname, model_name)
            self.assertFalse(worker_is_unacceptable)
            self.assertTrue(convo_finished)

            # Check the final results file saved by the world
            results_path = list(glob.glob(os.path.join(tmpdir, '*_*_sandbox.json')))[0]
            with open(results_path) as f:
                actual_results = json.load(f)
            for k, v in desired_results.items():
                if k == 'task_description':
                    for k2, v2 in desired_results[k].items():
                        self.assertEqual(actual_results[k].get(k2), v2)
                else:
                    self.assertEqual(actual_results.get(k), v)


class HumanLikeAgent:
    """
    Emulates a crowdsource worker for the purposes of testing.
    """

    def __init__(
        self,
        human_utterances: List[str],
        bucket_assignments: List[Dict[str, bool]],
        final_rating: int,
    ):
        """
        Stores a list of human utterances to deliver for each self.act().
        """
        self.human_utterances = human_utterances
        self.bucket_assignments = bucket_assignments
        self.final_rating = final_rating
        self.message_idx = 0

    def act(self, timeout=None) -> Message:
        _ = timeout  # This test agent doesn't use the timeout
        message = Message(
            {
                'text': self.human_utterances[self.message_idx],
                'id': 'Person1',
                'message_id': 'DUMMY_MESSAGE_ID',
                'problem_data_for_prior_message': {
                    'turn_idx': self.message_idx * 2 + 1,
                    **self.bucket_assignments[self.message_idx],
                },
            }
        )
        # The human agent turn_idx is computed differently
        self.message_idx += 1
        if len(self.human_utterances) >= self.message_idx:
            message['episode_done'] = True
            message['final_rating'] = str(self.final_rating)
        else:
            message['episode_done'] = False
        return message

    def observe(self, observation):
        """
        No-op.
        """
        pass


if __name__ == "__main__":
    unittest.main()
