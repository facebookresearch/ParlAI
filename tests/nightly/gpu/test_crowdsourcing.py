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
from parlai.core.agents import create_agent_from_shared
from parlai.core.build_data import modelzoo_path
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.turn_annotations import run
from parlai.mturk.tasks.turn_annotations.bot_agent import TurkLikeAgent
from parlai.mturk.tasks.turn_annotations.worlds import TurnAnnotationsChatWorld


HUMAN_LIKE_AGENT_WORKER_ID = 'HumanLikeAgent'
HUMAN_LIKE_AGENT_HIT_ID = "hit_id"
HUMAN_LIKE_AGENT_ASSIGNMENT_ID = "assignment_id"


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
            num_turn_pairs = 6
            config_folder = os.path.join(
                os.path.dirname(os.path.realpath(run.__file__)), 'config'
            )
            argparser = ParlaiParser(False, False)
            argparser.add_parlai_data_path()
            datapath = os.path.join(argparser.parlai_home, 'data')

            # Download zoo model file
            model_file = modelzoo_path(datapath, zoo_model_file)
            # First, trigger downloading multi_task_bst_tuned so that the model file
            # will exist
            base_model_folder = os.path.dirname(model_file)
            with open(os.path.join(config_folder, 'left_pane_text.html')) as f:
                left_pane_text = f.readlines()
            with open(os.path.join(config_folder, 'annotations_config.json')) as f:
                annotations_config = json.load(f)
            opt = Opt(
                {
                    'annotations_config': annotations_config,
                    'annotations_intro': 'Does this comment from your partner have any of the following attributes? (Check all that apply)',
                    'base_model_folder': base_model_folder,
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

            # Construct desired dialog
            human_agent_id = "Person1"
            human_utterances = [
                "Hi there! How has your day been?",
                "Mine has been pretty good. Do you have any plans for the weekend?",
                "Yeah that sounds great! What kind of food?",
                "I like both of those things! What's your favorite dessert?",
                "Hmm, good question - maybe ice cream?",
                "Oh you can't go wrong with vanilla! I like mint choco chip",
            ]
            bot_utterances = [
                "Hey!",
                "My day has gone well how is yours?",
                "No solid plans but it's friday so may spoil myself and pick up a takeaway and have a few drinks!",
                "Probably seafood or dessert!",
                "My favourite is chocolate sundae! Do you have a favourite food?",
                "Ice cream! Ice cream is good. What kind do you like? I like vanilla!",
                "That sounds delicious! My favorite is m m chocolate chip.",
            ]
            bucket_assignments = [
                {
                    'bucket_0': False,
                    'bucket_1': False,
                    'bucket_2': True,
                    'bucket_3': False,
                    'bucket_4': True,
                }
            ] * (num_turn_pairs + 1)
            # Arbitrary choose buckets. The +1 is for the final model response at the
            # end
            final_rating = 3
            fake_first_human_turn = {
                "left_pane_text": left_pane_text,
                "episode_done": False,
                "id": "Person1",
                "text": "Hi!",
                "fake_start": True,
                "agent_idx": 0,
                "config": {
                    "min_num_turns": num_turn_pairs,
                    "annotations_config": annotations_config,
                },
            }
            final_bot_turn = {
                "agent_idx": 1,
                "text": bot_utterances[num_turn_pairs],
                "id": "Polyencoder",
                "problem_data": {
                    "turn_idx": num_turn_pairs * 2 + 1,
                    **bucket_assignments[num_turn_pairs],
                    "final_rating": str(final_rating),
                },
            }
            dialog = [fake_first_human_turn]
            for turn_pair_idx in range(num_turn_pairs):
                bot_turn = (
                    {
                        "agent_idx": 1,
                        "text": bot_utterances[turn_pair_idx],
                        "id": "Polyencoder",
                        "problem_data": {
                            "turn_idx": turn_pair_idx * 2 + 1,
                            **bucket_assignments[turn_pair_idx],
                        },
                    },
                )
                human_turn = (
                    {
                        "agent_idx": 0,
                        "text": human_utterances[turn_pair_idx],
                        "id": human_agent_id,
                    },
                )
                dialog += [bot_turn, human_turn]
            dialog += [final_bot_turn]

            # Construct desired output
            desired_results = {
                "personas": None,
                "context_dataset": None,
                "person1_seed_utterance": None,
                "person2_seed_utterance": None,
                "additional_context": None,
                "dialog": dialog,
                "workers": [HUMAN_LIKE_AGENT_WORKER_ID, model_name],
                "bad_workers": [],
                "acceptability_violations": [None],
                "hit_ids": [HUMAN_LIKE_AGENT_HIT_ID, "none"],
                "assignment_ids": [HUMAN_LIKE_AGENT_ASSIGNMENT_ID, "none"],
                "task_description": {
                    "annotations_config": annotations_config,
                    "model_nickname": model_name,
                    "model_file": model_file,
                },
            }

            # Set up semaphore
            max_concurrent_responses = 1
            semaphore = threading.Semaphore(max_concurrent_responses)

            # Set up human agent
            human_worker = HumanLikeAgent(
                agent_id=human_agent_id,
                human_utterances=human_utterances,
                bucket_assignments=bucket_assignments,
                final_rating=final_rating,
            )

            # Set up bot agent
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
                num_turns=num_turn_pairs,
                semaphore=semaphore,
            )
            workers_including_bot = [human_worker, bot_worker]

            # Define world
            conv_idx = 0
            world = TurnAnnotationsChatWorld(
                opt=opt,
                agents=workers_including_bot,
                num_turns=num_turn_pairs,
                max_resp_time=180,
                tag='conversation t_{}'.format(conv_idx),
                context_info=None,
            )

            # Run conversation
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
        agent_id: str,
        human_utterances: List[str],
        bucket_assignments: List[Dict[str, bool]],
        final_rating: int,
    ):
        """
        Stores a list of human utterances to deliver for each self.act().
        """
        self.agent_id = agent_id
        self.worker_id = HUMAN_LIKE_AGENT_WORKER_ID
        self.human_utterances = human_utterances
        self.bucket_assignments = bucket_assignments
        self.final_rating = final_rating
        self.message_idx = 0

    def act(self, timeout=None) -> Message:
        _ = timeout  # This test agent doesn't use the timeout
        message = Message(
            {
                'text': self.human_utterances[self.message_idx],
                'id': self.agent_id,
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
