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
from parlai.mturk.core.agents import TIMEOUT_MESSAGE, MTURK_DISCONNECT_MESSAGE
from parlai.mturk.tasks.turn_annotations import run
from parlai.mturk.tasks.turn_annotations.bot_agent import TurkLikeAgent
from parlai.mturk.tasks.turn_annotations.constants import (
    ONBOARD_FAIL,
    ONBOARD_SUBMIT,
    ONBOARD_SUCCESS,
)
from parlai.mturk.tasks.turn_annotations.worlds import (
    TurnAnnotationsChatWorld,
    TurnAnnotationsOnboardWorld,
)


ANNOTATIONS_INTRO_TEXT = 'Does this comment from your partner have any of the following attributes? (Check all that apply)'
HUMAN_LIKE_AGENT_WORKER_ID = 'HumanLikeAgent'
HUMAN_LIKE_AGENT_HIT_ID = "hit_id"
HUMAN_LIKE_AGENT_ASSIGNMENT_ID = "assignment_id"


class TestTurnAnnotations(unittest.TestCase):
    """
    Test the turn annotations task.
    """

    def test_onboard_world(self):
        """
        Test functionality of the onboarding world.
        """

        # Params
        answers_passing_onboarding = {
            "1": ["bucket_0"],
            "3": ["bucket_1"],
            "5": ["bucket_2"],
            "7": ["bucket_3"],
            "9": ["bucket_4"],
        }
        answers_failing_onboarding = {
            "1": ["bucket_4"],
            "3": ["bucket_3"],
            "5": ["bucket_2"],
            "7": ["bucket_1"],
            "9": ["bucket_0"],
        }

        # Define test cases
        acts_statuses_and_saved_texts = [
            (
                Message({'text': MTURK_DISCONNECT_MESSAGE}),
                MTURK_DISCONNECT_MESSAGE,
                None,
            ),  # Disconnect
            (Message({'text': TIMEOUT_MESSAGE}), TIMEOUT_MESSAGE, None),  # Timeout
            (
                Message(
                    {
                        'text': ONBOARD_SUBMIT,
                        'id': 'onboarding',
                        'message_id': 'dummy_id',
                        'onboard_submission': answers_passing_onboarding,
                        'episode_done': False,
                    }
                ),
                ONBOARD_SUCCESS,
                {
                    "worker_id": HUMAN_LIKE_AGENT_WORKER_ID,
                    "worker_answers": answers_passing_onboarding,
                },
            ),  # Passing onboarding
            (
                Message(
                    {
                        'text': ONBOARD_SUBMIT,
                        'id': 'onboarding',
                        'message_id': 'dummy_id',
                        'onboard_submission': answers_failing_onboarding,
                        'episode_done': False,
                    }
                ),
                ONBOARD_FAIL,
                {
                    "worker_id": HUMAN_LIKE_AGENT_WORKER_ID,
                    "worker_answers": answers_failing_onboarding,
                },
            ),  # Failing onboarding
        ]

        # Loop over test cases
        for act, desired_status, desired_saved_text in acts_statuses_and_saved_texts:
            with testing_utils.tempdir() as tmpdir:
                onboard_worker_answer_folder = tmpdir

                # Params
                config_folder = os.path.join(
                    os.path.dirname(os.path.realpath(run.__file__)), 'task_config'
                )

                # Define opt
                with open(os.path.join(config_folder, 'annotations_config.json')) as f:
                    annotations_config = json.load(f)
                with open(os.path.join(config_folder, 'onboard_task_data.json')) as f:
                    onboard_task_data = json.load(f)
                opt = Opt(
                    {
                        'annotations_config': annotations_config,
                        'annotations_intro': ANNOTATIONS_INTRO_TEXT,
                        'is_sandbox': True,
                        'max_onboard_time': 300,
                        'onboard_task_data': onboard_task_data,
                        'onboard_worker_answer_folder': onboard_worker_answer_folder,
                    }
                )

                # Set up human agent
                human_worker = HumanLikeOnboardingAgent(
                    onboarding_act=act,
                    disconnect_act=Message({'text': MTURK_DISCONNECT_MESSAGE}),
                )

                world = TurnAnnotationsOnboardWorld(opt, human_worker)
                world.onboard_failures_max_allowed = 0
                # To make it easier to test giving the wrong responses
                actual_status = world.parley()
                self.assertEqual(actual_status, desired_status)
                if desired_saved_text is not None:
                    worker_answers_path = os.path.join(
                        onboard_worker_answer_folder, 'worker_answers.json'
                    )
                    with open(worker_answers_path) as f:
                        actual_saved_text = json.load(f)
                    for k, v in desired_saved_text.items():
                        self.assertEqual(actual_saved_text.get(k), v)

    def test_chat_world(self):
        """
        Test functionality of the chat world.
        """

        with testing_utils.tempdir() as tmpdir:
            save_folder = tmpdir

            # Params
            model_name = 'blender_90M'
            zoo_model_file = 'zoo:blender/blender_90M/model'
            model = 'TransformerGenerator'
            num_turn_pairs = 6
            config_folder = os.path.join(
                os.path.dirname(os.path.realpath(run.__file__)), 'task_config'
            )
            datapath = os.path.join(tmpdir, 'data')

            # Download zoo model file
            model_file = modelzoo_path(datapath, zoo_model_file)

            # Define opt
            base_model_folder = os.path.dirname(os.path.dirname(model_file))
            # Get the folder that encloses the innermost model folder
            with open(os.path.join(config_folder, 'left_pane_text.html')) as f:
                left_pane_text = f.readlines()
            with open(os.path.join(config_folder, 'annotations_config.json')) as f:
                annotations_config = json.load(f)
            opt = Opt(
                {
                    'annotations_config': annotations_config,
                    'annotations_intro': ANNOTATIONS_INTRO_TEXT,
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
            bot_utterances = [
                "Hello, how are you today? I just got back from a long day at work, so I'm nervous.",
                "I just don't know what to do. I've never been so nervous in my life.",
                "Yes, I'll probably go to the movies. What about you? What do you like to do?",
                "That's great! What kind of restaurant do you usually go to? I love italian food.",
                "I love thai as well. What's your favorite kind of thai food? I like thai food the best.",
                'Oh, I love peanuts! I love all kinds of peanuts. Do you eat a lot of peanuts?',
                "I eat peanuts a lot, but only a few times a week. It's good for you.",
            ]
            human_utterances = [
                "What are you nervous about?",
                "Do you have any plans for the weekend?",
                "Yeah that sounds great! I like to bike and try new restaurants.",
                "Oh, Italian food is great. I also love Thai and Indian.",
                "Hmmm - anything with peanuts? Or I like when they have spicy licorice-like herbs.",
                "Ha, a decent amount, probably. What about you?",
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
                "id": model,
                "problem_data": {
                    "turn_idx": num_turn_pairs * 2 + 1,
                    **bucket_assignments[num_turn_pairs],
                    "final_rating": str(final_rating),
                },
            }
            dialog = [fake_first_human_turn]
            for turn_pair_idx in range(num_turn_pairs):
                bot_turn = {
                    "agent_idx": 1,
                    "text": bot_utterances[turn_pair_idx],
                    "id": model,
                    "problem_data": {
                        "turn_idx": turn_pair_idx * 2 + 1,
                        **bucket_assignments[turn_pair_idx],
                    },
                }
                human_turn = {
                    "agent_idx": 0,
                    "text": human_utterances[turn_pair_idx],
                    "id": human_agent_id,
                }
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
            human_worker = HumanLikeChatAgent(
                agent_id=human_agent_id,
                human_utterances=human_utterances,
                bucket_assignments=bucket_assignments,
                final_rating=final_rating,
            )

            # Set up bot agent
            shared_bot_agents = TurkLikeAgent.get_bot_agents(
                opt=opt, active_models=[model_name]
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


class HumanLikeOnboardingAgent:
    """
    Emulates a crowdsource worker for the purposes of testing onboarding worlds.
    """

    def __init__(self, onboarding_act: Message, disconnect_act: Message):
        """
        Just store what to return from self.act().
        """
        self.mturk_manager = FakeMTurkManager()
        self.worker_id = HUMAN_LIKE_AGENT_WORKER_ID
        self.onboarding_act = onboarding_act
        self.disconnect_act = disconnect_act
        self.onboarding_act_returned = False

    def act(self, timeout=None) -> Message:
        """
        The first time this is called, the desired act message is returned; after that,
        the disconnected message is returned (this should only happen when testing what
        happens after the user has been softblocked).
        """
        _ = timeout  # This test agent doesn't use the timeout
        if not self.onboarding_act_returned:
            self.onboarding_act_returned = True
            return self.onboarding_act
        else:
            return self.disconnect_act

    def observe(self, observation):
        """
        No-op.
        """
        pass


class FakeMTurkManager:
    """
    Fake MTurkManager class that implements a no-op soft-blocking method.
    """

    def __init__(self):
        pass

    def soft_block_worker(self, worker_id, qual='block_qualification'):
        _ = worker_id  # Ignore
        _ = qual  # Ignore
        pass


class HumanLikeChatAgent:
    """
    Emulates a crowdsource worker for the purposes of testing chat worlds.
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

        # Constants
        self.id = agent_id
        self.worker_id = HUMAN_LIKE_AGENT_WORKER_ID
        self.hit_id = HUMAN_LIKE_AGENT_HIT_ID
        self.assignment_id = HUMAN_LIKE_AGENT_ASSIGNMENT_ID
        self.human_utterances = human_utterances
        self.bucket_assignments = bucket_assignments
        self.final_rating = final_rating
        self.hit_is_abandoned = False
        self.hit_is_returned = False
        self.disconnected = False
        self.hit_is_expired = False

        # Changing
        self.message_idx = 0

    def act(self, timeout=None) -> Message:
        _ = timeout  # This test agent doesn't use the timeout
        message = Message(
            {
                'id': self.id,
                'message_id': 'DUMMY_MESSAGE_ID',
                'problem_data_for_prior_message': {
                    'turn_idx': self.message_idx * 2 + 1,
                    **self.bucket_assignments[self.message_idx],
                },
            }
        )
        # The human agent turn_idx is computed differently
        if self.message_idx >= len(self.human_utterances):
            message.update(
                {
                    'text': "I am done with the chat and clicked the 'Done' button, thank you!",
                    'episode_done': True,
                }
            )
            # The text won't get used in this case
            message['problem_data_for_prior_message']['final_rating'] = str(
                self.final_rating
            )
        else:
            message.update(
                {'text': self.human_utterances[self.message_idx], 'episode_done': False}
            )
        self.message_idx += 1
        return message

    def observe(self, observation):
        """
        No-op.
        """
        pass


if __name__ == "__main__":
    unittest.main()
