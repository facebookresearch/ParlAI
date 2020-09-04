#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test components of specific crowdsourcing tasks.
"""

import json
import os
import threading
import unittest

from parlai.core.agents import create_agent_from_shared
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

        # Params
        model_name = 'multi_task__bst_tuned'
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

        # Set up semaphore
        max_concurrent_responses = 1
        semaphore = threading.Semaphore(max_concurrent_responses)

        # Set up bot agent
        # {{{TODO: first, trigger downloading multi_task__bst_tuned so that the model file will exist!}}}
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
            # {{{TODO: test all responses}}}

        # Check the output data
        model_nickname, worker_is_unacceptable, convo_finished = world.save_data()
        # {{{TODO: Make sure output values are correct}}}
        # {{{TODO: check the final saved output}}}


if __name__ == "__main__":
    unittest.main()
