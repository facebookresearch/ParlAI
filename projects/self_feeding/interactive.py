#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic example which allows local human keyboard input to talk to a trained model.

Note: this is identical to examples/interactive with the exception that we add
TransformerAgent command line args.
"""

import os
import random

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.tasks.self_feeding.build import build
from projects.self_feeding.self_feeding_agent import SelfFeedingAgent


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Interactive chat with a model')
    parser.set_defaults(interactive_mode=True, task='interactive')
    LocalHumanAgent.add_cmdline_args(parser)
    SelfFeedingAgent.add_cmdline_args(parser)
    parser.set_defaults(history_size=2)
    return parser


def interactive(opt):
    opt['task'] = 'self_feeding'
    build(opt)
    opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'
    cand_file = os.path.join(opt['datapath'], 'self_feeding/convai2_cands.txt')
    # Set values to override when the opt dict for the saved model is loaded
    opt['override'] = {
        'subtasks': ['dialog', 'satisfaction'],
        'interactive': True,
        'interactive_task': True,
        'prev_response_filter': True,
        'person_tokens': False,  # SelfFeedingAgent adds person_tokens on its own
        'partial_load': True,
        'history_size': 2,
        'eval_candidates': 'fixed',
        'encode_candidate_vecs': True,
        'fixed_candidates_path': cand_file,
        # Pull these from current opt dictionary
        'no_cuda': opt["no_cuda"],
        'fixed_candidate_vecs': opt['fixed_candidate_vecs'],
        'rating_frequency': opt['rating_frequency'],
        'rating_gap': opt['rating_gap'],
        'rating_threshold': opt['rating_threshold'],
        'request_feedback': opt['request_feedback'],
        'request_rating': opt['request_rating'],
    }

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)

    # Show some example dialogs:
    while True:
        world.parley()
        if world.epoch_done():
            print("EPOCH DONE")
            break


if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()
    interactive(parser.parse_args())
