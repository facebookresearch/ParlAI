#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Basic example which allows local human keyboard input to talk to a trained model.

Note: this is identical to examples/interactive with the exception that we add
TransformerAgent command line args.
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.agents.transformer.transformer import TransformerAgent

import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Interactive chat with a model')
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument(
        '--display-prettify',
        type='bool',
        default=False,
        help='Set to use a prettytable when displaying '
        'examples with text candidates',
    )
    parser.add_argument(
        '--display-ignore-fields',
        type=str,
        default='label_candidates,text_candidates',
        help='Do not display these fields',
    )
    LocalHumanAgent.add_cmdline_args(parser)
    TransformerAgent.add_cmdline_args(parser)
    parser.set_defaults(history_size=2)
    return parser


def interactive(opt, print_parser=None):
    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: interactive should be passed opt not Parser ]')
        opt = opt.parse_args()
    opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'
    # Set the task to dialog, since that's the type we want its outputs to be
    print("Warning: hardcoding history_size=2")
    opt['override'] = {
        'no_cuda': True,
        'subtasks': ['dialog', 'sentiment'],
        'interactive': True,
        'prev_response_filter': True,
        'person_tokens': True,
        'history_size': 2,
        'eval_candidates': 'fixed',
        'fixed_candidates_path': 'data/convai2_cands.txt',
        'fixed_candidate_vecs': opt['fixed_candidate_vecs'],
        # Pull these from current opt dictionary
        'rating_frequency': opt['rating_frequency'],
        'rating_gap': opt['rating_gap'],
        'rating_threshold': opt['rating_threshold'],
        'request_explanation': opt['request_explanation'],
        'request_rating': opt['request_rating'],
    }

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()

    # Show some example dialogs:
    while True:
        world.parley()
        if opt.get('display_examples'):
            print("---")
            print(world.display())
        if world.epoch_done():
            print("EPOCH DONE")
            break


if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()
    interactive(parser.parse_args(print_args=False), print_parser=parser)
