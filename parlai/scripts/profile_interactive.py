#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic script which allows to profile interaction with a model using repeat_query to
avoid human interaction (so we can time it, only).
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.repeat_query.repeat_query import RepeatQueryAgent
import parlai.utils.logging as logging

import random
import cProfile
import io
import pstats


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Interactive chat with a model')
    parser.add_argument('-d', '--display-examples', type='bool', default=True)
    parser.add_argument('-ne', '--num-examples', type=int, default=5)
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
    parser.add_argument(
        '-it',
        '--interactive-task',
        type='bool',
        default=True,
        help='Create interactive version of task',
    )
    parser.set_defaults(interactive_mode=True, task='interactive')
    return parser


def profile_interactive(opt, print_parser=None):
    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    if isinstance(opt, ParlaiParser):
        logging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    human_agent = RepeatQueryAgent(opt)
    world = create_task(opt, [human_agent, agent])

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()

    pr = cProfile.Profile()
    pr.enable()

    # Run
    cnt = 0
    while True:
        world.parley()
        if opt.get('display_examples'):
            print("---")
            print(world.display())
        cnt += 1
        if cnt >= opt.get('num_examples', 100):
            break
        if world.epoch_done():
            logging.info("epoch done")
            break

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()
    profile_interactive(parser.parse_args(print_args=False), print_parser=parser)
