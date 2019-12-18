#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic script which allows local human keyboard input to talk to a trained model.

Examples
--------

.. code-block:: shell

  python examples/interactive.py -m drqa -mf "models:drqa/squad/model"

When prompted, enter something like: ``Bob is Blue.\\nWhat is Bob?``

Input is often model or task specific, but in drqa, it is always
``context '\\n' question``.
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.local_human.local_human import LocalHumanAgent

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
    parser.add_argument(
        '-it',
        '--interactive-task',
        type='bool',
        default=True,
        help='Create interactive version of task',
    )
    parser.set_defaults(interactive_mode=True, task='interactive')
    LocalHumanAgent.add_cmdline_args(parser)
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

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    human_agent = LocalHumanAgent(opt)
    world = create_task(opt, [human_agent, agent])

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
