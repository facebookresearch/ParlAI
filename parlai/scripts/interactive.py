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
from parlai.scripts.script import ParlaiScript
from parlai.utils.world_logging import WorldLogger
from parlai.agents.local_human.local_human import LocalHumanAgent
import parlai.utils.logging as logging

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
    parser.add_argument(
        '--save-world-logs',
        type='bool',
        default=False,
        help='Saves a jsonl file containing all of the task examples and '
        'model replies. Must also specify --report-filename.',
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
        logging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()
    human_agent = LocalHumanAgent(opt)
    # set up world logger
    world_logger = WorldLogger(opt) if opt['save_world_logs'] else None
    world = create_task(opt, [human_agent, agent])

    # Show some example dialogs:
    while not world.epoch_done():
        world.parley()
        if world_logger is not None:
            world_logger.log(world)
        if opt.get('display_examples'):
            print("---")
            print(world.display())
        if world_logger is not None:
            # dump world acts to file
            world_logger.reset()  # add final acts to logs
            base_outfile = opt['report_filename'].split('.')[0]
            outfile = f'{base_outfile}_{opt["task"]}_replies.jsonl'
            world_logger.write(outfile, world, file_format=opt['save_format'])


class Interactive(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return interactive(self.opt, print_parser=self.parser)


if __name__ == '__main__':
    random.seed(42)
    Interactive.main()
