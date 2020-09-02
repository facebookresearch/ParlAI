#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

import os
import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Flatten task')
    # Get command line arguments
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(datatype='train:ordered')
    parser.set_defaults(model='repeat_query')
    return parser


def detect(opt, printargs=None, print_parser=None, to_write=None):
    """Checks a task for offensive language.
    """
    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    random.seed(42)

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)

    to_write_folder = os.path.join(opt.get('datapath'), opt.get('task') + '_flattened')
    if not os.path.exists(to_write_folder):
        os.makedirs(to_write_folder)

    datatype = opt['datatype'].split(':')[0]
    to_write = os.path.join(to_write_folder, datatype + '.txt')

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()

    with open(to_write, 'w') as f:
        history = []
        episode_done = False
        while not world.epoch_done():
            world.parley()
            if episode_done:
                history = []
                episode_done = False
            act = world.acts[0]
            if act['episode_done']:
                episode_done = True
            text_lines = act['text'].split('\n')
            history += text_lines

            history_list = enumerate(history)
            text = '\n'.join([str(x[0] + 1) + ' ' + x[1] for x in history_list])
            label_type = 'labels' if 'labels' in act else 'eval_labels'
            label = act[label_type][0]
            history.append(label)
            cands = '|'.join(act.get('label_candidates'))
            new_line = '\t'.join([text, label, '', cands]) + '\n'
            f.write(new_line)

        if world.epoch_done():
            print("EPOCH DONE")


if __name__ == '__main__':
    parser = setup_args()
    detect(parser.parse_args(print_args=False), print_parser=parser)
