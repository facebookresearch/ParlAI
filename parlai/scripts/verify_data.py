#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Verify data doesn't have basic mistakes, like empty text fields
or empty label candidates.

For example:
`python parlai/scripts/verify_data.py -t convai2 -dt train:ordered`
"""
from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.utils import TimeLogger


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Lint for ParlAI tasks')
    # Get command line arguments
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(datatype='train:ordered')
    return parser


def report(world, counts, log_time):
    report = world.report()
    log = {'missing_text': counts['missing_text'],
           'missing_labels': counts['missing_labels'],
           'missing_label_candidates': counts['missing_label_candidates'],
           'empty_label_candidates': counts['empty_label_candidates'],
           }
    text, log = log_time.log(report['exs'], world.num_examples(), log)
    print(text)


def verify(opt, printargs=None, print_parser=None):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    counts = {}
    counts['missing_text'] = 0
    counts['missing_labels'] = 0
    counts['missing_label_candidates'] = 0
    counts['empty_label_candidates'] = 0

    # Show some example dialogs.
    while not world.epoch_done():
        world.parley()

        act = world.acts[0]
        if 'text' not in act:
            print("warning: missing text field")
            counts['missing_text'] += 1

        if 'labels' not in act and 'eval_labels' not in act:
            print("warning: missing labels/eval_labels field")
            counts['missing_labels'] += 1
        else:
            if 'label_candidates' not in act:
                counts['missing_label_candidates'] += 1
            else:
                for c in act['label_candidates']:
                    if c == '':
                        print("warning: empty string label_candidate")
                        counts['empty_label_candidates'] += 1

        if log_time.time() > log_every_n_secs:
            print(report(world, counts, log_time))

    try:
        # print dataset size if available
        print('[ loaded {} episodes with a total of {} examples ]'.format(
            world.num_episodes(), world.num_examples()
        ))
    except Exception:
        pass
    report(world, counts, log_time)


if __name__ == '__main__':
    parser = setup_args()
    verify(parser.parse_args(print_args=False), print_parser=parser)
