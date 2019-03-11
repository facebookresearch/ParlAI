#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Basic example which iterates through the tasks specified and
checks them for offensive language.

Examples
--------

.. code-block:: shell

  python -m parlai.scripts.detect_offensive_language -t "convai_chitchat" --display-examples True
"""  # noqa: E501
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.utils import OffensiveLanguageDetector, TimeLogger

import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Check task for offensive language')
    parser.add_pytorch_datateacher_args()
    # Get command line arguments
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(datatype='train:ordered')
    parser.set_defaults(model='repeat_query')
    return parser


def detect(opt, printargs=None, print_parser=None):
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
    bad = OffensiveLanguageDetector()

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    # Show some example dialogs:
    cnt = 0
    while not world.epoch_done():
        world.parley()
        words = []
        for a in world.acts:
            offensive = bad.contains_offensive_language(a.get('text', ''))
            if offensive:
                words.append(offensive)
            labels = a.get('labels', a.get('eval_labels', ''))
            for l in labels:
                offensive = bad.contains_offensive_language(l)
                if offensive:
                    words.append(offensive)
        if len(words) > 0 and opt['display_examples']:
            print(world.display())
            print("[Offensive words detected:]", ', '.join(words))
            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        cnt += len(words)
        if log_time.time() > log_every_n_secs:
            report = world.report()
            log = {'offenses': cnt}
            text, log = log_time.log(report['exs'], world.num_examples(), log)
            print(text)

    if world.epoch_done():
        print("EPOCH DONE")
    print(str(cnt) + " offensive messages found out of " +
          str(world.num_examples()) + " messages.")
    return world.report()


if __name__ == '__main__':
    parser = setup_args()
    detect(parser.parse_args(print_args=False), print_parser=parser)
