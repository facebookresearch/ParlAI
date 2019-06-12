#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Basic example which iterates through the tasks specified and
evaluates the given model on them.

Examples
--------

.. code-block:: shell

  python eval_model.py -t "babi:Task1k:2" -m "repeat_label"
  python eval_model.py -t "#CornellMovie" -m "ir_baseline" -mp "-lp 0.5"
"""

from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.agents import create_agent
from parlai.core.logs import TensorboardLogger
from parlai.core.worlds import create_task
from parlai.core.utils import TimeLogger

import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Evaluate a model')
    parser.add_pytorch_datateacher_args()
    # Get command line arguments
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('--metrics', type=str, default="all",
                        help="list of metrics to show/compute, e.g. "
                             "ppl,f1,accuracy,hits@1."
                             "If 'all' is specified [default] all are shown.")
    TensorboardLogger.add_cmdline_args(parser)
    parser.set_defaults(datatype='valid')
    return parser


def eval_model(opt, printargs=None, print_parser=None):
    """Evaluates a model.

    :param opt: tells the evaluation function how to run
    :param bool print_parser: if provided, prints the options that are set within the
        model after loading the model
    :return: the final result of calling report()
    """
    if printargs is not None:
        print('[ Deprecated Warning: eval_model no longer uses `printargs` ]')
        print_parser = printargs
    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: eval_model should be passed opt not Parser ]')
        opt = opt.parse_args()

    random.seed(42)

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)

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
        cnt += opt.get('batchsize', 1)
        world.parley()
        if opt['display_examples']:
            print(world.display() + "\n~~")
        if log_time.time() > log_every_n_secs:
            report = world.report()
            text, report = log_time.log(report['exs'], world.num_examples(),
                                        report)
            print(text)
        if opt['num_examples'] > 0 and cnt >= opt['num_examples']:
            break
    if world.epoch_done():
        print("EPOCH DONE")
    print('finished evaluating task {} using datatype {}'.format(
          opt['task'], opt.get('datatype', 'N/A')))
    report = world.report()
    print(report)

    print_announcements(opt)

    return report


if __name__ == '__main__':
    parser = setup_args()
    eval_model(parser.parse_args(print_args=False), print_parser=parser)
