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
from parlai.core.metrics import aggregate_task_reports
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
    parser.add_argument(
        '-micro',
        '--aggregate-micro',
        type='bool',
        default=False,
        help='If multitasking, average metrics over the '
        'number of examples. If false, averages over the '
        'number of tasks.',
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default='all',
        help='list of metrics to show/compute, e.g. '
        'ppl, f1, accuracy, hits@1.'
        'If `all` is specified [default] all are shown.',
    )
    TensorboardLogger.add_cmdline_args(parser)
    parser.set_defaults(datatype='valid')
    return parser


def _eval_single_world(opt, agent, task):
    print(
        '[ Evaluating task {} using datatype {}. ] '.format(
            task, opt.get('datatype', 'N/A')
        )
    )
    task_opt = opt.copy()  # copy opt since we're editing the task
    task_opt['task'] = task
    world = create_task(task_opt, agent)  # create worlds for tasks

    # set up logging
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    # max number of examples to evaluate
    max_cnt = opt['num_examples'] if opt['num_examples'] > 0 else float('inf')
    cnt = 0

    while not world.epoch_done() and cnt < max_cnt:
        cnt += opt.get('batchsize', 1)
        world.parley()
        if opt['display_examples']:
            # display examples
            print(world.display() + '\n~~')
        if log_time.time() > log_every_n_secs:
            report = world.report()
            text, report = log_time.log(report['exs'], world.num_examples(), report)
            print(text)

    report = world.report()
    world.reset()
    return report


def eval_model(opt, print_parser=None):
    """Evaluates a model.

    :param opt: tells the evaluation function how to run
    :param bool print_parser: if provided, prints the options that are set within the
        model after loading the model
    :return: the final result of calling report()
    """
    random.seed(42)

    # load model and possibly print opt
    agent = create_agent(opt, requireModelExists=True)
    if print_parser:
        # show args after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()

    tasks = opt['task'].split(',')
    reports = []
    for task in tasks:
        task_report = _eval_single_world(opt, agent, task)
        reports.append(task_report)

    report = aggregate_task_reports(
        reports, tasks, micro=opt.get('aggregate_micro', True)
    )

    # print announcments and report
    print_announcements(opt)
    print(
        '[ Finished evaluating tasks {} using datatype {} ]'.format(
            tasks, opt.get('datatype', 'N/A')
        )
    )
    print(report)

    return report


if __name__ == '__main__':
    parser = setup_args()
    eval_model(parser.parse_args(print_args=False))
