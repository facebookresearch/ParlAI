#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic example which iterates through the tasks specified and evaluates the given model
on them.

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
from parlai.utils.misc import TimeLogger

import json
import random
from tqdm import tqdm


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Evaluate a model')
    parser.add_pytorch_datateacher_args()
    # Get command line arguments
    parser.add_argument(
        '-rf',
        '--report-filename',
        type=str,
        default='',
        help='Saves a json file of the evaluation report either as an '
        'extension to the model-file (if begins with a ".") or a whole '
        'file path. Set to the empty string to not save at all.',
    )
    parser.add_argument(
        '--save-model-replies',
        type='bool',
        default=False,
        help='Saves a jsonl file containing all of the task examples and '
        'model replies. Must also specify --report-filename.',
    )
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
        '-mcs',
        '--metrics',
        type=str,
        default='default',
        help='list of metrics to show/compute, e.g. all, default,'
        'or give a list split by , like '
        'ppl,f1,accuracy,hits@1,rouge,bleu'
        'the rouge metrics will be computed as rouge-1, rouge-2 and rouge-l',
    )
    TensorboardLogger.add_cmdline_args(parser)
    parser.set_defaults(datatype='valid')
    return parser


def _save_eval_stats(opt, report, replies):
    report_fname = opt['report_filename']
    if report_fname == '':
        return
    if report_fname.startswith('.'):
        report_fname = opt['model_file'] + report_fname

    # Save report
    with open(report_fname, 'w') as f:
        print(f'[ Saving model report to {report_fname} ... ]')
        json.dump({'opt': opt, 'report': report}, f, indent=4)

    # Save model replies
    if opt['save_model_replies']:
        for task, replies in replies.items():
            base_name = report_fname.split('.')[0]
            replies_fname = base_name + f'.{task}_replies.jsonl'
            print(f'[ Saving model replies for task {task} to {replies_fname} ... ]')
            with open(replies_fname, 'w') as f:
                for reply in tqdm(replies):
                    json_reply = json.dumps(reply)
                    f.write(json_reply + '\n')


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

    model_replies = []
    while not world.epoch_done() and cnt < max_cnt:
        cnt += opt.get('batchsize', 1)
        world.parley()
        if opt['save_model_replies']:
            acts = world.get_acts()
            model_replies.append(acts)
        if opt['display_examples']:
            # display examples
            print(world.display() + '\n~~')
        if log_time.time() > log_every_n_secs:
            report = world.report()
            text, report = log_time.log(report['exs'], world.num_examples(), report)
            print(text)

    report = world.report()
    world.reset()
    return report, model_replies


def eval_model(opt, print_parser=None):
    """
    Evaluates a model.

    :param opt: tells the evaluation function how to run
    :param bool print_parser: if provided, prints the options that are set within the
        model after loading the model
    :return: the final result of calling report()
    """
    random.seed(42)
    if 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']:
        raise ValueError(
            'You should use --datatype train:evalmode if you want to evaluate on '
            'the training set.'
        )

    if opt['save_model_replies'] and not opt['report_filename']:
        raise RuntimeError(
            'In order to save model replies, please specify the save path '
            'with --report-filename'
        )

    # load model and possibly print opt
    agent = create_agent(opt, requireModelExists=True)
    if print_parser:
        # show args after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()

    tasks = opt['task'].split(',')
    reports = []
    replies = {}
    for task in tasks:
        task_report, model_replies = _eval_single_world(opt, agent, task)
        reports.append(task_report)
        if opt['save_model_replies']:
            replies[task] = model_replies

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
    _save_eval_stats(opt, report, replies)
    return report


if __name__ == '__main__':
    parser = setup_args()
    eval_model(parser.parse_args(print_args=False))
