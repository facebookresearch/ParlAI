#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Verify data doesn't have basic mistakes, like empty text fields or empty label
candidates.

Examples
--------

.. code-block:: shell

  python parlai/scripts/verify_data.py -t convai2 -dt train:ordered
"""
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.utils.misc import TimeLogger, warn_once
from parlai.core.worlds import create_task


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
    log = {
        'missing_text': counts['missing_text'],
        'missing_labels': counts['missing_labels'],
        'missing_label_candidates': counts['missing_label_candidates'],
        'empty_string_label_candidates': counts['empty_string_label_candidates'],
        'label_candidates_with_missing_label': counts[
            'label_candidates_with_missing_label'
        ],
        'did_not_return_message': counts['did_not_return_message'],
    }
    text, log = log_time.log(report['exs'], world.num_examples(), log)
    return text, log


def warn(txt, act, opt):
    if opt.get('display_examples'):
        print(txt + ":\n" + str(act))
    else:
        warn_once(txt)


def verify(opt, printargs=None, print_parser=None):
    if opt['datatype'] == 'train':
        print("[ note: changing datatype from train to train:ordered ]")
        opt['datatype'] = 'train:ordered'
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
    counts['empty_string_label_candidates'] = 0
    counts['label_candidates_with_missing_label'] = 0
    counts['did_not_return_message'] = 0

    # Show some example dialogs.
    while not world.epoch_done():
        world.parley()

        act = world.acts[0]

        if not isinstance(act, Message):
            counts['did_not_return_message'] += 1

        if 'text' not in act and 'image' not in act:
            warn("warning: missing text field:\n", act, opt)
            counts['missing_text'] += 1

        if 'labels' not in act and 'eval_labels' not in act:
            warn("warning: missing labels/eval_labels field:\n", act, opt)
            counts['missing_labels'] += 1
        else:
            if 'label_candidates' not in act:
                counts['missing_label_candidates'] += 1
            else:
                labels = act.get('labels', act.get('eval_labels'))
                is_label_cand = {}
                for l in labels:
                    is_label_cand[l] = False
                for c in act['label_candidates']:
                    if c == '':
                        warn("warning: empty string label_candidate:\n", act, opt)
                        counts['empty_string_label_candidates'] += 1
                    if c in is_label_cand:
                        if is_label_cand[c] is True:
                            warn(
                                "warning: label mentioned twice in candidate_labels:\n",
                                act,
                                opt,
                            )
                        is_label_cand[c] = True
                for _, has in is_label_cand.items():
                    if has is False:
                        warn("warning: label missing in candidate_labels:\n", act, opt)
                        counts['label_candidates_with_missing_label'] += 1

        if log_time.time() > log_every_n_secs:
            text, log = report(world, counts, log_time)
            if print_parser:
                print(text)

    try:
        # print dataset size if available
        print(
            '[ loaded {} episodes with a total of {} examples ]'.format(
                world.num_episodes(), world.num_examples()
            )
        )
    except Exception:
        pass

    return report(world, counts, log_time)


if __name__ == '__main__':
    parser = setup_args()
    report_text, report_log = verify(
        parser.parse_args(print_args=False), print_parser=parser
    )
    print(report_text)
