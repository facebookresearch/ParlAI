#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Count and display statistics of the data.

Examples
--------

.. code-block:: shell

  python parlai/scripts/data_stats.py -t convai2 -dt train:ordered
"""
from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.dict import DictionaryAgent

import parlai.utils.logging as logging


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, False, 'Lint for ParlAI tasks')
    # Get command line arguments
    parser.add_argument('-n', '-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument(
        '--agent',
        type=int,
        default=0,
        help='Use teacher (agent 0) or model (agent 1)',
        choices=[0, 1],
    )
    parser.add_argument(
        '--new_line_new_utt',
        type='bool',
        default=False,
        help='New lines treat substrings as separate utterances.',
    )
    parser.add_argument(
        '--ignore_tokens',
        type=str,
        default='',
        help='ignore tokens containings these substrings (comma-separated)',
    )
    parser.set_defaults(datatype='train:ordered')
    DictionaryAgent.add_cmdline_args(parser)
    return parser


def report(world, counts, log_time):
    report = world.report()
    stats = '\n'
    for t in ['input', 'labels', 'both']:
        stats += t + ":\n"
        for s in [
            'utterances_in_',
            'avg_utterance_length_in_',
            'tokens_in_',
            'unique_tokens_in_',
            'unique_utterances_in_',
        ]:
            snice = s.replace('_in_', '').replace('_', ' ')
            stats += "   " + snice + ': ' + str(counts[s + t]) + '\n'
    log = {}
    log['stats'] = stats
    text, log = log_time.log(report['exs'], world.num_examples(), log)
    return text, log


def verify(opt, printargs=None, print_parser=None):
    if opt['datatype'] == 'train':
        logging.warn('changing datatype from train to train:ordered')
        opt['datatype'] = 'train:ordered'

    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    dictionary = DictionaryAgent(opt)
    ignore_tokens = opt.get('ignore_tokens').split(',')

    counts = {}
    for t in {'input', 'labels', 'both'}:
        counts['tokens_in_' + t] = 0
        counts['utterances_in_' + t] = 0
        counts['avg_utterance_length_in_' + t] = 0
        counts['unique_tokens_in_' + t] = 0
        counts['unique_utterances_in_' + t] = 0
        # for counting the stats..
        counts['token_dict_' + t] = {}
        counts['utterance_dict_' + t] = {}

    def tokenize(txt):
        return dictionary.tokenize(txt)

    def keep_token(t):
        for s in ignore_tokens:
            if s != '' and s in t:
                return False
        return True

    # max number of examples to evaluate
    max_cnt = opt['num_examples'] if opt['num_examples'] > 0 else float('inf')
    cnt = 0

    # Show some example dialogs.
    while not world.epoch_done() and cnt < max_cnt:
        cnt += opt.get('batchsize', 1)
        world.parley()
        act = world.get_acts()[opt.get('agent')]
        for itype in {'input', 'labels'}:
            if itype == 'input':
                if opt.get('new_line_new_utt'):
                    txts = act.get('text').split('\n')
                else:
                    txts = [act.get('text')]
            else:
                txts = act.get('labels', act.get('eval_labels', ['']))

            for txt in txts:
                tokens = tokenize(txt)
                retxt = []
                for t in tokens:
                    if keep_token(t):
                        retxt.append(t)
                counts['tokens_in_' + itype] += len(retxt)
                counts['tokens_in_' + 'both'] += len(retxt)
                counts['utterances_in_' + itype] += 1
                counts['utterances_in_' + 'both'] += 1
                counts['avg_utterance_length_in_' + itype] = (
                    counts['tokens_in_' + itype] / counts['utterances_in_' + itype]
                )
                counts['avg_utterance_length_in_' + 'both'] = (
                    counts['tokens_in_' + 'both'] / counts['utterances_in_' + 'both']
                )
                for t in retxt:
                    if t not in counts['token_dict_' + itype]:
                        counts['unique_tokens_in_' + itype] += 1
                        counts['token_dict_' + itype][t] = True
                    if t not in counts['token_dict_' + 'both']:
                        counts['unique_tokens_in_' + 'both'] += 1
                        counts['token_dict_' + 'both'][t] = True
                retxt = ' '.join(retxt)
                if retxt not in counts['utterance_dict_' + itype]:
                    counts['unique_utterances_in_' + itype] += 1
                    counts['utterance_dict_' + itype][retxt] = True
                if retxt not in counts['utterance_dict_' + 'both']:
                    counts['unique_utterances_in_' + 'both'] += 1
                    counts['utterance_dict_' + 'both'][retxt] = True

        if log_time.time() > log_every_n_secs:
            text, log = report(world, counts, log_time)
            if print_parser:
                logging.info(text)

    try:
        # print dataset size if available
        logging.info(
            f'loaded {world.num_episodes()} episodes with a total '
            f'of {world.num_examples()} examples'
        )
    except Exception:
        pass
    return report(world, counts, log_time)


if __name__ == '__main__':
    parser = setup_args()
    report_text, report_log = verify(
        parser.parse_args(print_args=False), print_parser=parser
    )
    print(report_text.replace('\\n', '\n'))
