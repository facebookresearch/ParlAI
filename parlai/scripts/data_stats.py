#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Count and display statistics of the data.

## Examples

```shell
parlai data_stats -t convai2 -dt train:ordered
```
"""
from parlai.core.params import ParlaiParser
from parlai.agents.fixed_response.fixed_response import FixedResponseAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger, nice_report
from parlai.core.metrics import AverageMetric
from parlai.core.dict import DictionaryAgent
from parlai.core.script import ParlaiScript, register_script

import parlai.utils.logging as logging


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, False, 'Compute data statistics')
    # Get command line arguments
    parser.add_argument('-n', '-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=10)
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


def _report(world, counts):
    report = world.report()
    for k, v in counts.items():
        if not isinstance(v, dict):
            report[k] = v
    return report


def verify(opt):
    if opt['datatype'] == 'train':
        logging.warn('changing datatype from train to train:ordered')
        opt['datatype'] = 'train:ordered'

    # create repeat label agent and assign it to the specified task
    opt['fixed_response'] = None
    agent = FixedResponseAgent(opt)
    world = create_task(opt, agent)
    opt.log()

    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    dictionary = DictionaryAgent(opt)
    ignore_tokens = opt.get('ignore_tokens').split(',')

    counts = {}
    for t in {'input', 'labels', 'both'}:
        counts[f'{t}/tokens'] = 0
        counts[f'{t}/utterances'] = 0
        counts[f'{t}/avg_utterance_length'] = None
        counts[f'{t}/unique_tokens'] = 0
        counts[f'{t}/unique_utterances'] = 0
        # for counting the stats..
        counts[f'{t}/token_dict'] = {}
        counts[f'{t}/utterance_dict'] = {}

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
    while not world.epoch_done() and world.total_exs < max_cnt:
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
                retxt = [t for t in tokens if keep_token(t)]
                counts[f'{itype}/tokens'] += len(retxt)
                counts['both/tokens'] += len(retxt)
                counts[f'{itype}/utterances'] += 1
                counts['both/utterances'] += 1
                counts[f'{itype}/avg_utterance_length'] += AverageMetric(len(retxt), 1)
                counts[f'both/avg_utterance_length'] += AverageMetric(len(retxt), 1)
                for t in retxt:
                    if t not in counts[f'{itype}/token_dict']:
                        counts[f'{itype}/unique_tokens'] += 1
                        counts[f'{itype}/token_dict'][t] = True
                    if t not in counts['both/token_dict']:
                        counts['both/unique_tokens'] += 1
                        counts['both/token_dict'][t] = True
                retxt = ' '.join(retxt)
                if retxt not in counts[f'{itype}/utterance_dict']:
                    counts[f'{itype}/unique_utterances'] += 1
                    counts[f'{itype}/utterance_dict'][retxt] = True
                if retxt not in counts['both/utterance_dict']:
                    counts['both/unique_utterances'] += 1
                    counts['both/utterance_dict'][retxt] = True

        if log_time.time() > log_every_n_secs:
            report = _report(world, counts)
            cnt = report.pop('exs')
            text, log = log_time.log(cnt, world.num_examples(), report)
            logging.info(text)

    try:
        # print dataset size if available
        logging.info(
            f'loaded {world.num_episodes()} episodes with a total '
            f'of {world.num_examples()} examples'
        )
    except AttributeError:
        pass

    retval = _report(world, counts)
    retval.pop('exs')
    return retval


def obtain_stats(opt):
    report = verify(opt)
    print(nice_report(report))
    return report


@register_script('data_stats', hidden=True)
class DataStats(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return obtain_stats(self.opt)


if __name__ == '__main__':
    DataStats.main()
