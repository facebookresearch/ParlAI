#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This helper script can be used alone with modelfile and task: the output will contain
the word statistics of the model outputs. One can also use the function defined here in
other places in order to get such statistic for any agent given the agent object (with
corr. dict) and a sequence.

Additionally provides function `get_word_stats` that can be used in
other parts of runtime code since it depends only on the agent object.
For example:

```python
from parlai.scripts.eval_wordstat import get_word_stats
reqs, cnt = get_word_stats(predictions.tolist(), self.dict)
```

## Examples

```shell
parlai eval_wordstat -mf data/model -t convai2:self --freq-bins 10,100,1000
```
"""

from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.metrics import normalize_answer
from parlai.core.logs import TensorboardLogger
from collections import Counter
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.io import PathManager

import copy
import numpy
import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Compute statistics from model predictions')
    DictionaryAgent.add_cmdline_args(parser)
    # Get command line arguments
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument(
        '-ed',
        '--external-dict',
        type=str,
        default=None,
        help='External dictionary for stat computation',
    )
    parser.add_argument(
        '-fb',
        '--freq-bins',
        type=str,
        default='0,100,1000,10000',
        help='Bins boundaries for rare words stat',
    )
    parser.add_argument(
        '-dup',
        '--dump-predictions-path',
        type=str,
        default=None,
        help='Dump predictions into file',
    )
    parser.add_argument(
        '-cun',
        '--compute-unique',
        type='bool',
        default=True,
        help='Compute %% of unique responses from the model',
    )
    parser.set_defaults(datatype='valid')
    TensorboardLogger.add_cmdline_args(parser)
    return parser


def get_word_stats(text, agent_dict, bins=(0, 100, 1000, 100000)):
    """
    Function which takes text sequence and dict, returns word freq and length
    statistics.

    :param sequence: text sequence
    :param agent_dict: can be external dict or dict from the model
    :param bins: list with range boundaries
    :return: freqs dictionary, num words, avg word length, avg char length
    """
    pred_list = agent_dict.tokenize(text)
    pred_freq = [agent_dict.freq[word] for word in pred_list]
    freqs = {i: 0 for i in bins}
    for f in pred_freq:
        for b in bins:
            if f <= b:
                freqs[b] += 1
                break

    wlength = len(pred_list)
    clength = len(text)  # including spaces
    return freqs, len(pred_freq), wlength, clength


def eval_wordstat(opt):
    """
    Evaluates a model.

    :param opt: tells the evaluation function how to run
    """
    random.seed(42)

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)
    agent.opt.log()

    if opt.get('external_dict'):
        print('[ Using external dictionary from: {} ]'.format(opt['external_dict']))
        dict_opt = copy.deepcopy(opt)
        dict_opt['dict_file'] = opt['external_dict']
        dictionary = DictionaryAgent(dict_opt)
    else:
        print('[ Using model bundled dictionary ]')
        dictionary = agent.dict

    batch_size = opt['batchsize']

    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    cnt = 0
    max_cnt = opt['num_examples'] if opt['num_examples'] > 0 else float('inf')
    word_statistics = {
        'mean_wlength': [],
        'mean_clength': [],
        'freqs_cnt': Counter(),
        'word_cnt': 0,
        'pred_list': [],
        'pure_pred_list': [],
        'context_list': [],
        'unique_words': set(),
    }
    bins = [int(i) for i in opt['freq_bins'].split(',')]

    def process_prediction(prediction, word_statistics):
        normalized = normalize_answer(prediction)
        word_statistics['pred_list'].append(normalized)
        freqs, _cnt, wlength, clength = get_word_stats(
            prediction, dictionary, bins=bins
        )
        word_statistics['word_cnt'] += _cnt
        word_statistics['mean_wlength'].append(wlength)
        word_statistics['mean_clength'].append(clength)
        word_statistics['freqs_cnt'] += Counter(freqs)
        word_statistics['unique_words'] |= set(normalized.split(" "))
        return word_statistics

    while not world.epoch_done():
        world.parley()
        if batch_size == 1:
            cnt += 1
            prediction = world.acts[-1]['text']
            word_statistics['context_list'].append(world.acts[0]['text'])
            word_statistics['pure_pred_list'].append(prediction)
            word_statistics = process_prediction(prediction, word_statistics)
        else:
            for w in world.worlds:
                try:
                    if 'text' not in w.acts[-1]:
                        continue
                    prediction = w.acts[-1]['text']
                    word_statistics['context_list'].append(w.acts[0]['text'])
                    word_statistics['pure_pred_list'].append(prediction)
                except IndexError:
                    continue
                cnt += 1
                word_statistics = process_prediction(prediction, word_statistics)

        if log_time.time() > log_every_n_secs:
            report = world.report()
            text, report = log_time.log(
                report['exs'], min(max_cnt, world.num_examples()), report
            )
            print(text)
            stat_str = 'total_words: {}, '.format(word_statistics['word_cnt'])
            stat_str += ', '.join(
                [
                    '<{}:{} ({:.{prec}f}%)'.format(
                        b,
                        word_statistics['freqs_cnt'].get(b, 0),
                        (
                            word_statistics['freqs_cnt'].get(b, 0)
                            / word_statistics['word_cnt']
                        )
                        * 100,
                        prec=2,
                    )
                    for b in bins
                ]
            )
            print(
                "Word statistics: {}, avg_word_length: {:.{prec}f}, "
                "avg_char_length: {:.{prec}f}".format(
                    stat_str,
                    numpy.array(word_statistics['mean_wlength']).mean(),
                    numpy.array(word_statistics['mean_clength']).mean(),
                    prec=2,
                )
            )
        if cnt >= max_cnt:
            break
    if world.epoch_done():
        print("EPOCH DONE")

    if opt['compute_unique'] is True:
        unique_list = []
        cntr = Counter(word_statistics['pred_list'])
        for k, v in cntr.items():
            if v == 1:
                unique_list.append(k)
        print(
            "Unique responses: {:.{prec}f}%".format(
                len(unique_list) / len(word_statistics['pred_list']) * 100, prec=2
            )
        )
    print("Total unique tokens:", len(word_statistics['unique_words']))

    if opt['dump_predictions_path'] is not None:
        with PathManager.open(opt['dump_predictions_path'], 'w') as f:
            f.writelines(
                [
                    'CONTEXT: {}\nPREDICTION:{}\n\n'.format(c, p)
                    for c, p in zip(
                        word_statistics['context_list'],
                        word_statistics['pure_pred_list'],
                    )
                ]
            )
        if opt['compute_unique'] is True:
            with PathManager.open(opt['dump_predictions_path'] + '_unique', 'w') as f:
                f.writelines(['{}\n'.format(i) for i in unique_list])

    stat_str = 'total_words: {}, '.format(word_statistics['word_cnt'])
    stat_str += ', '.join(
        [
            '<{}:{} ({:.{prec}f}%)'.format(
                b,
                word_statistics['freqs_cnt'].get(b, 0),
                (word_statistics['freqs_cnt'].get(b, 0) / word_statistics['word_cnt'])
                * 100,
                prec=2,
            )
            for b in bins
        ]
    )
    print(
        "Word statistics: {}, avg_word_length: {:.{prec}f}, "
        "avg_char_length: {:.{prec}f}".format(
            stat_str,
            numpy.array(word_statistics['mean_wlength']).mean(),
            numpy.array(word_statistics['mean_clength']).mean(),
            prec=2,
        )
    )

    report = world.report()
    print(report)
    return report


@register_script('eval_wordstat', hidden=True)
class EvalWordStat(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return eval_wordstat(self.opt)


if __name__ == '__main__':
    EvalWordStat.main()
