#!/usr/bin/env python
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""
This helper script can be used alone with modelfile and task: the output will contain the
word statistics of the model outputs.
One can also use the function defined here in other places in order to get such statistic
for any agent given the agent object (with corr. dict) and a sequence.

Example:
    python eval_wordstat.py -mf data/model -t convai2:self

One can specify bins boundaries with argument -fb | --freq-bins 10,100,1000 or so

Also function get_word_stats can be used in other parts of runtime code since it depends only on
the agent object. To use it - firstly do the import:

    from parlai.scripts.eval_wordstat import get_word_stats

then you can call this function like this:

    reqs, cnt = get_word_stats(predictions.tolist(), self.dict)
"""

from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.utils import TimeLogger
from parlai.core.metrics import normalize_answer
from collections import Counter

import copy
import numpy
import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    DictionaryAgent.add_cmdline_args(parser)
    # Get command line arguments
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('-ed', '--external-dict', type=str, default=None,
                        help='External dictionary for stat computation')
    parser.add_argument('-fb', '--freq-bins', type=str, default='0,100,1000,10000',
                        help='Bins boundaries for rare words stat')
    parser.add_argument('-dup', '--dump-predictions-path', type=str, default=None,
                        help='Dump predictions into file')
    parser.add_argument('-cun', '--compute-unique', type=bool, default=True,
                        help='Compute % of unique responses from the model')
    parser.set_defaults(datatype='valid', model='repeat_label')
    return parser


def get_word_stats(text, agent_dict, bins=[0, 100, 1000, 100000]):
    """
    Function which takes text sequence and dict, returns word freq and length statistics
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


def eval_wordstat(opt, print_parser=None):
    """Evaluates a model.

    Arguments:
    opt -- tells the evaluation function how to run
    print_parser -- if provided, prints the options that are set within the
        model after loading the model
    """
    random.seed(42)

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)

    if opt.get('external_dict'):
        print('[ Using external dictionary from: {} ]'.format(
            opt['external_dict']))
        dict_opt = copy.deepcopy(opt)
        dict_opt['dict_file'] = opt['external_dict']
        dictionary = DictionaryAgent(dict_opt)
    else:
        print('[ Using model bundled dictionary ]')
        dictionary = agent.dict

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    cnt = 0
    mean_wlength = []
    mean_clength = []
    freqs_cnt = Counter()
    word_cnt = 0
    bins = [int(i) for i in opt['freq_bins'].split(',')]
    pred_list = []

    while not world.epoch_done():
        cnt += 1
        world.parley()
        prediction = world.acts[-1]['text']
        pred_list.append(normalize_answer(prediction))
        freqs, _cnt, wlength, clength = get_word_stats(prediction, dictionary, bins=bins)
        word_cnt += _cnt

        mean_wlength.append(wlength)
        mean_clength.append(clength)

        freqs_cnt += Counter(freqs)

        if log_time.time() > log_every_n_secs or (opt['num_examples'] > 0 and cnt >= opt['num_examples']) or world.epoch_done():
            report = world.report()
            text, report = log_time.log(report['exs'], world.num_examples(), report)
            print(text)
            stat_str = 'total_words: {}, '.format(word_cnt) + ', '.join(
                ['<{}:{} ({:.{prec}f}%)'.format(b, freqs_cnt.get(b, 0), (freqs_cnt.get(b, 0) / word_cnt) * 100, prec=2)
                 for b in bins])
            print("Word statistics: {}, avg_word_length: {:.{prec}f}, avg_char_length: {:.{prec}f}".format(
                stat_str, numpy.array(mean_wlength).mean(), numpy.array(mean_clength).mean(), prec=2))
        if opt['num_examples'] > 0 and cnt >= opt['num_examples']:
            break
    if world.epoch_done():
        print("EPOCH DONE")

    if opt['compute_unique'] is True:
        unique_list = []
        cntr = Counter(pred_list)
        for k,v in cntr.items():
            if v == 1:
                unique_list.append(k)
        print("Unique responses: {:.{prec}f}%".format(len(unique_list) / len(pred_list) * 100, prec=2))

    if opt['dump_predictions_path'] is not None:
        with open(opt['dump_predictions_path'], 'w') as f:
            f.writelines(['{}\n'.format(i) for i in pred_list])
        if opt['compute_unique'] is True:
            with open(opt['dump_predictions_path']+'_unique', 'w') as f:
                f.writelines(['{}\n'.format(i) for i in unique_list])

    report = world.report()
    print(report)
    return report


if __name__ == '__main__':
    parser = setup_args()
    eval_wordstat(parser.parse_args(print_args=False), print_parser=parser)
