#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is derived from parlai/core/scripts/eval_wordstat.py.

This script measures many different metrics of the text generated for the validation
set - including all the controllable attributes.
"""

from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.metrics import normalize_answer
from parlai.core.logs import TensorboardLogger
from controllable_seq2seq.controls import (
    ATTR2SENTSCOREFN,
    eval_attr,
    initialize_control_information,
)
from controllable_seq2seq.util import ConvAI2History
from collections import Counter

import copy
import random
import json
import time
import os


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'compute statistics from model predictions')
    DictionaryAgent.add_cmdline_args(parser)

    # These defaults can be overriden by both .opt file and user's command line flags
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
        '-gr',
        '--gold-response',
        type=bool,
        default=False,
        help='Compute stats for gold response',
    )

    # These settings override .opt file but not user's command line flags
    parser.set_params(
        datatype='valid',
        task='projects.controllable_dialogue.tasks.agents',
        model='projects.controllable_dialogue.controllable_seq2seq.controllable_seq2seq:ControllableSeq2seqAgent',  # noqa: E501
        batchsize=64,
        beam_size=20,
        beam_min_n_best=10,
        use_reply='model',
    )
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


def update_sent_attr_stats(sent_attrs, history, prediction):
    """
    Update the sent_attrs dict with the attributes of a prediction with given history.

    Inputs:
      sent_attrs: dictionary mapping each attr (a string) to a list of floats
        (the scores).
      history: a ConvAI2History
      prediction: string. the response text for which we measure sent attributes
    """
    for attr in sent_attrs.keys():
        attr_score = eval_attr(prediction, history, attr)
        sent_attrs[attr].append(attr_score)
    return sent_attrs


def eval_wordstat(opt, print_parser=None):
    """
    Evaluates a model.

    :param opt: tells the evaluation function how to run
    :param print_parser: if provided, prints the options that are set within the
        model after loading the model
    """
    random.seed(42)

    # Setup control information
    initialize_control_information(opt)

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)

    if opt.get('external_dict'):
        print('[ Using external dictionary from: {} ]'.format(opt['external_dict']))
        dict_opt = copy.deepcopy(opt)
        dict_opt['dict_file'] = opt['external_dict']
        dictionary = DictionaryAgent(dict_opt)
    else:
        print('[ Using model bundled dictionary ]')
        dictionary = agent.dict

    batch_size = opt['batchsize']

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    data = {}  # This will be written to the output json file
    data['opt'] = agent.opt  # Save the opt to json

    # Determine the output filename
    if opt['gold_response']:  # Special output file for gold response
        model_dir, _ = os.path.split(opt.get('model_file'))
        outfile = os.path.join(model_dir, 'goldresponse')
        if opt['use_reply'] != 'label':
            raise ValueError(
                'You should set --use-reply label (not --use-reply model) '
                'when measuring goldresponse stats'
            )
    else:
        outfile = "%s.%s.%s.%s" % (
            opt.get('model_file'),
            opt.get('datatype'),
            "use%sreply" % agent.opt['use_reply'],
            "beam%i" % agent.opt['beam_size'],
        )
        if agent.opt['beam_size'] > 1:
            outfile += ".beamminnbest%i" % agent.opt['beam_min_n_best']
        if len(agent.control_settings) > 0:
            outfile += ".setcontrols:" + "_".join(
                [
                    "%s%s" % (c, str(agent.control_settings[c]['set_value']))
                    for c in sorted(agent.control_settings.keys())
                ]
            )
        if agent.opt['beam_reorder'] not in ['none', False]:
            outfile += ".beamreorder_%s" % agent.opt['beam_reorder']
        if len(agent.wd_features) > 0:
            sorted_bfw = sorted(
                list(zip(agent.wd_features, agent.wd_wts)), key=lambda x: x[0]
            )
            outfile += ".WDfeatures:" + "_".join(
                ["%s%s" % (f, str(w)) for f, w in sorted_bfw]
            )
    if opt['num_examples'] != -1:
        outfile += ".numex%i" % opt['num_examples']
    outfile += ".wordstats.json"
    print("\nOutfile: %s\n" % outfile)

    cnt = 0
    word_statistics = {
        'mean_wlength': [],  # list of length (in words) of utterances
        'mean_clength': [],  # list of length (in chars) of utterances
        'freqs_cnt': Counter(),  # Counter for word frequencies, bucketed
        'word_cnt': 0,  # total number of words in all utterances
        'pred_list': [],  # list of generated utterances after applying normalize_answer
        'pure_pred_list': [],  # list of generated utterances
        'context_list': [],  # list of text inputs (persona and conversation history)
    }
    bins = [int(i) for i in opt['freq_bins'].split(',')]

    # This dictionary records all the sentence-level controllable attributes
    # For each attribute, we have a list of all the values
    sent_attrs = {attr: [] for attr in ATTR2SENTSCOREFN.keys()}  # str to list of floats

    # histories will be a list of ConvAI2History objects
    histories = []

    def process_prediction(prediction, word_statistics):
        word_statistics['pred_list'].append(normalize_answer(prediction))
        freqs, _cnt, wlength, clength = get_word_stats(
            prediction, dictionary, bins=bins
        )
        word_statistics['word_cnt'] += _cnt
        word_statistics['mean_wlength'].append(wlength)
        word_statistics['mean_clength'].append(clength)
        word_statistics['freqs_cnt'] += Counter(freqs)
        return word_statistics

    t0 = time.time()
    while not world.epoch_done():
        world.parley()
        # orig eval_wordstat.py handles bsz=1 but for simplicity we assume bsz>1
        assert batch_size != 1
        for w in world.worlds:
            try:
                try:
                    response_act = w.acts[-1]
                    prediction = response_act['text']
                except KeyError:
                    continue
                if opt['gold_response']:
                    # If we're measuring gold response, use eval_label as prediction
                    prediction = w.acts[0]['eval_labels'][0]
                    response_act = {'text': prediction}
                word_statistics['context_list'].append(w.acts[0]['text'])
                word_statistics['pure_pred_list'].append(prediction)
            except IndexError:
                continue
            cnt += 1
            word_statistics = process_prediction(prediction, word_statistics)

            # Compute and record sentence-level attributes
            history = ConvAI2History(w.acts[0]['text'])
            histories.append(history)
            sent_attrs = update_sent_attr_stats(sent_attrs, history, prediction)

        # Periodically log some info
        if log_time.time() > log_every_n_secs:
            report = world.report()
            text, report = log_time.log(report['exs'], world.num_examples(), report)
            print(text)

        if opt['num_examples'] > 0 and cnt >= opt['num_examples']:
            break
    if world.epoch_done():
        print("EPOCH DONE")
    print("Time to process %i examples: %f seconds" % (cnt, time.time() - t0))

    # Compute percent unique
    # Note this is w.r.t. normalized pred_list not original pure_pred_list
    unique_list = []
    cntr = Counter(word_statistics['pred_list'])
    for k, v in cntr.items():
        if v == 1:
            unique_list.append(k)
    unique_percent = len(unique_list) / len(word_statistics['pred_list']) * 100

    # Print a final report
    report = world.report()
    if opt['gold_response']:
        report['ppl'] = 0.0  # For gold responses, overwrite the perplexity
    print(report)

    # Put all information in data dict
    data['unique_percent'] = unique_percent  # percent of all responses that are unique
    data['word_statistics'] = word_statistics  # word stats, as in orig eval_wordstat
    data['report'] = report  # the final report
    data['histories'] = [
        (hist.persona_lines, hist.partner_utts, hist.own_utts) for hist in histories
    ]  # history for each example
    data['sent_attrs'] = sent_attrs  # all sentence attribute values for responses

    # Write data to outfile
    print("Writing to %s..." % outfile)
    with open(outfile, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    parser = setup_args()
    eval_wordstat(parser.parse_args(print_args=False), print_parser=parser)
