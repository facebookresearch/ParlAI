#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base script for model-agnostic perplexity evaluation.

While resistent to choices of model-added tokens like START and END, this
requires fixing a specific vocabulary. Be sure to use the same build_dict
parameters for all comparisons.

Tokens which are present in the data being evaluated but not in the vocabulary
do not contribute to the perplexity score, but they are still sent to the model
so the model can update its state. If the token is in the vocabulary but
receives a probability score of zero by the model, the model will get a
perplexity score of `inf`.

This requires agents to implement the following function:

def next_word_probability(self, partial_out):
    Return probability distribution over next words given a partial true output.
    This is used to calculate the per-word perplexity.

    Arguments:
    partial_out -- list of previous "true" words

    Returns a dict, where each key is a word and each value is a probability
    score for that word. Unset keys assume a probability of zero.

    e.g.
    (previous observation: {'text': 'Run test program.'})
    [] => {'hello': 1.0}
    ['hello'] => {'world': 1.0}
"""

from parlai.core.agents import create_agent, create_agents_from_shared
from parlai.core.params import ParlaiParser
from parlai.utils.misc import Timer, round_sigfigs, no_lock
from parlai.utils.thread import SharedTable
from parlai.core.worlds import create_task, World

import copy
import math


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Evaluate perplexity')
    parser.set_defaults(datatype='valid')
    return parser


class PerplexityWorld(World):
    """
    Instead of just calling act/observe on each agent, this world just calls act on the
    teacher and then calls `next_word_probability` on the agent.

    The label for each example is parsed by the provided tokenizer, and then
    for each word in the parsed label the model is given the input and all of
    the tokens up to the current word and asked to predict the current word.

    The model must return a probability of any words it thinks are likely in
    the form of a dict mapping words to scores. If the scores do not sum to 1,
    they are normalized to do so. If the correct word is not present or has a
    probablity of zero, it will be assigned a probability of 1e-8.

    The API of the next_word_probability function which agents must implement
    is mentioned in the documentation for this file.
    """

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt)
        if shared:
            # Create agents based on shared data.
            self.task, self.agent, self.dict = create_agents_from_shared(
                shared['agents']
            )
            self.metrics = shared['metrics']
        else:
            if len(agents) != 3:
                raise RuntimeError('There must be exactly three agents.')
            if opt.get('batchsize', 1) > 1:
                raise RuntimeError(
                    'This world only works with bs=1. Try '
                    'using multiple threads instead, nt>1.'
                )
            self.task, self.agent, self.dict = agents
            if not hasattr(self.agent, 'next_word_probability'):
                raise RuntimeError(
                    'Agent must implement function ' '`next_word_probability`.'
                )
            self.metrics = {'exs': 0, 'loss': 0.0, 'num_tokens': 0, 'num_unk': 0}
            if opt.get('numthreads', 1) > 1:
                self.metrics = SharedTable(self.metrics)
        self.agents = [self.task, self.agent, self.dict]
        self.acts = [None, None]

    def _lock(self):
        if hasattr(self.metrics, 'get_lock'):
            # use the shared_table's lock
            return self.metrics.get_lock()
        else:
            # otherwise do nothing
            return no_lock()

    def parley(self):
        action = self.task.act()
        self.acts[0] = action.copy()

        # hide labels from model
        labels = action.get('eval_labels', action.get('labels', None))
        if 'label_candidates' in action:
            action.pop('label_candidates')
        if labels is None:
            # empty example, move on
            return

        parsed = self.dict.tokenize(labels[0])
        loss = 0
        num_tokens = 0
        num_unk = 0
        self.agent.observe(action)
        for i in range(len(parsed)):
            if parsed[i] in self.dict:
                # only score words which are in the dictionary
                probs = self.agent.next_word_probability(parsed[:i])
                # get probability of correct answer, divide by total prob mass
                prob_true = probs.get(parsed[i], 0)
                if prob_true > 0:
                    prob_true /= sum((probs.get(k, 0) for k in self.dict.keys()))
                    loss -= math.log(prob_true)
                else:
                    loss = float('inf')
                num_tokens += 1
            else:
                num_unk += 1
        with self._lock():
            self.metrics['exs'] += 1
            self.metrics['loss'] += loss
            self.metrics['num_tokens'] += num_tokens
            self.metrics['num_unk'] += num_unk

    def epoch_done(self):
        return self.task.epoch_done()

    def num_examples(self):
        return self.task.num_examples()

    def num_episodes(self):
        return self.task.num_episodes()

    def share(self):
        shared = super().share()
        shared['metrics'] = self.metrics
        return shared

    def reset_metrics(self):
        with self._lock():
            self.metrics['exs'] = 0
            self.metrics['loss'] = 0
            self.metrics['num_tokens'] = 0
            self.metrics['num_unk'] = 0

    def report(self):
        m = {}
        with self._lock():
            m['exs'] = self.metrics['exs']
            if m['exs'] > 0:
                # m['num_unk'] = self.metrics['num_unk']
                # m['num_tokens'] = self.metrics['num_tokens']
                m['loss'] = round_sigfigs(
                    self.metrics['loss'] / self.metrics['num_tokens'], 3
                )
                m['ppl'] = round_sigfigs(
                    math.exp(self.metrics['loss'] / self.metrics['num_tokens']), 4
                )
        return m


def eval_ppl(opt, build_dict=None, dict_file=None):
    """
    Evaluates the the perplexity of a model.

    This uses a dictionary which implements the following functions:
    - tokenize(text): splits string up into list of tokens
    - __in__(text): checks whether dictionary contains a token
    - keys(): returns an iterator over all tokens in the dictionary

    :param opt: option dict
    :param build_dict: function which returns a dictionary class implementing
        the functions above.
    :param dict_file: file used when loading the dictionary class set via the
        "dictionary_class" argument (defaults to
        parlai.core.dict:DictionaryAgent).

    Either build_dict or dict_file must be set (both default to None) to
    determine the dictionary used for the evaluation.
    """
    if not build_dict and not dict_file:
        raise RuntimeError(
            'eval_ppl script either needs a dictionary build '
            'function or a dictionary file.'
        )

    if build_dict:
        dict_agent = build_dict()
    else:
        dict_opt = copy.deepcopy(opt)
        dict_opt['model'] = dict_opt.get(
            'dictionary_class', 'parlai.core.dict:DictionaryAgent'
        )
        dict_opt['model_file'] = dict_file
        if 'override' in dict_opt:
            del dict_opt['override']
        dict_agent = create_agent(dict_opt, requireModelExists=True)

    # create agents
    agent = create_agent(opt)
    world = create_task(opt, [agent, dict_agent], default_world=PerplexityWorld)

    # set up logging
    log_time = Timer()
    tot_time = 0

    while not world.epoch_done():
        world.parley()  # process an example

        if log_time.time() > 1:  # log every 1 sec
            tot_time += log_time.time()
            report = world.report()
            print(
                '{}s elapsed, {}%% complete, {}'.format(
                    int(tot_time),
                    round_sigfigs(report['exs'] / world.num_examples() * 100, 3),
                    report,
                )
            )
            log_time.reset()
    print('EPOCH DONE')
    tot_time += log_time.time()
    final_report = world.report()
    print('{}s elapsed: {}'.format(int(tot_time), final_report))
    print("============================")
    print("FINAL PPL: " + str(final_report['ppl']))
    if final_report.get('ppl', 0) == float('inf'):
        print(
            'Note: you got inf perplexity. Consider adding (or raising) the '
            'minimum probability you assign to each possible word. If you '
            'assign zero probability to the correct token in the evaluation '
            'vocabulary, you get inf probability immediately.'
        )


if __name__ == '__main__':
    parser = setup_args()
    # try with --numthreads N to go fast
    opt = parser.parse_args()
    eval_ppl(opt, dict_file=opt.get('dict_file'))
