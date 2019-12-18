#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base script for running official ConvAI2 validation eval for perplexity. This uses a the
version of the dataset which does not contain candidates. Leaderboard scores will be run
in the same form but on a hidden test set.

The official vocabulary for the competition is based on using the
"split_tokenize" method on in the ParlAI core dictionary (parlai/core/dict.py)
and is built on the training and validation sets of the "convai2" task.
This dictionary contains a total of 19304 tokens. The test set contains some
tokens which are not in this dictionary--this tokens will not be provided, but
we will also *SKIP* calculating perplexity on these tokens. The model should
still produce a good guess for the remaining tokens in the sentence, so
handling unknown words or expanding the vocabulary with pre-trained or
multitasked embeddings are legitimate strategies that may or may not impact the
score of the models.

Note that this tokenizer will also be used during the perplexity evaluation:
the model will be asked to predict one word at a time according to this
tokenizer's parsing of the text.

This requires agents to implement the following function:

def next_word_probability(self, partial_out):
    Return probability distribution over next words given a partial true output.
    This is used to calculate the per-word perplexity.

    Arguments:
    partial_out -- list of previous "true" words

    Returns a dict, where each key is a word and each value is a probability
    score for that word. Unset keys assume a probability of zero.

    e.g.
    {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
"""

from parlai.core.agents import Agent

from parlai.scripts.eval_ppl import (
    eval_ppl as run_eval_ppl,
    setup_args as setup_ppl_args,
)
from projects.convai2.build_dict import build_dict


def setup_args(parser=None):
    parser = setup_ppl_args(parser)
    parser.set_defaults(
        task='convai2:self:no_cands', datatype='valid', dict_tokenizer='split'
    )
    return parser


class WordFrequencyEntry(Agent):
    """
    This is an example entry which tries to use the RepeatLabelAgent. Since no labels
    are given to the model, it will guess something useless.

    It builds the official dictionary first, so that it can provide a minimum probablity
    for each word as well as use the official tokenizer.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            # build official eval dictionary
            self.dict = build_dict()
        else:
            # only build dict once
            self.dict = shared['dict']
        max_freq = self.dict.max_freq()
        # set probability of each word, skipping the invalid words like __NULL__
        # (which have frequency more than max_freq)
        self.freqs = {k: f for k, f in self.dict.freqs().items() if f <= max_freq}

    def share(self):
        shared = super().share()
        # share dict with other threads instead of rebuilding in each
        shared['dict'] = self.dict
        return shared

    def next_word_probability(self, partial_out):
        """
        Example implementation of next word probability.
        """
        obs = self.observation
        # initialize probabilities with inverse word frequency
        freqs = self.freqs.copy()

        # increase likelihood of predicting input words
        tokens = self.dict.tokenize(obs.get('text', ''))
        for t in tokens:
            if t in freqs:
                freqs[t] += 10000
        return freqs


def eval_ppl(opt):
    return run_eval_ppl(opt, build_dict)


if __name__ == '__main__':
    parser = setup_args()
    # example model just uses word frequencies
    parser.set_defaults(model='projects.convai2.eval_ppl:WordFrequencyEntry')
    # try with --numthreads N to go fast
    opt = parser.parse_args()
    eval_ppl(opt)
    if opt['model'] == 'projects.convai2.eval_ppl:WordFrequencyEntry':
        print(
            'This run just used the example filler model. To get better '
            'results, try implementing your own!'
        )
