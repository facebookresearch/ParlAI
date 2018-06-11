# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


from parlai.core.agents import Agent

from parlai.scripts.eval_ppl import eval_ppl as run_eval_ppl, setup_args as setup_ppl_args
from projects.twitter.build_dict import build_dict

import math


def setup_args(parser=None):
    parser = setup_ppl_args(parser)
    parser.set_defaults(
        task='twitter',
        datatype='valid',
    )
    return parser


class WordFrequencyEntry(Agent):
    """This is an example entry which tries to use the RepeatLabelAgent.
    Since no labels are given to the model, it will guess something useless.

    It builds the official dictionary first, so that it can provide a minimum
    probablity for each word as well as use the official tokenizer.
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
        """Example implementation of next word probability."""
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
    parser.set_defaults(model='projects.twitter.eval_ppl:WordFrequencyEntry')
    # try with --numthreads N to go fast
    opt = parser.parse_args()
    eval_ppl(opt)
    if opt['model'] == 'projects.twitter.eval_ppl:WordFrequencyEntry':
        print('This run just used the example filler model. To get better '
              'results, try implementing your own!')
