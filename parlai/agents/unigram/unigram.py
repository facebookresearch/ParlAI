#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import pickle
import re
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from itertools import islice


class UnigramAgent(Agent):
    """
    Baseline agent which always emits the N most common unigrams.
    """

    @classmethod
    def add_cmdline_args(cls, parser):
        parser.add_argument(
            '--num-words', type=int, default=10,
            help='Number of unigrams to output.'
        )
        cls.dictionary_class().add_cmdline_args(parser)

    @classmethod
    def dictionary_class(cls):
        return DictionaryAgent

    def __init__(self, opt, shared=None):
        self.id = 'UnigramAgent'
        self.unigram_cache = None
        self.opt = opt
        self.num_words = opt['num_words']

        if shared is not None:
            self.dict = shared['dict']
        else:
            self.dict = self.dictionary_class()(opt)

    def share(self):
        return {'dict': self.dict}

    def observe(self, obs):
        pass

    def is_valid_word(self, word):
        return (
            not word.startswith('__') and
            word != '\n' and
            not re.match('[^\w]', word)
        )

    def get_prediction(self):
        if self.unigram_cache is None:
            most_common = sorted(
                self.dict.freq.items(), key=lambda x: x[1], reverse=True
            )
            most_common = ((u, v) for u, v in most_common if self.is_valid_word(u))
            most_common = islice(most_common, self.num_words)
            most_common = (u for u, v in most_common)
            self.unigram_cache = ' '.join(list(most_common))
        return self.unigram_cache

    def act(self):
        return {
            'id': self.getID(),
            'text': self.get_prediction(),
        }

    def save(self, path=None):
        if not path:
            return

        with open(path, 'w') as f:
            f.write(self.get_prediction() + '\n')

        with open(path + '.opt', 'wb') as f:
            pickle.dump(self.opt, f)

    def load(self, path):
        # don't actually do anything on load
        pass
