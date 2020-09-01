#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline model which always emits the N most common non-punctuation unigrams. Typically
this is mostly stopwords. This model is a poor conversationalist, but may get reasonable
F1.

UnigramAgent has one option, --num-words, which controls the unigrams
outputted.

This also makes a nice reference for a simple, minimalist agent.
"""

import json
import re
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from itertools import islice
from parlai.utils.io import PathManager


class UnigramAgent(Agent):
    @classmethod
    def add_cmdline_args(cls, parser):
        """
        Adds command line arguments.
        """
        parser.add_argument(
            '--num-words', type=int, default=10, help='Number of unigrams to output.'
        )
        cls.dictionary_class().add_cmdline_args(parser)

    @classmethod
    def dictionary_class(cls):
        """
        Returns the DictionaryAgent used for tokenization.
        """
        return DictionaryAgent

    def __init__(self, opt, shared=None):
        """
        Construct a UnigramAgent.

        :param opt: parlai options
        :param shared: Used to duplicate the model for batching/hogwild.
        """
        self.id = 'UnigramAgent'
        self.unigram_cache = None
        self.opt = opt
        self.num_words = opt['num_words']

        if shared is not None:
            self.dict = shared['dict']
        else:
            self.dict = self.dictionary_class()(opt)

    def share(self):
        """
        Basic sharing function.
        """
        return {'dict': self.dict}

    def observe(self, obs):
        """
        Stub observe method.
        """
        self.observation = obs

    def is_valid_word(self, word):
        """
        Marks whether a string may be included in the unigram list.

        Used to filter punctuation and special tokens.
        """
        return (
            not word.startswith('__') and word != '\n' and not re.match(r'[^\w]', word)
        )

    def get_prediction(self):
        """
        Core algorithm, which gathers the most common unigrams into a string.
        """
        # we always make the same prediction, so cache it for speed
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
        """
        Stub act, which always makes the same prediction.
        """
        return {'id': self.getID(), 'text': self.get_prediction()}

    def save(self, path=None):
        """
        Stub save which dumps options.

        Necessary for evaluation scripts to load the model.
        """
        if not path:
            return

        with PathManager.open(path, 'w') as f:
            f.write(self.get_prediction() + '\n')

        with PathManager.open(path + '.opt', 'w') as f:
            json.dump(self.opt, f)

    def load(self, path):
        """
        Stub load which ignores the model on disk, as UnigramAgent depends on the
        dictionary, which is saved elsewhere.
        """
        # we rely on the dict, so we don't actually need to load anything
        pass
