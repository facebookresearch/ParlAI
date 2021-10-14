#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utility functions and classes for detecting offensive language.
"""

from parlai.agents.transformer.transformer import TransformerClassifierAgent
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.tasks.dialogue_safety.agents import OK_CLASS, NOT_OK_CLASS
from parlai.utils.typing import TShared
from parlai.utils.io import PathManager
import parlai.utils.logging as logging
import os


class OffensiveLanguageClassifier:
    """
    Load model trained to detect offensive language in the context of single- turn
    dialogue utterances.

    This model was trained to be robust to adversarial examples created by humans. See
    <http://parl.ai/projects/dialogue_safety/> for more information.
    """

    def __init__(
        self,
        shared: TShared = None,
        custom_model_file='zoo:dialogue_safety/single_turn/model',
    ):
        if not shared:
            self.model = self._create_safety_model(custom_model_file)
        else:
            self.model = create_agent_from_shared(shared['model'])
        self.classes = {OK_CLASS: False, NOT_OK_CLASS: True}

    def share(self):
        shared = {'model': self.model.share()}
        return shared

    def _create_safety_model(self, custom_model_file):
        from parlai.core.params import ParlaiParser

        parser = ParlaiParser(False, False)
        TransformerClassifierAgent.add_cmdline_args(parser, partial_opt=None)
        parser.set_params(
            model='transformer/classifier',
            model_file=custom_model_file,
            print_scores=True,
            data_parallel=False,
        )
        safety_opt = parser.parse_args([])
        return create_agent(safety_opt, requireModelExists=True)

    def contains_offensive_language(self, text):
        """
        Returns the probability that a message is safe according to the classifier.
        """
        if not text:
            return False, 1.0
        act = {'text': text, 'episode_done': True}
        self.model.observe(act)
        response = self.model.act()['text']
        pred_class, prob = [x.split(': ')[-1] for x in response.split('\n')]
        pred_not_ok = self.classes[pred_class]  # check whether classified as NOT OK
        prob = float(prob)  # cast string to float

        return pred_not_ok, prob

    def __contains__(self, key):
        """
        A simple way of checking whether the model classifies an utterance as offensive.

        Returns True if the input phrase is offensive.
        """
        pred_not_ok, prob = self.contains_offensive_language(key)
        return pred_not_ok


class OffensiveStringMatcher:
    """
    Detects offensive language using a list of offensive language and phrases from
    https://github.com/LDNOOBW.
    """

    def __init__(self, datapath: str = None):
        """
        Get data from external sources and build data representation.

        If datapath ends in '.txt' it is assumed a custom model file is already given.
        """
        import parlai.core.build_data as build_data
        from parlai.core.dict import DictionaryAgent

        self.tokenize = DictionaryAgent.split_tokenize

        def _path():
            # Build the data if it doesn't exist.
            build()
            return os.path.join(
                self.datapath, 'OffensiveLanguage', 'OffensiveLanguage.txt'
            )

        def build():
            version = 'v1.0'
            dpath = os.path.join(self.datapath, 'OffensiveLanguage')
            if not build_data.built(dpath, version):
                logging.info(f'building data: {dpath}')
                if build_data.built(dpath):
                    # An older version exists, so remove these outdated files.
                    build_data.remove_dir(dpath)
                build_data.make_dir(dpath)

                # Download the data.
                fname = 'OffensiveLanguage.txt'
                url = 'http://parl.ai/downloads/offensive_language/' + fname
                build_data.download(url, dpath, fname)

                # Mark the data as built.
                build_data.mark_done(dpath, version)

        if datapath is not None and datapath.endswith('.txt'):
            # Load custom file.
            self.datafile = datapath
        else:
            # Build data from zoo, and place in given datapath.
            if datapath is None:
                # Build data from zoo.
                from parlai.core.params import ParlaiParser

                parser = ParlaiParser(False, False)
                self.datapath = parser.parse_args([])['datapath']
            else:
                self.datapath = datapath
            self.datafile = _path()

        # store a token trie: e.g.
        # {'2': {'girls': {'1': {'cup': {'__END__': True}}}}
        self.END = '__END__'
        self.max_len = 1
        self.offensive_trie = {}
        self.word_prefixes = [
            'de',
            'de-',
            'dis',
            'dis-',
            'ex',
            'ex-',
            'mis',
            'mis-',
            'pre',
            'pre-',
            'non',
            'non-',
            'semi',
            'semi-',
            'sub',
            'sub-',
            'un',
            'un-',
        ]
        self.word_suffixes = [
            'a',
            'able',
            'as',
            'dom',
            'ed',
            'er',
            'ers',
            'ery',
            'es',
            'est',
            'ful',
            'fy',
            'ies',
            'ify',
            'in',
            'ing',
            'ish',
            'less',
            'ly',
            's',
            'y',
        ]
        self.allow_list = [
            'butter',
            'buttery',
            'spicy',
            'spiced',
            'spices',
            'spicier',
            'spicing',
            'twinkies',
        ]

        with PathManager.open(self.datafile, 'r') as f:
            for p in f.read().splitlines():
                mod_ps = [p]
                mod_ps += [pref + p for pref in self.word_prefixes]
                mod_ps += [p + suff for suff in self.word_suffixes]
                for mod_p in mod_ps:
                    if mod_p not in self.allow_list:
                        self.add_phrase(mod_p)

    def add_phrase(self, phrase):
        """
        Add a single phrase to the filter.
        """
        toks = self.tokenize(phrase)
        curr = self.offensive_trie
        for t in toks:
            if t not in curr:
                curr[t] = {}
            curr = curr[t]
        curr[self.END] = True
        self.max_len = max(self.max_len, len(toks))

    def add_words(self, phrase_list):
        """
        Add list of custom phrases to the filter.
        """
        for phrase in phrase_list:
            self.add_phrase(phrase)

    def _check_sequence(self, toks, idx, node):
        """
        Check if words from the sequence are in the trie.

        This checks phrases made from toks[i], toks[i:i+2] ... toks[i:i + self.max_len]
        """
        right = min(idx + self.max_len, len(toks))
        for i in range(idx, right):
            if toks[i] in node:
                node = node[toks[i]]
                if self.END in node:
                    return ' '.join(toks[j] for j in range(idx, i + 1))
            else:
                break
        return False

    def contains_offensive_language(self, text):
        """
        Determine if text contains any offensive words in the filter.
        """
        if not text:
            return None
        if type(text) is str:
            toks = self.tokenize(text.lower())
        elif type(text) is list or type(text) is tuple:
            toks = text

        for i in range(len(toks)):
            res = self._check_sequence(toks, i, self.offensive_trie)
            if res:
                return res

        return None

    def find_all_offensive_language(self, text):
        """
        Find all offensive words from text in the filter.
        """
        if type(text) is str:
            toks = self.tokenize(text.lower())
        elif type(text) is list or type(text) is tuple:
            toks = text

        all_offenses = []
        for i in range(len(toks)):
            res = self._check_sequence(toks, i, self.offensive_trie)
            if res:
                all_offenses.append(res)
        return all_offenses

    def __contains__(self, key):
        """
        Determine if text contains any offensive words in the filter.
        """
        return self.contains_offensive_language(key)
