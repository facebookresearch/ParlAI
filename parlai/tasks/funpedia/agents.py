#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDeprecatedDialogTeacher, FixedDialogTeacher
from parlai.utils.io import PathManager
from .build import build

import copy
import os

"""
Task for rephrasing sentences from Wikipedia conditioned on a persona.
"""


def _path(opt):
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    return os.path.join(
        opt['datapath'],
        'rephrase_sentences',
        'rephrase_sentences_' + datatype + '_0703.txt',
    )


def _choose_sentence_path(opt):
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    return os.path.join(
        opt['datapath'], 'rephrase_sentences', 'choose_sentence_' + datatype + '.txt'
    )


def _strip_reader(filename):
    """
    Reads a file, stripping line endings.
    """
    with PathManager.open(filename) as f:
        for line in f:
            yield line.rstrip()


# this is a standard itertools recipe, but not included by default.
# see https://docs.python.org/3/library/itertools.html#itertools-recipes
# (included in `pip install more-itertools`)
def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    from itertools import zip_longest

    return zip_longest(*args, fillvalue=fillvalue)


class FunpediaTeacher(FixedDialogTeacher):
    """
    Generic teacher which extracts each of the fields.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if shared:
            self.entries = shared['entries']
        else:
            self.entries = []
            self.datafile = _path(opt)
            self._setup_data()

        self.reset()

    def share(self):
        shared = super().share()
        shared['entries'] = self.entries
        return shared

    def num_episodes(self):
        # only one example per episode
        return self.num_examples()

    def num_examples(self):
        return len(self.entries)

    def _setup_data(self):
        title_prefix = '1 passage title: '
        title_prefix_len = len(title_prefix)
        persona_prefix = '2 personality: '
        persona_prefix_len = len(persona_prefix)
        text_prefix = '3 '
        text_prefix_len = len(text_prefix)

        data_reader = grouper(_strip_reader(self.datafile), 3, '')

        for title, persona, text in data_reader:
            if not title:
                break
            assert title.startswith(title_prefix)
            # strip 'passage title: ' from the title
            title = title[title_prefix_len:]

            assert persona.startswith(persona_prefix)
            # strip 'persona: ' from the persona
            persona = persona[persona_prefix_len:]

            assert text.startswith(text_prefix)
            text = text[text_prefix_len:]

            passage, label = text.split("\t")

            self.entries.append(
                {'title': title, 'label': label, 'passage': passage, 'persona': persona}
            )

    def get_text(self, entry):
        return '\n'.join([entry['title'], entry['persona'], entry['passage']])

    def _build_action(self, entry):
        return {
            'text': self.get_text(entry),
            'labels': [entry['label']],
            'reward': 0,
            'episode_done': True,
        }

    def get(self, episode_idx, entry_idx=0):
        assert entry_idx == 0
        return self._build_action(self.entries[episode_idx])


class NopersonaTeacher(FunpediaTeacher):
    """
    Strips persona out entirely.
    """

    def get_text(self, entry):
        return entry['title'] + "\n" + entry['passage']


class LmTeacher(FunpediaTeacher):
    """
    Modifies the data to drop the query entirely, creating a language modeling task.
    """

    def get_text(self, entry):
        return ''


class EchoTeacher(FunpediaTeacher):
    """
    Replaces answers with an echo of the passage.

    Useful for measuring how much a model learns to simply repeat what is said.
    """

    def _setup_data(self):
        super()._setup_data()
        for i in range(len(self.entries)):
            self.entries[i]['label'] = self.entries[i]['passage']


class SentencechooseTeacher(FbDeprecatedDialogTeacher):
    """
    Teacher for the sentence choosing task.

    Turkers were instructed to choose the 'most interesting' sentence from a paragraph.
    """

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _choose_sentence_path(opt)
        super().__init__(opt, shared)

    def next_example(self):
        action, epoch_done = super().next_example()
        action['label_candidates'] = list(action['label_candidates'])
        if '' in action['label_candidates']:
            action['label_candidates'].remove('')
        return action, epoch_done


class DefaultTeacher(FunpediaTeacher):
    pass
