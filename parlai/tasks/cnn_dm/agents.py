#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.


from parlai.core.teachers import DialogTeacher
from .build import build
import os
import unicodedata


def _fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    dm_single_close_quote = u'\u2019'
    dm_double_close_quote = u'\u201d'
    END_TOKENS = [
        '.', '!', '?', '...', "'", "`", '"',
        dm_single_close_quote, dm_double_close_quote, ")"
    ]  # acceptable ways to end a sentence
    if "@highlight" in line or line == "" or line[-1] in END_TOKENS:
        return line
    return line + "."


class CNNDMTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.dt = opt.get('datatype', 'train').split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'cnn_dm'
        self.datapath = os.path.join(opt['datapath'], 'CNN_DM')

        opt['datafile'] = self._path(opt)

        super().__init__(opt, shared)

    def _path(self, opt):
        build(opt)
        dt = opt['datatype'].split(':')[0]
        return os.path.join(self.datapath, dt + '.txt')

    def setup_data(self, input_path):
        self.question = 'What is the summary?'
        new_episode = True
        num_missing = 0
        num_added = 0

        print('loading: ' + input_path)

        with open(input_path) as stories_file:
            for story in stories_file:
                try:
                    story_file = open(os.path.join(self.datapath, story.strip()))
                except EnvironmentError:
                    num_missing += 1
                    continue

                num_added += 1
                article, highlights = [], []
                is_highlight = False
                for line in story_file:
                    line = _fix_missing_period(line.strip())
                    if line == "":
                        continue
                    if line.startswith("@highlight"):
                        is_highlight = True
                        continue
                    if is_highlight:
                        highlights.append(line)
                    else:
                        article.append(line)
                text = (
                    unicodedata.normalize('NFKC', ' '.join(article)) +
                    '\n' + self.question
                )
                label = [unicodedata.normalize('NFKC', ' '.join(highlights))]
                yield((text, label, None, None), new_episode)

        print("{} stories added, {} stories missing.".format(num_added, num_missing))


class DefaultTeacher(CNNDMTeacher):
    pass
