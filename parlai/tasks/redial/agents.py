#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from parlai.utils.data import DatatypeHelper
from .build import build
import os
import json
import re
import csv
from typing import Optional


def _path(opt):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    data_path = os.path.join(opt['datapath'], 'redial')
    return data_path


# Turns title from format "Title (Year)" to "Title" or leaves as is if no (Year)
def remove_year_from_title(title):
    matches = re.finditer(r"\s\(", title)
    indices = [m.start(0) for m in matches]
    if indices:
        title_end = indices[-1]
        return title[:title_end]
    else:
        return title


def replace_movie_ids(id_string, id_map):
    pattern = r'@\d+'
    return re.sub(pattern, lambda s: id_map[s.group()], id_string)


class ReDialTeacher(DialogTeacher):
    """
    ReDial Teacher. By default, this learns the suggestor
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        group = parser.add_argument_group('ReDial dataset arguments')
        group.add_argument(
            '--redial-include-confounder-options',
            type=bool,
            default=False,
            help="Add other movies as suggestions for first turn",
        )

    def __init__(self, opt, shared=None):
        opt['datafile'] = DatatypeHelper.fold(opt['datatype'])
        self.data_path = _path(opt)
        self.title_id_map = {}
        self.get_title_dict(self.data_path)
        self.id = 'redial'
        super().__init__(opt, shared)

    def get_title_dict(self, path):
        csv_path = os.path.join(path, 'movies_with_mentions.csv')
        with PathManager.open(csv_path, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.title_id_map['@' + row[0]] = remove_year_from_title(row[1])

    def setup_data(self, fold):
        train_path = os.path.join(self.data_path, 'train_data.jsonl')
        test_path = os.path.join(self.data_path, 'test_data.jsonl')
        # The test data has 1341 episodes. Making valid this size gives
        # about 80/10/10 train/test/valid split
        test_set_episodes = 1341
        if fold.startswith('test'):
            unmerged_episodes = self.get_data_from_file(test_path)
        elif fold.startswith('valid'):
            unmerged_episodes = self.get_data_from_file(train_path)
            unmerged_episodes = unmerged_episodes[:test_set_episodes]
        else:
            unmerged_episodes = self.get_data_from_file(train_path)
            unmerged_episodes = unmerged_episodes[test_set_episodes:]

        # some speakers speak multiple times in a row.
        for unmerged_episode in unmerged_episodes:
            curr_text = []
            train_text = []
            first = True
            seeker_id = unmerged_episode["initiatorWorkerId"]
            recommmender_id = unmerged_episode["respondentWorkerId"]
            prev_speaker = None
            curr_speaker = None
            for message in unmerged_episode['messages']:
                # Multiple turns might come from the same speaker; need to
                # merge these.
                curr_speaker = message['senderWorkerId']
                if prev_speaker is None and curr_speaker != seeker_id:
                    continue  # Skip turns were recommender starts
                text = replace_movie_ids(message['text'], self.title_id_map)
                if curr_speaker == prev_speaker:
                    curr_text.append(text)
                else:
                    if prev_speaker == seeker_id:
                        train_text = curr_text
                    elif prev_speaker == recommmender_id:
                        yield {
                            "id": self.id,
                            "movieMentions": unmerged_episode["movieMentions"],
                            "respondentQuestions": unmerged_episode[
                                "respondentQuestions"
                            ],
                            "initiatorQuestions": unmerged_episode[
                                "initiatorQuestions"
                            ],
                            "text": " ".join(train_text),
                            "label": " ".join(curr_text),
                        }, first
                        first = False
                    curr_text = []
                    curr_text.append(text)
                    prev_speaker = curr_speaker
            if curr_speaker == recommmender_id:  # fetch last turn
                yield {
                    "id": self.id,
                    "movieMentions": unmerged_episode["movieMentions"],
                    "respondentQuestions": unmerged_episode["respondentQuestions"],
                    "initiatorQuestions": unmerged_episode["initiatorQuestions"],
                    "text": " ".join(train_text),
                    "label": " ".join(curr_text),
                }, first

    def get_data_from_file(self, filepath):
        data = []
        with PathManager.open(filepath) as f:
            for line in f:
                data.append(json.loads(line))
        return data


class DefaultTeacher(ReDialTeacher):
    pass
