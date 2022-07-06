#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
The Ubuntu dataset used for dstc 7.
"""

from parlai.core.teachers import FixedDialogTeacher
from parlai.utils.io import PathManager
from .build import build

import json
import os
import random


class DSTC7Teacher(FixedDialogTeacher):
    """
    Teacher that corresponds to the default DSTC7 ubuntu track 1.

    The data hasn't been augmented by using the multi-turn utterances.
    """

    def __init__(self, opt, shared=None):
        self.split = 'train'
        if 'valid' in opt['datatype']:
            self.split = 'dev'
        if 'test' in opt['datatype']:
            self.split = 'test'
        build(opt)

        basedir = os.path.join(opt['datapath'], 'dstc7')
        filepath = os.path.join(
            basedir, 'ubuntu_%s_subtask_1%s.json' % (self.split, self.get_suffix())
        )
        if shared is not None:
            self.data = shared['data']
        else:
            with PathManager.open(filepath, 'r') as f:
                self.data = json.loads(f.read())

            # special case of test set
            if self.split == "test":
                id_to_res = {}
                with PathManager.open(
                    os.path.join(basedir, "ubuntu_responses_subtask_1.tsv"), 'r'
                ) as f:
                    for line in f:
                        splited = line[0:-1].split("\t")
                        id_ = splited[0]
                        id_res = splited[1]
                        res = splited[2]
                        id_to_res[id_] = [{"candidate-id": id_res, "utterance": res}]
                for sample in self.data:
                    sample["options-for-correct-answers"] = id_to_res[
                        str(sample["example-id"])
                    ]

        super().__init__(opt, shared)
        self.reset()

    def get_suffix(self):
        return ""

    def _setup_data(self, datatype):
        pass

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def get(self, episode_idx, entry_idx=0):
        rand = random.Random(episode_idx)
        episode = self.data[episode_idx]
        texts = []
        for m in episode["messages-so-far"]:
            texts.append(m["speaker"].replace("_", " ") + ": ")
            texts.append(m["utterance"] + "\n")
        text = "".join(texts)
        labels = [m["utterance"] for m in episode["options-for-correct-answers"]]
        candidates = [m["utterance"] for m in episode["options-for-next"]]
        if labels[0] not in candidates:
            candidates = labels + candidates
        rand.shuffle(candidates)
        label_key = "labels" if self.split == "train" else "eval_labels"
        return {
            "text": text,
            label_key: labels,
            "label_candidates": candidates,
            "episode_done": True,
        }

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class DSTC7TeacherAugmented(DSTC7Teacher):
    """
    Augmented Data.

    To mimic the way ParlAI generally handles dialogue datasets, the data associated
    with this teacher is presented in a format such that a single "episode" is split
    across multiple entries.

    I.e., suppose we have the following dialogue between speakers 1 and 2:
    utterances: [A, B, C, D, E],
    label: F

    The data in this file is split such that we have the following episodes:

    ep1:
        utterances: [A],
        label: B
    ep2:
        utterances [A, B, C]
        label: D
    ep3:
        utterances: [A, B, C, D, E],
        label: F
    """

    def get_suffix(self):
        if self.split != "train":
            return ""
        return "_augmented"


class DSTC7TeacherAugmentedSampled(DSTC7Teacher):
    """
    The dev and test set are the same, but the training set has been augmented using the
    other utterances.

    Moreover, only 16 candidates are used (including the right one)
    """

    def get_suffix(self):
        if self.split != "train":
            return ""
        return "_sampled"

    def get_nb_cands(self):
        return 16

    def get(self, episode_idx, entry_idx=0):
        sample = super().get(episode_idx, entry_idx)
        if self.split != 'train':
            return sample
        new_cands = [sample['labels'][0]]
        counter = 0
        while len(new_cands) < self.get_nb_cands():
            if sample['label_candidates'][counter] not in sample['labels']:
                new_cands.append(sample['label_candidates'][counter])
            counter += 1
        sample['label_candidates'] = new_cands
        return sample


class DefaultTeacher(DSTC7Teacher):
    pass
