#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from .build import build

import copy
import os
import json
import random


class WICTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt["datatype"]
        # build(opt)  # NOTE: the call to build here
        suffix = "train" if opt["datatype"].startswith("train") else "val"
        # whatever is placed into datafile will be passed as the argument to
        # setup_data in the next section.
        opt["datafile"] = os.path.join(opt["datapath"], "WiC", suffix + ".jsonl")
        self.id = "WiC"
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print("loading: " + path)
        with open(path, "r") as f:
            self.data = f.readlines()

        for data in self.data:
            json_ = json.loads(data)

            sentence1, sentence2 = json_["sentence1"], json_["sentence2"]
            word, label = json_["word"], json_["label"]

            templates = [
                (
                    f'Does the word "{word}" have the same meaning in these two sentences? Yes, No? {sentence1} {sentence2}',
                    "yes" if label else "no",
                ),
                (
                    f'Sentence A: {sentence1} Sentence B: {sentence2} "{word}" has a similar meaning in sentences A and B. True or False?',
                    "True" if label else "False",
                ),
                (
                    f'Decide whether the word "{word}" is used with the same meaning in the two following sentences. Answer by yes or no. {sentence1} {sentence2}',
                    "yes" if label else "no",
                ),
                (
                    f'{sentence1} {sentence2} Question: Is the word "{word}" used in the same sense in the two sentences above?',
                    "yes" if label else "no",
                ),
                (
                    f'Sentence 1: {sentence1} Sentence 2: {sentence2} Determine whether the word "{word}" is used in the same sense in both sentences. Yes or no?',
                    "yes" if label else "no",
                ),
                (
                    f"Determine if the word '{{word}}' is used in the same way in the two sentences below. {sentence1} {sentence2}",
                    "yes" if label else "no",
                ),
                (
                    f"{sentence1} {sentence2} Question: Is the word '{word}' used in the same sense in the two sentences above? Yes, No?",
                    "yes" if label else "no",
                ),
                (
                    f'The word "{word}" has multiple meanings. Does it have the same meaning in sentences 1 and 2? Yes or no? Sentence 1: {sentence1} Sentence 2: {sentence2}',
                    "yes" if label else "no",
                ),
                (
                    f"{sentence1} {sentence2} Similar sense of {word}?",
                    "yes" if label else "no",
                ),
            ]

            # import pdb; pdb.set_trace()
            new_episode = True
            context, answer = random.choice(templates)
            yield (context.lower(), [answer.lower()], None, None), new_episode


class DefaultTeacher(WICTeacher):
    pass
