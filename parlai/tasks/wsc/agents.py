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
        opt["datafile"] = os.path.join(opt["datapath"], "WSC", suffix + ".jsonl")
        self.id = "WSC"
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print("loading: " + path)
        with open(path, "r") as f:
            self.data = f.readlines()

        for data in self.data:
            json_ = json.loads(data)

            target = json_["target"]

            span1_text, span2_text = target["span1_text"], target["span2_text"]
            span1_index, span2_index = target["span1_index"], target["span2_index"]
            text, label = json_["text"], json_["label"]

            templates = [
                (
                    f'{text} In the previous sentence, does the pronoun "{span2_text}" refer to {span1_text}? Yes or no?',
                    "yes" if label else "no",
                ),
                (
                    f'{text} Here, by "{span2_text}" they mean "{span1_text}". Yes or no?',
                    "yes" if label else "no",
                ),
                (
                    f"{text} In other words, {' '.join(text.split(' ')[span2_index:]).replace(span2_text, span1_text)} True or false?",
                    "True" if label else "False",
                ),
                (
                    f"{text} I think they mean \"{' '.join(text.split(' ')[span2_index:]).replace(span2_text, span1_text)}\" Yes or no?",
                    "yes" if label else "no",
                ),
                (
                    f'{text} Here, does "{span2_text}" stand for {span1_text}? Yes or no?',
                    "yes" if label else "no",
                ),
                (
                    f'Passage: {text} Question: In the passage above, does the pronoun "{span2_text}" refer to {span1_text}? Answer:',
                    "yes" if label else "no",
                ),
                (
                    f'{text} In the previous sentence, can the pronoun "{span2_text}" be replaced with "{span1_text}"? Yes or no?',
                    "yes" if label else "no",
                ),
                (
                    f'{text} In the passage above, the pronoun "{span2_text}" refers to {span1_text}. True or false?',
                    "True" if label else "False",
                ),
            ]

            # import pdb; pdb.set_trace()

            context, answer = random.choice(templates)
            yield (context.lower(), [answer.lower()]), True


class DefaultTeacher(WICTeacher):
    pass
