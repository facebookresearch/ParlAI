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
import pandas as pd


class WICTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt["datatype"]
        # build(opt)  # NOTE: the call to build here
        suffix = "train" if opt["datatype"].startswith("train") else "dev"
        # whatever is placed into datafile will be passed as the argument to
        # setup_data in the next section.
        opt["datafile"] = os.path.join(opt["datapath"], "WNLI", suffix + ".tsv")
        self.id = "WNLI"
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print("loading: " + path)
        data_df = pd.read_csv(path, delimiter="\t")

        for idx, row in data_df.iterrows():

            sentence1, sentence2, label = (
                row["sentence1"],
                row["sentence2"],
                row["label"],
            )

            templates = [
                (
                    f'Is it correct to infer "{sentence2}" from "{sentence1}"? Yes or no?',
                    "yes" if label else "no",
                ),
                (
                    f'Can we say that "{sentence2}" from "{sentence1}"? Yes or no?',
                    "yes" if label else "no",
                ),
                (
                    f'With the sentence "{sentence1}", is it correct to say that "{sentence2}"? True or False?',
                    "True" if label else "False",
                ),
                (f"{sentence1}. Does this mean {sentence2}?", "yes" if label else "no"),
                (f'What can we infer from this sentence: "{sentence1}"?', sentence2),
            ]

            # import pdb; pdb.set_trace()

            context, answer = random.choice(templates)
            yield (context.lower(), [answer.lower()]), True


class DefaultTeacher(WICTeacher):
    pass
