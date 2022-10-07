#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DST on Augmented Multiwoz2.3 from LAUG 
"""
import sys, os
import json, random

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from .build import build

# from .utils.reformat import reformat_parlai
import parlai.utils.logging as logging
import pandas as pd
from collections import defaultdict


def load_msr_corpus(fn: str):
    """
    custom loading function for msr paraphrase corpus because read_csv complains for unidentified reasons for certain lines
    """
    # doesn't get the full list for some reason: the following ignores some lines
    # msr_df = pd.read_csv("/Users/justincho/tod/paraphrase_identification/dataset/msr-paraphrase-corpus/msr_paraphrase_test.txt", sep="\t", names=column_names, skiprows=[0])

    with open(fn, "r") as f:
        msr = f.read().splitlines()
    # column_names = msr[0].split("\t")

    dd = defaultdict(list)
    for row in msr[1:]:
        split = row.split("\t")
        dd["label"].append(split[0])
        dd["id1"].append(split[1])
        dd["id2"].append(split[2])
        dd["str1"].append(split[3])
        dd["str2"].append(split[4])

    return pd.DataFrame(dd)


def load_qqp(fn: str):
    """
    load a subset of the qqp questions corpus
    """

    qqp = pd.read_csv(fn)
    qqp_filtered = qqp[
        (qqp["question1"].str.split().str.len() > 4)
        & (qqp["question2"].str.split().str.len() > 4)
    ]

    return qqp_filtered


def normalize_df(df: pd.DataFrame):
    """
    Utility function to make dataframes have the same column names for those that matter (two columns that contain sentences and the label)
    """

    if "question1" in df.columns:
        df["str1"] = df["question1"]
        df["str2"] = df["question2"]
        df.drop(["question1", "question2"], axis=1, inplace=True)

    if "is_duplicate" in df.columns:
        df["label"] = df["is_duplicate"].apply(lambda x: int(x))
        df.drop(["is_duplicate"], axis=1, inplace=True)

    return df


class GenerativeParaphraseClassificationTeacher(FixedDialogTeacher):
    """
    Generative Paraphrase Classification
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = "generative paraphrase classification"

        # # # reading args
        self.just_test = opt.get("just_test", False)
        self.seed = opt.get("rand_seed", 0)
        self.data_version = opt.get("data_version", "all")
        self.val_reduced = opt.get("val_reduced", False)
        self.reduce_train_factor = opt.get("reduce_train_factor", 1)
        self.flag_compute = 0
        # # # set random seeds
        random.seed(self.seed)

        opt["datafile"], data_dir = self._path(opt)
        self._setup_data(opt["datafile"], data_dir)

        self.reset()

    @classmethod
    # def add_cmdline_args(cls, argparser):
    def add_cmdline_args(cls, argparser, partial_opt):
        agent = argparser.add_argument_group("MultiWozDST Teacher Args")
        agent.add_argument(
            "-dv",
            "--data_version",
            type=str,
            default="all",
            help="one of ['all', 'msr', qq]",
        )
        agent.add_argument(
            "--just_test",
            type="bool",
            default=False,
            help="True if one would like to test agents with small amount of data (default: False).",
        )
        agent.add_argument(
            "--rand_seed",
            type=int,
            default=0,
            help="specify to set random seed (default: 0).",
        )

        agent.add_argument(
            "--val_reduced",
            type="bool",
            default=False,
            help="use smaller evaluation set.",
        )

        return argparser

    def _path(self, opt):
        # set up path to data (specific to each dataset)
        data_dir = os.path.join(opt["datapath"], "paraphrase")
        # data_dir = os.path.join('/checkpoint/kunqian/multiwoz/data/MultiWOZ_2.1/')

        data_path = ""
        # build the data if it does not exist (not implemented yet)
        # build(opt)

        return data_path, data_dir

    def _setup_data(self, data_path, data_dir):

        msr_train_path = os.path.join(
            data_dir, "msr-paraphrase-corpus", "msr_paraphrase_train.txt"
        )
        msr_test_path = os.path.join(
            data_dir, "msr-paraphrase-corpus", "msr_paraphrase_test.txt"
        )
        qqp_path = os.path.join(data_dir, "quora-question-pairs", "train.csv")

        # unify column names
        qqp_df = normalize_df(load_qqp(qqp_path))
        msr_train_df = normalize_df(load_msr_corpus(msr_train_path))
        msr_test_df = normalize_df(load_msr_corpus(msr_test_path))

        qqp_df["dial_id"] = [f"qqp-{idx}" for idx in range(len(qqp_df))]
        msr_train_df["dial_id"] = [
            f"msr-train-{idx}" for idx in range(len(msr_train_df))
        ]
        msr_test_df["dial_id"] = [f"msr-test-{idx}" for idx in range(len(msr_test_df))]

        msr_train_data = msr_train_df.to_dict(orient="records")
        msr_test_data = msr_test_df.to_dict(orient="records")
        qqp_data = qqp_df.to_dict(orient="records")

        # not necessary?
        # random.shuffle(msr_data)
        # random.shuffle(qqp_data)
        # import pdb; pdb.set_trace()

        # do a 8:1:1 split
        qqp_split_idx = int(len(qqp_data) * 0.1)
        msr_split_idx = int(len(msr_test_data) * 0.5)
        if self.data_version == "all":

            test_data = qqp_data[-1 * qqp_split_idx :] + msr_test_data[msr_split_idx:]
            valid_data = (
                qqp_data[-2 * qqp_split_idx : -1 * qqp_split_idx]
                + msr_test_data[:msr_split_idx]
            )
            train_data = qqp_data[: -2 * qqp_split_idx] + msr_train_data

        elif self.data_version == "qqp":

            test_data = qqp_data[-1 * qqp_split_idx :]
            valid_data = qqp_data[-2 * qqp_split_idx : -1 * qqp_split_idx]
            train_data = qqp_data[: -2 * qqp_split_idx]

        elif self.data_version == "msr":

            test_data = msr_test_data[msr_split_idx:]
            valid_data = msr_test_data[:msr_split_idx]
            train_data = msr_train_data

        if self.datatype.startswith("train"):
            self.messages = train_data
        elif self.datatype.startswith("valid"):
            self.messages = valid_data
        elif self.datatype.startswith("test"):
            self.messages = test_data

        if self.just_test:
            self.messages = self.messages[:10]

        # shuffle for training only: important for invariance scoring
        if self.datatype.startswith("train"):
            random.shuffle(self.messages)

    def num_examples(self):
        # each turn be seen as a individual dialog
        return len(self.messages)

    def num_episodes(self):
        return len(self.messages)

    def format_context_and_label(self, str1, str2, label):
        """
        Transform task to include an instruction with custom labels that are all in natural text
        """

        qqp_templates = [
            (
                f'I\'m an administrator on the website Quora. There are two posts, one that asks "{str1}" and another that asks "{str2}". I can merge questions if they are asking the same thing. Can I merge these two questions?',
                "Yes" if label else "No",
            ),
            (
                f'{str1} {str2} Pick one: These questions are "duplicates" or "not duplicates"',
                "Duplicates" if label else "Not duplicates",
            ),
            (
                f'Are the questions "{str1}" and "{str2}" asking the same thing?',
                "Yes" if label else "No",
            ),
            (
                f'Can an answer to "{str1}" also be used to answer "{str2}"?',
                "Yes" if label else "No",
            ),
            (
                f"Question 1: {str1} Question 2: {str2} Do these two questions convey the same meaning? Yes or no?",
                "Yes" if label else "No",
            ),
            (
                f'I received the questions "{str1}" and "{str2}" Are they duplicates?',
                "Yes" if label else "No",
            ),
            (
                f'Give me another question that asks for the same thing as "{str1}".'
                if label
                else f'Give me a question that seems similar but actually asks for something that is different from "{str1}".',
                str2,
            ),
        ]

        if label:
            qqp_templates += [
                (f"Paraphrase this question: {str1}", str2),
                (
                    f'Rephrase this question "{str1}" without changing what it is asking for.',
                    str2,
                ),
            ]

        msr_templates = [
            (
                f"I want to know whether the following two sentences mean the same thing. {str1} {str2} Do they?",
                "Yes" if label else "No",
            ),
            (
                f"Does the sentence {str1} parpahrase (that is, mean the same thing as) this sentence? {str2}",
                "Yes" if label else "No",
            ),
            (
                f'Are the following two sentences "equivalent" or "not equivalent"? {str1} {str2}',
                "equivalent" if label else "not equivalent",
            ),
            (
                f"Can I replace the sentence {str1} with the sentence {str2} and have it mean the same thing?",
                "Yes" if label else "No",
            ),
            (
                f"Do the following sentences mean the same thing? {str1} {str2}",
                "Yes" if label else "No",
            ),
        ]

        if label:
            msr_templates += [
                (f"Paraphrase the following sentence: {str1}", str2),
                (
                    f"Generate a sentence that means the same thing as this one: {str1}",
                    str2,
                ),
            ]

        if str1[-1] == "?":
            context, label = random.choice(qqp_templates)
        else:
            context, label = random.choice(msr_templates)

        return context.lower(), label.lower()

    def get(self, episode_idx, entry_idx=0):
        # log_idx = entry_idx
        context, label = self.format_context_and_label(
            self.messages[episode_idx]["str1"],
            self.messages[episode_idx]["str2"],
            self.messages[episode_idx]["label"],
        )
        episode_done = True
        action = {
            "id": self.id,
            "text": context.lower(),
            "episode_done": episode_done,
            "labels": [label],
            "dial_id": self.messages[episode_idx]["dial_id"],
            "turn_num": 0,
        }

        return action


class DefaultTeacher(GenerativeParaphraseClassificationTeacher):
    """
    Default teacher.
    """

    pass
