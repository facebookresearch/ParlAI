#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DST on Google The Schema-Guided Dialogue(SGD) Dataset implementation for ParlAI.
"""

import os
import json, random
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher, DialogTeacher
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, BleuMetric
import parlai.utils.logging as logging

from .utils.reformat import reformat_parlai

from .build import build


def _path(opt):
    # set up path to data (specific to each dataset)
    data_dir = os.path.join(opt["datapath"], "google_sgd_dst")
    data_path = os.path.join(data_dir, "data_reformat.json")

    # build the data if it does not exist
    build(opt)

    # reformat data for DST
    reformat_parlai(data_dir)

    return data_path, data_dir


class Google_SGD_DST_Teacher(FixedDialogTeacher):
    """
    MultiWOZ DST Teacher.
    """

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        opt["datafile"], jsons_path = _path(opt)
        self.id = "google_sgd_dst"
        # import pdb; pdb.set_trace()
        self.seed = opt.get("rand_seed", 0)
        # # # reading args
        self.decode_all = opt.get("decode_all", False)
        self.just_test = opt.get("just_test", False)
        self.val_reduced = opt.get("val_reduced", True)
        self.test_reduced = opt.get("test_reduced", True)
        self.use_prompts = opt.get("use_prompts", True)

        random.seed(self.seed)

        self._setup_data(opt["datafile"], jsons_path)
        self.reset()

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    @classmethod
    def add_cmdline_args(cls, argparser, partial_opt):
        agent = argparser.add_argument_group("Google SGD DST Teacher Args")
        agent.add_argument(
            "--just_test",
            type=bool,
            default=False,
            help="True if one would like to test agents with small amount of data (default: False).",
        )
        agent.add_argument(
            "--val_reduced",
            type=bool,
            default=False,
            help="True if one would like to validate on limited set: 1000 episodes",
        )
        agent.add_argument(
            "--test_reduced",
            type=bool,
            default=False,
            help="True if one would like to validate on limited set: 10000 episodes",
        )
        agent.add_argument(
            "--rand_seed",
            type=int,
            default=0,
            help="specify to set random seed (default: 0).",
        )
        agent.add_argument(
            "--use_prompts",
            type=bool,
            default=True,
            help="add natural text instructions for the DST task.",
        )
        return argparser

    def _setup_data(self, data_path, jsons_path):
        # print('loading: ' + data_path)

        # # # loading directly from test file or val file
        if self.datatype.startswith("test"):
            test_path = data_path.replace(".json", "_test.json")
            test_data = self._load_json(test_path)
            self.messages = list(test_data.values())
            if self.test_reduced:
                self.messages = self.messages[:10000]
        elif self.datatype.startswith("valid"):
            valid_path = data_path.replace(".json", "_dev.json")
            valid_data = self._load_json(valid_path)
            self.messages = list(valid_data.values())
            if self.val_reduced:
                # print(self.messages[0])
                self.messages = random.sample(list(valid_data.values()), k=1000)
                # self.messages = self.messages[:30]
        else:
            train_path = data_path.replace(".json", "_train.json")
            train_data = self._load_json(train_path)
            self.messages = list(train_data.values())
            random.shuffle(self.messages)

        # print("successfully loaded")

        if self.just_test:
            self.messages = self.messages[:10]

    def num_examples(self):
        # examples = 0
        # for data in self.messages:
        #     examples += len(data)
        # return examples

        return len(self.messages)

    def num_episodes(self):
        return len(self.messages)

    def format_context_and_label(self, context, label):
        """
        Transform task to include an instruction with custom labels that are all in natural text
        """

        templates = [
            (
                f"Conversation: {context} \n Question: What is the dialogue belief state of this conversation?",
                label,
            ),
            (f"Tell me the dialogue belief state of this dialogue: {context}", label),
            (
                f"List out the dialogue belief state of the following conversation in 'domain slot type slot value' format separated by commas: {context}",
                label,
            ),
            (
                f"Here's a conversation: {context} \n Question: What is it's dialogue belief state?",
                label,
            ),
            (
                f"Dialogue: {context} Question: What is it's dialogue belief state?",
                label,
            ),
            (
                f"Conversation: {context} Question: What is it's dialogue belief state? Don't provide any entities that were not given in the conversation.",
                label,
            ),
        ]

        # add templates that try to generate the user utterance based on the belief state
        # only do it for conversations that we have at least one system utterance
        # if "<system>" in context:
        #     splits = context.split("<user>")
        #     last_user_turn = splits[-1].strip()
        #     new_context = context[: context.index(splits[-2]) + len(splits[-2])]

        #     templates += [
        #         (
        #             f"Here's a conversation: {new_context}. Provide a response from the user that would lead to this belief state: {label}",
        #             last_user_turn,
        #         ),
        #         (
        #             f"Conversation: {new_context} Question: What's a user utterance that would lead to this belief state \"{label}\"?",
        #             last_user_turn,
        #         ),
        #         (
        #             f"Generate an utterance that would lead to this belief state \"{label}\" following this dialogue: {new_context}",
        #             last_user_turn,
        #         ),
        #         (
        #             f"Given that the dialogue belief state is \"{label}\" after a response from the user to this conversation: \"{new_context}\", what could be a valid response?",
        #             last_user_turn,
        #         ),
        #     ]

        template, label = random.choice(templates)

        return template, label

    def get(self, episode_idx, entry_idx=0):
        # log_idx = entry_idx
        entry = self.messages[episode_idx]["context"]
        label = self.messages[episode_idx]["slots_inf"]
        # entry = self.messages[episode_idx][entry_idx]['context']
        # episode_done = entry_idx == len(self.messages[episode_idx]) - 1
        if self.use_prompts:
            context, label = self.format_context_and_label(entry, label)
        else:
            context, label = entry, label
        episode_done = True
        action = {
            "id": self.id,
            "text": context,
            "episode_done": episode_done,
            "labels": [label],
            "turn_num": self.messages[episode_idx]["turn_num"],
            # 'labels': [self.messages[episode_idx][entry_idx]['slots_inf']],
        }
        return action

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom slot_type slot_val", ... ]
        """
        slots_list = []

        # # # split according to ","
        str_split = slots_string.split(",")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        slots_list = [slot.strip() for slot in str_split]

        return slots_list

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        resp = model_response.get("text")
        if not resp:
            return

        # # # extract ground truth from labels
        slots_truth = self._extract_slot_from_string(labels[0])

        # # # extract generated slots from model_response
        slots_pred = self._extract_slot_from_string(resp)

        self.metrics.add(
            "joint goal acc", AverageMetric(set(slots_truth) == set(slots_pred))
        )
        if set(slots_truth) != set(slots_pred):
            logging.info(f"slots_truth: {slots_truth}\nslots_pred: {slots_pred}")


class DefaultTeacher(Google_SGD_DST_Teacher):
    pass
