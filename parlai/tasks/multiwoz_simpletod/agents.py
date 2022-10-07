#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DST on Multiwoz2.1 Dataset implementation for ParlAI.
"""
import sys, os
import json, random

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, SumMetric
from .build import build
from .utils.trade_proc import trade_process
from .utils.reformat import reformat_parlai
from .utils.data_aug import data_aug, data_scr
from .utils.prompts import format_context_and_label
import parlai.utils.logging as logging


class MultiWozDSTTeacher(FixedDialogTeacher):
    """
    MultiWOZ DST Teacher.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = "multiwoz_dst"

        # # # reading args
        self.decode_all = opt.get("decode_all", False)
        self.just_test = opt.get("just_test", False)
        self.seed = opt.get("rand_seed", 0)
        self.data_aug = opt.get("data_aug", False)
        self.create_aug_data = opt.get("create_aug_data", False)
        self.data_scr = opt.get("data_scr", False)
        self.create_scr_data = opt.get("create_scr_data", False)
        self.version = opt.get("version", "2.3")
        self.val_reduced = opt.get("val_reduced", True)
        self.test_reduced = opt.get("test_reduced", False)
        self.few_shot = opt.get("few_shot", False)
        self.use_prompts = opt.get("use_prompts", True)
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
            "-dall",
            "--decode_all",
            type="bool",
            default=False,
            help="True if one would like to decode dst for all samples in training data, probably for \
            training a correction model (default: False).",
        )
        agent.add_argument(
            "--just_test",
            type="bool",
            default=False,
            help="True if one would like to test agents with small amount of data (default: False).",
        )
        agent.add_argument(
            "--reduce_train_factor",
            type=int,
            default=1,
            help="Factor to use in shrinking the training dataset size",
        )
        agent.add_argument(
            "-fs",
            "--few_shot",
            type=bool,
            default=False,
            help="Whether to simulate few shot setting by limiting trainig to 50 conversaions per domain",
        )
        agent.add_argument(
            "--rand_seed",
            type=int,
            default=0,
            help="specify to set random seed (default: 0).",
        )
        agent.add_argument(
            "--data_aug",
            type="bool",
            default=False,
            help="True if using augmented training (default: False).",
        )
        agent.add_argument(
            "--create_aug_data",
            type="bool",
            default=False,
            help="True if create augmented training data, used only during display_data(default: False).",
        )
        agent.add_argument(
            "--data_scr",
            type="bool",
            default=False,
            help="True if using scrambled training (default: False).",
        )
        agent.add_argument(
            "--create_scr_data",
            type="bool",
            default=False,
            help="True if create scrambled training data, used only during display_data(default: False).",
        )

        agent.add_argument(
            "--val_reduced",
            type="bool",
            default=False,
            help="use smaller evaluation set.",
        )
        agent.add_argument(
            "--test_reduced", type="bool", default=False, help="use smaller test set."
        )

        agent.add_argument(
            "--version",
            type=str,
            default="2.3",
            help="specify to use multiwoz 2.1, 2.2, 2.3 or laug (default: 2.3).",
        )
        agent.add_argument(
            "-up",
            "--use_prompts",
            type=bool,
            default=True,
            help="add natural text instructions for the DST task.",
        )
        return argparser

    def _load_txt(self, file_path):
        with open(file_path) as df:
            data = df.read().lower().split("\n")
            data.remove("")
        return data

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def _path(self, opt):
        # set up path to data (specific to each dataset)
        if str(self.version) not in ["2.1", "2.2", "2.3"]:
            # if not set to any one of the versions above, set default to 2.3
            # this is a hack for multitasking.
            self.version = "2.3"

        data_dir = os.path.join(
            opt["datapath"], "multiwoz_dst", "MULTIWOZ" + str(self.version)
        )
        if self.version == "laug":
            # no real use for this unless we want to see what results we get with faulty labels and context from the LAUG data
            data_path = os.path.join(
                opt["datapath"], "laug_dst", "orig", "data_reformat.json"
            )
        else:
            data_path = os.path.join(data_dir, f"data_reformat.json")

        # build the data if it does not exist
        build(opt)

        # process the data with TRADE's code, if it does not exist
        if (
            not os.path.exists(os.path.join(data_dir, "dials_trade.json"))
            and self.version == "2.1"
        ):
            trade_process(data_dir)

        # reformat data for DST
        if not os.path.exists(data_path) and self.version != "laug":
            reformat_parlai(data_dir, self.version)

        # data augmentation
        aug_data_path = data_path.replace(".json", "_train_aug.json")
        if not os.path.exists(aug_data_path) and self.create_aug_data:
            data_aug(data_path, data_dir)

        # data augmentation
        scr_data_path = data_path.replace(".json", "_train_scr.json")
        if (
            self.data_scr and not os.path.exists(scr_data_path)
        ) and self.create_scr_data:
            data_scr(data_path, data_dir)

        return data_path, data_dir

    def _setup_data(self, data_path, jsons_path):
        # # # loading directly from test file or val file
        if self.decode_all:
            all_data = self._load_json(data_path)
            self.messages = list(all_data.values())
        elif self.datatype.startswith("test"):
            test_path = data_path.replace(".json", "_test.json")
            if self.few_shot:
                test_path = test_path.replace(".json", "_fewshot.json")
            test_data = self._load_json(test_path)
            self.messages = list(test_data.values())
            if self.test_reduced:
                self.messages = self.messages[:100]

        elif self.datatype.startswith("valid"):
            valid_path = data_path.replace(".json", "_valid.json")
            if self.few_shot:
                valid_path = valid_path.replace(".json", "_fewshot.json")
            valid_data = self._load_json(valid_path)
            self.messages = list(valid_data.values())
            if self.val_reduced:
                k = min(len(self.messages), 500)
                # k = min(len(self.messages), 3000)
                self.messages = random.sample(list(valid_data.values()), k=k)

        else:
            train_path = data_path.replace(".json", "_train.json")
            if self.data_aug:
                train_path = train_path.replace(".json", "_aug_all.json")
            if self.data_scr:
                train_path = train_path.replace(".json", "_scr_all.json")
            if self.few_shot:
                train_path = train_path.replace(".json", "_fewshot.json")
            train_data = self._load_json(train_path)
            self.messages = list(train_data.values())

            random.shuffle(self.messages)

        if self.just_test:
            self.messages = self.messages[:10]

        # add prompts
        if self.use_prompts:
            for episode_idx in range(len(self.messages)):
                context, label = format_context_and_label(
                    self.messages[episode_idx]["context"],
                    self.messages[episode_idx]["slots_inf"],
                    seed=episode_idx,
                )
                self.messages[episode_idx]["context"] = context
                self.messages[episode_idx]["slots_inf"] = label

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom--slot_type--slot_val", ... ]
        """
        domains = [
            "attraction",
            "hotel",
            "hospital",
            "restaurant",
            "police",
            "taxi",
            "train",
        ]

        slot_types = [
            "stay",
            "price",
            "addr",
            "type",
            "arrive",
            "day",
            "depart",
            "dest",
            "area",
            "leave",
            "stars",
            "department",
            "people",
            "time",
            "food",
            "post",
            "phone",
            "name",
            "internet",
            "parking",
            "book stay",
            "book people",
            "book time",
            "book day",
            "pricerange",
            "destination",
            "leaveat",
            "arriveby",
            "departure",
        ]
        slots_list = []

        # # # remove start and ending token
        str_split = slots_string.strip().split()
        if str_split != [] and str_split[0] in ["<bs>", "</bs>"]:
            str_split = str_split[1:]
        if "</bs>" in str_split:
            str_split = str_split[: str_split.index("</bs>")]

        # # # split according to ","
        str_split = " ".join(str_split).split(",")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        str_split = [slot.strip() for slot in str_split]

        for slot_ in str_split:
            slot = slot_.split()
            if len(slot) > 2 and slot[0] in domains:
                domain = slot[0]
                if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                    slot_type = slot[1] + " " + slot[2]
                    slot_val = " ".join(slot[3:])
                else:
                    slot_type = slot[1]
                    slot_val = " ".join(slot[2:])
                if not slot_val == "dontcare":
                    slots_list.append(domain + "--" + slot_type + "--" + slot_val)
        return slots_list

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        resp = model_response.get("text", "")
        if not resp:
            self.metrics.add("empty_ct", SumMetric(1))
        if resp is None:
            return

        # import pdb; pdb.set_trace()
        # # # extract ground truth from labels
        slots_truth = self._extract_slot_from_string(labels[0])

        # # # extract generated slots from model_response
        slots_pred = self._extract_slot_from_string(resp)

        jga = set(slots_truth) == set(slots_pred)

        # import pdb; pdb.set_trace()

        self.metrics.add("joint goal acc", AverageMetric(jga))
        if resp:
            self.metrics.add("nonempty jga", AverageMetric(jga))

        # print out when predictions are wrong
        if set(slots_truth) != set(slots_pred):
            logging.info(
                f"\n\tteacher_action: {teacher_action}\n\tslots_truth: {slots_truth}\n\tslots_pred: {slots_pred}"
            )

        # keep track of coreference JGA
        if teacher_action.get("need_coref", False):
            self.metrics.add(
                "coref_jga", AverageMetric(set(slots_truth) == set(slots_pred))
            )
            self.metrics.add("coref_ct", SumMetric(1))

    def num_examples(self):
        # each turn be seen as a individual dialog
        return len(self.messages)

    def num_episodes(self):
        return len(self.messages)

    def get(self, episode_idx, entry_idx=0):
        # log_idx = entry_idx
        entry = self.messages[episode_idx]["context"]

        episode_done = True
        action = {
            "id": self.id,
            "text": entry,
            "episode_done": episode_done,
            "labels": [self.messages[episode_idx]["slots_inf"]],
            "dial_id": self.messages[episode_idx]["dial_id"],
            "turn_num": self.messages[episode_idx]["turn_num"],
        }
        if "need_coref" in self.messages[episode_idx]:
            action["need_coref"] = self.messages[episode_idx]["need_coref"]
        return action


class DefaultTeacher(MultiWozDSTTeacher):
    """
    Default teacher.
    """

    pass
