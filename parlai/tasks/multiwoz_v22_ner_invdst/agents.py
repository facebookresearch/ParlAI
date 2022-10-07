#!/usr/bin/env python3

"""
Multiwoz 2.2 Dataset implementation for ParlAI.
"""

import os
import json
import string
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
import parlai.tasks.multiwoz_v22_ner_invdst.build as build_
from parlai.core.metrics import AverageMetric
from parlai.core.message import Message
import numpy as np
from copy import deepcopy
from thefuzz import fuzz
import random
import parlai.utils.logging as logging


SEED = 42


class EntityMutator(object):
    SUPPORTED_MODES = {"scramble", "gibberish", "delex", "real"}
    SUPPORTED_ENTITY_SOURCEs = {"internal", "external"}
    ALLOWLIST_GIBBERISH = list("abcdefghijklmnopqrstuvwxyz")

    def __init__(self, opt, rng):
        self.opt = opt
        self.rng = rng

    def scramble(self, entity_name):
        modified_name = list(entity_name)
        self.rng.shuffle(modified_name)
        modified_name = "".join(modified_name)
        return modified_name

    def shuffle_words(self, entity_name):
        sub_words = entity_name.split(" ")
        sub_words = sub_words * 5
        self.rng.shuffle(sub_words)
        num_words = self.rng.randint(3, 10)
        return " ".join(sub_words[:num_words])

    def create_madeup_entities(self, entities):
        sub_words = []
        for entity in self.rng.choice(entities, 10):
            sub_words += entity.split(" ")
        self.rng.shuffle(sub_words)
        num_words = self.rng.randint(4, 15)
        return " ".join(sub_words[:num_words])

    def create_gibberish_entity(self, entity_name):
        min_num_words = 3
        max_num_words = 12

        min_word_len = 3
        max_word_len = 20
        num_entities = self.rng.randint(min_num_words, max_num_words)
        gibberish_entities = []
        for _ in range(num_entities):
            str_len = self.rng.randint(min_word_len, max_word_len)
            gibberish_entities.append(
                "".join(self.rng.choice(self.ALLOWLIST_GIBBERISH, str_len))
            )

        return " ".join(gibberish_entities)

    def delex(self, tag="a", basename="name"):
        return f"{basename}-{tag}"


class MultiWOZv22DSTTeacher(DialogTeacher):

    BELIEF_STATE_DELIM = ", "
    domains = [
        "attraction",
        "hotel",
        "hospital",
        "restaurant",
        "police",
        "taxi",
        "train",
    ]
    named_entity_slots = {
        "attraction--name",
        "restaurant--name",
        "hotel--name",
        "bus--departure",
        "bus--destination",
        "taxi--departure",
        "taxi--destination",
        "train--departure",
    }
    named_entity_interested = {"restaurant--name", "hotel--name"}
    rng = np.random.RandomState(SEED)

    @classmethod
    def add_cmdline_args(cls, argparser, partial_opt):
        argparser.add_argument(
            "--scramble-mode",
            type=str,
            default="scramble",
            choices=["scramble", "create_gibberish_entity"],
        )
        argparser.add_argument(
            "--entity", type=str, default="multiwoz", choices=["multiwoz", "g_sgd"]
        )
        argparser.add_argument(
            "--entity1", type=str, default="multiwoz", choices=["multiwoz", "g_sgd"]
        )
        argparser.add_argument(
            "--entity2", type=str, default="multiwoz", choices=["multiwoz", "g_sgd"]
        )
        argparser.add_argument("--comp-train", type=bool, default=False)
        argparser.add_argument("--comp-scramble", type=bool, default=False)
        argparser.add_argument("--uniform", type=bool, default=False)
        argparser.add_argument("--test-mode", type=bool, default=False)

        argparser.add_argument("--new-metric", type=bool, default=False)
        argparser.add_argument("--one-entity", type=bool, default=False)

        argparser.add_argument("--seed-np", type=int, default=42)

        return argparser

    def _my_strip(self, s):

        while not s[0].isalpha():
            s = s[1:]

        while not s[-1].isalpha():
            s = s[:-1]

        return s

    def _my_shuffle(self, s):

        s = list(s)
        random.shuffle(s)
        return "".join(s)

    def _my_scramble(self, s):

        s = s.split(" ")
        tmp = [self._my_shuffle(word) for word in s]
        random.shuffle(tmp)
        return " ".join(tmp)

    def __init__(self, opt: Opt, shared=None, *args, **kwargs):
        self.opt = opt
        self.rng = np.random.RandomState(SEED)
        self.fold = opt["datatype"].split(":")[0]
        opt["datafile"] = self.fold
        self.dpath = os.path.join(opt["datapath"], "multiwoz_v22")
        print(self.dpath)
        self.id = "multiwoz_v22"
        self.current_dialogue_id = "null"
        self.perf = 0
        self.flag_compute = 0
        self.flag_compute2 = False
        mutator = EntityMutator(self.opt, self.rng)
        self.mutator = {
            "scramble": mutator.scramble,
            "create_gibberish_entity": mutator.create_gibberish_entity,
        }

        hotel_names = {}
        hotel_names["train"] = set()
        hotel_names["dev"] = set()
        hotel_names["test"] = set()
        hotel_names_train = set()

        restaurant_names = {}
        restaurant_names["train"] = set()
        restaurant_names["dev"] = set()
        restaurant_names["test"] = set()

        ff = "/data/home/tianjianh/dstc8-schema-guided-dialogue/"
        # ff = "/home/w/Downloads/dstc8-schema-guided-dialogue/"

        for i in range(1, 128):
            with open(ff + "train/dialogues_%03d.json" % i, "r") as f:
                for line in f:
                    line = line.strip()
                    if (
                        '"restaurant_name":' in line
                        and not '"restaurant_name": [' in line
                    ):
                        word = line.split('": "')[1][:-2]
                        restaurant_names["train"].add(word.lower())
                    if '"hotel_name":' in line and not '"hotel_name": [' in line:
                        word = line.split('": "')[1][:-2]
                        hotel_names["train"].add(word.lower())

        for i in range(1, 20):
            with open(ff + "dev/dialogues_%03d.json" % i, "r") as f:
                for line in f:
                    line = line.strip()
                    if (
                        '"restaurant_name":' in line
                        and not '"restaurant_name": [' in line
                    ):
                        word = line.split('": "')[1][:-2]
                        restaurant_names["dev"].add(word.lower())
                    if '"hotel_name":' in line and not '"hotel_name": [' in line:
                        word = line.split('": "')[1][:-2]
                        hotel_names["dev"].add(word.lower())

        for i in range(1, 34):
            with open(ff + "test/dialogues_%03d.json" % i, "r") as f:
                for line in f:
                    line = line.strip()
                    if (
                        '"restaurant_name":' in line
                        and not '"restaurant_name": [' in line
                    ):
                        word = line.split('": "')[1][:-2]
                        restaurant_names["test"].add(word.lower())
                    if '"hotel_name":' in line and not '"hotel_name": [' in line:
                        word = line.split('": "')[1][:-2]
                        hotel_names["test"].add(word.lower())

        for key in hotel_names:
            hotel_names[key] = list(hotel_names[key])
            hotel_names[key].sort()

        for key in restaurant_names:
            restaurant_names[key] = list(restaurant_names[key])
            restaurant_names[key].sort()

        if self.opt["one_entity"] == True:
            hotel_names["train"] = ["holiday inn"]
            restaurant_names["train"] = ["taco bell"]

        g_sgd_bank = {"hotel-name": hotel_names, "restaurant-name": restaurant_names}

        g_sgd_bank["hotel-name"]["test"] = g_sgd_bank["hotel-name"]["dev"]

        # print(len(g_sgd_bank["hotel-name"]["train"]))
        # print(len(g_sgd_bank["hotel-name"]["dev"]))
        # print(len(g_sgd_bank["hotel-name"]["test"]))
        # print(len(g_sgd_bank["restaurant-name"]["train"]))
        # print(len(g_sgd_bank["restaurant-name"]["dev"]))
        # print(len(g_sgd_bank["restaurant-name"]["test"]))
        # exit(0)

        hotel_names = {}
        hotel_names["train"] = set()
        hotel_names["dev"] = set()
        hotel_names["test"] = set()
        hotel_names_train = set()

        restaurant_names = {}
        restaurant_names["train"] = set()
        restaurant_names["dev"] = set()
        restaurant_names["test"] = set()

        fname = "/data/home/tianjianh/multiwoz/data/MultiWOZ_2.2/"
        # fname = "/home/w/Downloads/multiwoz/data/MultiWOZ_2.2/"

        for i in range(1, 18):
            with open(fname + "train/dialogues_%03d.json" % i, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if '"restaurant-name": [' in lines[i]:
                        word = self._my_strip(lines[i + 1].strip()[1:-1])
                        restaurant_names["train"].add(word.lower())
                    if '"hotel-name": [' in lines[i]:
                        word = self._my_strip(lines[i + 1].strip()[1:-1])
                        hotel_names["train"].add(word.lower())

        for i in range(1, 3):
            with open(fname + "dev/dialogues_%03d.json" % i, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if '"restaurant-name": [' in lines[i]:
                        word = self._my_strip(lines[i + 1].strip()[1:-1])
                        restaurant_names["dev"].add(word.lower())
                    if '"hotel-name": [' in lines[i]:
                        word = self._my_strip(lines[i + 1].strip()[1:-1])
                        hotel_names["dev"].add(word.lower())

        for i in range(1, 3):
            with open(fname + "test/dialogues_%03d.json" % i, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if '"restaurant-name": [' in lines[i]:
                        word = self._my_strip(lines[i + 1].strip()[1:-1])
                        restaurant_names["test"].add(word.lower())
                    if '"hotel-name": [' in lines[i]:
                        word = self._my_strip(lines[i + 1].strip()[1:-1])
                        hotel_names["test"].add(word.lower())

        for key in hotel_names:
            hotel_names[key] = list(hotel_names[key])
            hotel_names[key].sort()

        for key in restaurant_names:
            restaurant_names[key] = list(restaurant_names[key])
            restaurant_names[key].sort()

        if self.opt["one_entity"] == True:
            hotel_names["train"] = ["holiday inn"]
            restaurant_names["train"] = ["taco bell"]

        multiwoz_bank = {"hotel-name": hotel_names, "restaurant-name": restaurant_names}

        # print(len(multiwoz_bank["hotel-name"]["train"]))
        # print(len(multiwoz_bank["hotel-name"]["dev"]))
        # print(len(multiwoz_bank["hotel-name"]["test"]))
        # print(len(multiwoz_bank["restaurant-name"]["train"]))
        # print(len(multiwoz_bank["restaurant-name"]["dev"]))
        # print(len(multiwoz_bank["restaurant-name"]["test"]))

        # exit(0)

        self.entity_bank = {"multiwoz": multiwoz_bank, "g_sgd": g_sgd_bank}

        if shared is None:
            build_.build(opt)
        super().__init__(opt, shared)

    def _load_data(self, fold):
        dataset_fold = "dev" if fold == "valid" else fold
        fold_path = os.path.join(self.dpath, dataset_fold)
        dialogs = []
        for file_id in range(1, build_.fold_size(dataset_fold) + 1):
            filename = os.path.join(fold_path, f"dialogues_{file_id:03d}.json")
            with PathManager.open(filename, "r") as f:
                dialogs += json.load(f)
        return dialogs

    def _get_curr_belief_states(self, turn):
        belief_states = []
        for frame in turn["frames"]:
            if "state" in frame:
                if "slot_values" in frame["state"]:
                    for domain_slot_type in frame["state"]["slot_values"]:
                        for slot_value in frame["state"]["slot_values"][
                            domain_slot_type
                        ]:
                            domain, slot_type = domain_slot_type.split("-")
                            belief_state = (
                                f"{domain} {slot_type} {slot_value.lower()}"
                                + "//"
                                + frame["service"]
                                + "//"
                                + domain_slot_type
                                + "//"
                                + slot_value.lower()
                            )
                            belief_states.append(belief_state)
        return belief_states

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom--slot_type--slot_val", ... ]
        """
        slots_list = []
        per_domain_slot_lists = {}
        named_entity_slot_lists = []
        named_entity_slot_interested_lists = []

        # # # remove start and ending token if any
        str_split = slots_string.strip().split()
        if str_split != [] and str_split[0] in ["<bs>", "</bs>"]:
            str_split = str_split[1:]
        if "</bs>" in str_split:
            str_split = str_split[: str_split.index("</bs>")]

        # split according to ";"
        # str_split = slots_string.split(self.BELIEF_STATE_DELIM)
        str_split = " ".join(str_split).split(",")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        str_split = [slot.strip() for slot in str_split]
        for slot_ in str_split:
            slot = slot_.split()
            if len(slot) > 2 and slot[0] in self.domains:
                domain = slot[0]
                slot_type = slot[1]
                slot_val = " ".join(slot[2:])
                if not slot_val == "dontcare":
                    slots_list.append(domain + "--" + slot_type + "--" + slot_val)
                if domain in per_domain_slot_lists:
                    per_domain_slot_lists[domain].add(slot_type + "--" + slot_val)
                else:
                    per_domain_slot_lists[domain] = {slot_type + "--" + slot_val}
                if domain + "--" + slot_type in self.named_entity_slots:
                    named_entity_slot_lists.append(
                        domain + "--" + slot_type + "--" + slot_val
                    )
                if domain + "--" + slot_type in self.named_entity_interested:
                    named_entity_slot_interested_lists.append(
                        domain + "--" + slot_type + "--" + slot_val
                    )

        return (
            slots_list,
            per_domain_slot_lists,
            named_entity_slot_lists,
            named_entity_slot_interested_lists,
        )

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        """
        for dialog state tracking, we compute the joint goal accuracy, which is
        the percentage of the turns where the model correctly and precisely
        predicts all slots(domain, slot_type, slot_value).
        """

        # print(teacher_action.keys())
        # print(teacher_action["iidx"])
        # idxx = teacher_action.get("idxx")
        # print(idxx)
        # exit(0)

        ta_text = teacher_action.get("text")

        resp = model_response.get("text")
        # idxx = teacher_action.get("idxx")
        # idxx = teacher_action["iidx"]
        idxx = 0
        if not resp:
            return
        # extract ground truth from labels
        (
            slots_truth,
            slots_truth_per_domain,
            slots_truth_named_entity,
            slots_truth_named_entity_interested,
        ) = self._extract_slot_from_string(labels[0])
        # extract generated slots from model_response
        (
            slots_pred,
            slots_pred_per_domain,
            slots_pred_named_entity,
            slots_pred_named_entity_interested,
        ) = self._extract_slot_from_string(resp)

        # if set(slots_truth) != set(slots_pred):
        # logging.info(f"\n\tslots_truth: {slots_truth}\n\tslots_pred: {slots_pred}")

        # if set(slots_truth) == set(slots_pred):
        #     print("JGA_NEW", 1)
        # else:
        #     print("JGA_NEW", 0)
        # print("IDXX", idxx)
        # print("TEXT", ta_text)

        if self.opt["new_metric"] == False:
            if (len(slots_truth_named_entity_interested) > 0) or (
                len(slots_pred_named_entity_interested) > 0
            ):

                self.metrics.add(
                    "jga_c", AverageMetric(set(slots_truth) == set(slots_pred))
                )

                # if set(slots_truth) == set(slots_pred):
                #     print("CJGA_NEW", 1)
                # else:
                #     print("CJGA_NEW", 0)
                # print("IDXX", idxx)
                # print("TEXT", ta_text)

                for gt_slot in slots_truth:
                    self.metrics.add("slot_r_c", AverageMetric(gt_slot in slots_pred))

                for predicted_slot in slots_pred:
                    self.metrics.add(
                        "slot_p_c", AverageMetric(predicted_slot in slots_truth)
                    )

        if self.opt["new_metric"] == True:

            # print("!" * 20)
            # print(self.flag_compute)

            self.flag_compute2 = self.flag_compute2 or (
                len(slots_truth_named_entity_interested) > 0
            )
            self.flag_compute2 = self.flag_compute2 or (
                len(slots_pred_named_entity_interested) > 0
            )

            slot_p_curr = True
            slot_r_curr = True

            for gt_slot in slots_truth_named_entity_interested:
                slot_r_curr = slot_r_curr and (
                    gt_slot in slots_pred_named_entity_interested
                )

            for pred_slot in slots_pred_named_entity_interested:
                slot_p_curr = slot_p_curr and (
                    pred_slot in slots_truth_named_entity_interested
                )

            jga_curr = set(slots_pred_named_entity_interested) == set(
                slots_truth_named_entity_interested
            )

            if self.flag_compute == 1:

                self.metrics.add(
                    "NEW_METRIC_SLOT_P", AverageMetric(slot_p_curr == self.slot_p_prev)
                )
                self.metrics.add(
                    "NEW_METRIC_SLOT_R", AverageMetric(slot_r_curr == self.slot_r_prev)
                )
                self.metrics.add(
                    "NEW_METRIC_JGA", AverageMetric(jga_curr == self.jga_prev)
                )
                if self.jga_prev == True:
                    self.metrics.add("DAIM", AverageMetric(jga_curr == self.jga_prev))

                # only cares about examples which has hotel names and restaurant names
                if self.flag_compute2 == True:

                    # print("999999", self.slot_p_prev)
                    self.metrics.add(
                        "NEW_METRIC_SLOT_P_C",
                        AverageMetric(slot_p_curr == self.slot_p_prev),
                    )
                    self.metrics.add(
                        "NEW_METRIC_SLOT_R_C",
                        AverageMetric(slot_r_curr == self.slot_r_prev),
                    )
                    self.metrics.add(
                        "NEW_METRIC_JGA_C", AverageMetric(jga_curr == self.jga_prev)
                    )
                    if self.jga_prev == True:
                        self.metrics.add(
                            "DAIM_C", AverageMetric(jga_curr == self.jga_prev)
                        )
                    self.flag_compute2 = False

                    # print("computed")

            self.flag_compute = 1 - self.flag_compute

            self.slot_p_prev = slot_p_curr
            self.slot_r_prev = slot_r_curr
            self.jga_prev = jga_curr
            print("888888", self.slot_p_prev)

        if self.opt["test_mode"] == True:

            # print(self.perf)
            # print(self.current_dialogue_id)

            if teacher_action.get("dialogue_id") != self.current_dialogue_id:

                if self.current_dialogue_id != "null":

                    p = self.perf / 10
                    tmp = -p * np.log(p + 1e-7) - (1 - p) * np.log(1 - p + 1e-7)
                    self.metrics.add("NEI", AverageMetric(tmp))
                    self.metrics.add("NEI2", AverageMetric(self.perf))
                    self.perf = 0

                    # print("computed")

                self.current_dialogue_id = teacher_action.get("dialogue_id")

            predicted_slot_ne = set()
            true_slot_ne = set()
            for slot in slots_truth:
                if "hotel--name" in slot:
                    true_slot_ne.add(slot.replace("the ", ""))
                if "restaurant--name" in slot:
                    true_slot_ne.add(slot.replace("the ", ""))

            for slot in slots_pred:
                # print("?")
                # print(slot)
                if "hotel--name" in slot:
                    predicted_slot_ne.add(slot.replace("the ", ""))
                    # print("!")
                if "restaurant--name" in slot:
                    predicted_slot_ne.add(slot.replace("the ", ""))
                    # print("!")

            self.perf += true_slot_ne == predicted_slot_ne

            # print()
            # print(predicted_slot_ne)
            # print(true_slot_ne)

        for gt_slot in slots_truth:
            self.metrics.add("all/slot_r", AverageMetric(gt_slot in slots_pred))
            curr_domain = gt_slot.split("--")[0]
            self.metrics.add(
                f"{curr_domain}/slot_r", AverageMetric(gt_slot in slots_pred)
            )
        for predicted_slot in slots_pred:
            self.metrics.add("all/slot_p", AverageMetric(predicted_slot in slots_truth))
            curr_domain = predicted_slot.split("--")[0]
            self.metrics.add(
                f"{curr_domain}/slot_p", AverageMetric(predicted_slot in slots_truth)
            )
        self.metrics.add("jga", AverageMetric(set(slots_truth) == set(slots_pred)))
        self.metrics.add(
            "named_entities/jga",
            AverageMetric(
                set(slots_truth_named_entity) == set(slots_pred_named_entity)
            ),
        )
        for gt_slot in slots_truth_named_entity:
            self.metrics.add("all_ne/slot_r", AverageMetric(gt_slot in slots_pred))
            curr_domain = gt_slot.split("--")[0]
            self.metrics.add(
                f"{curr_domain}_ne/slot_r", AverageMetric(gt_slot in slots_pred)
            )
        for predicted_slot in slots_pred_named_entity:
            self.metrics.add(
                "all_ne/slot_p", AverageMetric(predicted_slot in slots_truth)
            )
            curr_domain = predicted_slot.split("--")[0]
            ne = predicted_slot.split("--")[-1]
            for tmp_slot in slots_truth:
                slot_name = tmp_slot.split("--")[0] + " " + tmp_slot.split("--")[1]
                ne = ne.replace(slot_name, "")

            for tmp_slot in slots_pred:
                slot_name = tmp_slot.split("--")[0] + " " + tmp_slot.split("--")[1]
                ne = ne.replace(slot_name, "")

            # print(",,,,,", ne, ",,,,,")
            self.metrics.add(
                f"{curr_domain}_ne/slot_p", AverageMetric(predicted_slot in slots_truth)
            )
            self.metrics.add(
                f"{curr_domain}_ne/hallucination",
                AverageMetric(not (ne.strip() in teacher_action.get("text"))),
            )

            # get combined hallucination
            self.metrics.add(
                f"all_ne/hallucination",
                AverageMetric(not (ne.strip() in teacher_action.get("text"))),
            )
        for domain in slots_truth_per_domain:
            if domain in slots_pred_per_domain:
                self.metrics.add(
                    f"{domain}/jga",
                    AverageMetric(
                        slots_truth_per_domain[domain] == slots_pred_per_domain[domain]
                    ),
                )

        print("-" * 30)

    def setup_data(self, fold):

        iidx = 0

        random.seed(self.opt["seed_np"])
        # print("!!!!!", fold)
        # i = 0

        dialogs = self._load_data(fold)
        examples = []
        for dialog in dialogs:
            context = []
            # print("-------")
            # print(type(dialog['turns']))
            # print(len(dialog['turns']))

            if self.opt["test_mode"] == True:

                for turn in dialog["turns"][:-1]:

                    curr_turn = turn["utterance"].lower()
                    curr_speaker = "<user>" if turn["speaker"] == "USER" else "<system>"
                    curr_context = f"{curr_speaker} {curr_turn}".lower()
                    context.append(curr_context)
                    cum_belief_states = self._get_curr_belief_states(turn)

                # print("????")

                found = False
                for item in cum_belief_states:
                    if "hotel-name" in item or "restaurant-name" in item:
                        found = True
                        break

                if found:

                    # cum_belief_states_ = [item.split("//")[0] for item in cum_belief_states]
                    # cum_belief_states_ = list(set(cum_belief_states_))
                    # cum_belief_states_.sort()
                    # examples.append({
                    #     'dialogue_id': dialog['dialogue_id'],
                    #     'turn_num': turn['turn_id'],
                    #     'text': " ".join(context),
                    #     'labels': ", ".join(cum_belief_states_)
                    # })

                    packed_example = []

                    for kk in range(10):
                        tmp_text = " ".join(context)
                        tmp_cum_belief_states = list(set(cum_belief_states))
                        tmp_cum_belief_states.sort()
                        processed_cum_belief_states = []
                        replaced_names = []
                        for bs in tmp_cum_belief_states:
                            bs_ = bs.split("//")

                            # print(bs_[2])

                            if bs_[2] in self.entity_bank[self.opt["entity"]]:

                                # print("???????")

                                if bs_[3] in tmp_text:

                                    # print("!!!!!!!!")

                                    new_slot_value = random.choice(
                                        self.entity_bank[self.opt["entity"]][bs_[2]][
                                            fold
                                        ]
                                    )
                                    tmp_text = tmp_text.replace(bs_[3], new_slot_value)
                                    processed_cum_belief_states.append(
                                        bs_[0].replace(bs_[3], new_slot_value)
                                    )
                                    replaced_names.append((bs_[3], new_slot_value))

                                else:
                                    processed_cum_belief_states.append(bs_[0])

                            else:
                                processed_cum_belief_states.append(bs_[0])

                        for (x, x_) in replaced_names:
                            for idx in range(len(processed_cum_belief_states)):
                                processed_cum_belief_states[
                                    idx
                                ] = processed_cum_belief_states[idx].replace(x, x_)

                        packed_example.append(
                            {
                                "dialogue_id": dialog["dialogue_id"],
                                "turn_num": turn["turn_id"],
                                "text": tmp_text,
                                "labels": ", ".join(processed_cum_belief_states),
                            }
                        )

                    examples.append(packed_example)

            else:

                for turn in dialog["turns"]:

                    # print(turn['speaker'])

                    curr_turn = turn["utterance"].lower()
                    curr_speaker = "<user>" if turn["speaker"] == "USER" else "<system>"
                    curr_context = f"{curr_speaker} {curr_turn}".lower()
                    context.append(curr_context)
                    cum_belief_states = self._get_curr_belief_states(turn)
                    # import pdb; pdb.set_trace()

                    if curr_speaker == "<user>":

                        if self.opt["new_metric"] == True:

                            if self.opt["entity1"] == "multiwoz":
                                # just clean up and format sample
                                cum_belief_states_ = [
                                    item.split("//")[0] for item in cum_belief_states
                                ]
                                cum_belief_states_ = list(set(cum_belief_states_))
                                cum_belief_states_.sort()

                                a = {
                                    "dialogue_id": dialog["dialogue_id"],
                                    "turn_num": turn["turn_id"],
                                    "text": " ".join(context),
                                    "labels": ", ".join(cum_belief_states_),
                                }

                            else:

                                tmp_text = " ".join(context)
                                tmp_cum_belief_states = list(set(cum_belief_states))
                                tmp_cum_belief_states.sort()
                                processed_cum_belief_states = []
                                replaced_names = []
                                for bs in tmp_cum_belief_states:

                                    bs_ = bs.split("//")

                                    if bs_[2] in self.entity_bank[self.opt["entity1"]]:

                                        if bs_[3] in tmp_text:

                                            new_slot_value = random.choice(
                                                self.entity_bank[self.opt["entity1"]][
                                                    bs_[2]
                                                ][fold]
                                            )
                                            tmp_text = tmp_text.replace(
                                                bs_[3], new_slot_value
                                            )
                                            processed_cum_belief_states.append(
                                                bs_[0].replace(bs_[3], new_slot_value)
                                            )
                                            replaced_names.append(
                                                (bs_[3], new_slot_value)
                                            )

                                        else:
                                            processed_cum_belief_states.append(bs_[0])

                                    else:
                                        processed_cum_belief_states.append(bs_[0])

                                for (x, x_) in replaced_names:
                                    for idx in range(len(processed_cum_belief_states)):
                                        processed_cum_belief_states[
                                            idx
                                        ] = processed_cum_belief_states[idx].replace(
                                            x, x_
                                        )

                                a = {
                                    "dialogue_id": dialog["dialogue_id"],
                                    "turn_num": turn["turn_id"],
                                    "text": tmp_text,
                                    "labels": ", ".join(processed_cum_belief_states),
                                }

                            tmp_text = " ".join(context)
                            tmp_cum_belief_states = list(set(cum_belief_states))
                            tmp_cum_belief_states.sort()
                            processed_cum_belief_states = []
                            replaced_names = []
                            for bs in tmp_cum_belief_states:

                                bs_ = bs.split("//")

                                # check if it is in the entity bank
                                if bs_[2] in self.entity_bank[self.opt["entity2"]]:

                                    if bs_[3] in tmp_text:

                                        new_slot_value = random.choice(
                                            self.entity_bank[self.opt["entity2"]][
                                                bs_[2]
                                            ][fold]
                                        )
                                        if self.opt["comp_scramble"] == True:
                                            new_slot_value = self.mutator[
                                                self.opt["scramble_mode"]
                                            ](new_slot_value)
                                        tmp_text = tmp_text.replace(
                                            bs_[3], new_slot_value
                                        )
                                        processed_cum_belief_states.append(
                                            bs_[0].replace(bs_[3], new_slot_value)
                                        )
                                        replaced_names.append((bs_[3], new_slot_value))

                                    else:
                                        processed_cum_belief_states.append(bs_[0])

                                # if not, pass
                                else:
                                    processed_cum_belief_states.append(bs_[0])

                            for (x, x_) in replaced_names:
                                for idx in range(len(processed_cum_belief_states)):
                                    processed_cum_belief_states[
                                        idx
                                    ] = processed_cum_belief_states[idx].replace(x, x_)

                            b = {
                                "dialogue_id": dialog["dialogue_id"],
                                "turn_num": turn["turn_id"],
                                "text": tmp_text,
                                "labels": ", ".join(processed_cum_belief_states),
                            }

                            # only add examples that have entities swapped.
                            if b["labels"] != a["labels"]:
                                examples.append((a, b))
                            # examples.append((a, b))

                        elif self.opt["comp_train"] == False:

                            # if self.opt["test_mode"] == True:

                            #     print("????")

                            #     found = False
                            #     for item in cum_belief_states:
                            #         if "hotel-name" in item or "restaurant-name" in item:
                            #             found = True
                            #             break

                            #     if found:

                            #         cum_belief_states_ = [item.split("//")[0] for item in cum_belief_states]
                            #         cum_belief_states_ = list(set(cum_belief_states_))
                            #         cum_belief_states_.sort()
                            #         examples.append({
                            #             'dialogue_id': dialog['dialogue_id'],
                            #             'turn_num': turn['turn_id'],
                            #             'text': " ".join(context),
                            #             'labels': ", ".join(cum_belief_states_)
                            #         })

                            if (
                                self.opt["entity"] == "multiwoz"
                                and self.opt["uniform"] == False
                            ):
                                cum_belief_states_ = [
                                    item.split("//")[0] for item in cum_belief_states
                                ]
                                cum_belief_states_ = list(set(cum_belief_states_))
                                cum_belief_states_.sort()
                                iidx += 1
                                examples.append(
                                    {
                                        "dialogue_id": dialog["dialogue_id"],
                                        "turn_num": turn["turn_id"],
                                        "text": " ".join(context),
                                        "labels": ", ".join(cum_belief_states_),
                                        "iidx": iidx,
                                    }
                                )

                            else:
                                tmp_text = " ".join(context)
                                tmp_cum_belief_states = list(set(cum_belief_states))
                                tmp_cum_belief_states.sort()
                                processed_cum_belief_states = []
                                replaced_names = []
                                for bs in tmp_cum_belief_states:
                                    bs_ = bs.split("//")

                                    # print(bs_[2])

                                    if bs_[2] in self.entity_bank[self.opt["entity"]]:

                                        # print("???????")

                                        if bs_[3] in tmp_text:

                                            # print("!!!!!!!!")

                                            new_slot_value = random.choice(
                                                self.entity_bank[self.opt["entity"]][
                                                    bs_[2]
                                                ][fold]
                                            )
                                            tmp_text = tmp_text.replace(
                                                bs_[3], new_slot_value
                                            )
                                            processed_cum_belief_states.append(
                                                bs_[0].replace(bs_[3], new_slot_value)
                                            )
                                            replaced_names.append(
                                                (bs_[3], new_slot_value)
                                            )

                                        else:
                                            processed_cum_belief_states.append(bs_[0])

                                    else:
                                        processed_cum_belief_states.append(bs_[0])

                                for (x, x_) in replaced_names:
                                    for idx in range(len(processed_cum_belief_states)):
                                        processed_cum_belief_states[
                                            idx
                                        ] = processed_cum_belief_states[idx].replace(
                                            x, x_
                                        )

                                iidx += 1
                                examples.append(
                                    {
                                        "dialogue_id": dialog["dialogue_id"],
                                        "turn_num": turn["turn_id"],
                                        "text": tmp_text,
                                        "labels": ", ".join(
                                            processed_cum_belief_states
                                        ),
                                        "iidx": iidx,
                                    }
                                )

                        else:

                            if self.opt["entity"] == "multiwoz":

                                cum_belief_states_ = [
                                    item.split("//")[0] for item in cum_belief_states
                                ]
                                cum_belief_states_ = list(set(cum_belief_states_))
                                cum_belief_states_.sort()

                                a = {
                                    "dialogue_id": dialog["dialogue_id"],
                                    "turn_num": turn["turn_id"],
                                    "text": " ".join(context),
                                    "labels": ", ".join(cum_belief_states_),
                                }

                            else:

                                tmp_text = " ".join(context)
                                tmp_cum_belief_states = list(set(cum_belief_states))
                                tmp_cum_belief_states.sort()
                                processed_cum_belief_states = []
                                replaced_names = []
                                for bs in tmp_cum_belief_states:

                                    bs_ = bs.split("//")

                                    if bs_[2] in self.entity_bank[self.opt["entity"]]:

                                        if bs_[3] in tmp_text:

                                            new_slot_value = random.choice(
                                                self.entity_bank[self.opt["entity"]][
                                                    bs_[2]
                                                ][fold]
                                            )
                                            tmp_text = tmp_text.replace(
                                                bs_[3], new_slot_value
                                            )
                                            processed_cum_belief_states.append(
                                                bs_[0].replace(bs_[3], new_slot_value)
                                            )
                                            replaced_names.append(
                                                (bs_[3], new_slot_value)
                                            )

                                        else:
                                            processed_cum_belief_states.append(bs_[0])

                                    else:
                                        processed_cum_belief_states.append(bs_[0])

                                for (x, x_) in replaced_names:
                                    for idx in range(len(processed_cum_belief_states)):
                                        processed_cum_belief_states[
                                            idx
                                        ] = processed_cum_belief_states[idx].replace(
                                            x, x_
                                        )

                                a = {
                                    "dialogue_id": dialog["dialogue_id"],
                                    "turn_num": turn["turn_id"],
                                    "text": tmp_text,
                                    "labels": ", ".join(processed_cum_belief_states),
                                }

                            tmp_text = " ".join(context)
                            tmp_cum_belief_states = list(set(cum_belief_states))
                            tmp_cum_belief_states.sort()
                            processed_cum_belief_states = []
                            replaced_names = []
                            for bs in tmp_cum_belief_states:

                                bs_ = bs.split("//")

                                if bs_[2] in self.entity_bank[self.opt["entity"]]:

                                    if bs_[3] in tmp_text:

                                        new_slot_value = random.choice(
                                            self.entity_bank[self.opt["entity"]][
                                                bs_[2]
                                            ][fold]
                                        )
                                        if self.opt["comp_scramble"] == True:
                                            new_slot_value = self.mutator[
                                                self.opt["scramble_mode"]
                                            ](new_slot_value)
                                        tmp_text = tmp_text.replace(
                                            bs_[3], new_slot_value
                                        )
                                        processed_cum_belief_states.append(
                                            bs_[0].replace(bs_[3], new_slot_value)
                                        )
                                        replaced_names.append((bs_[3], new_slot_value))

                                    else:
                                        processed_cum_belief_states.append(bs_[0])

                                else:
                                    processed_cum_belief_states.append(bs_[0])

                            for (x, x_) in replaced_names:
                                for idx in range(len(processed_cum_belief_states)):
                                    processed_cum_belief_states[
                                        idx
                                    ] = processed_cum_belief_states[idx].replace(x, x_)

                            b = {
                                "dialogue_id": dialog["dialogue_id"],
                                "turn_num": turn["turn_id"],
                                "text": tmp_text,
                                "labels": ", ".join(processed_cum_belief_states),
                            }

                            examples.append((a, b))

        # self.rng.shuffle(examples)

        if self.opt["comp_train"] == True or self.opt["new_metric"] == True:

            tmp = []
            for example in examples:
                tmp.append(example[0])
                tmp.append(example[1])
            examples = tmp

        if self.opt["test_mode"] == True:

            tmp = []
            for example in examples:
                for item in example:
                    tmp.append(item)

            examples = tmp

        # for i in range(32):

        #     print(examples[i])
        #     print()

        # print("?!")
        # print(len(examples))

        for example in examples:

            yield example, True


class DefaultTeacher(MultiWOZv22DSTTeacher):
    pass
