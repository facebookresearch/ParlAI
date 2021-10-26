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
from collections import defaultdict
from .build import build

# from .utils.reformat import reformat_parlai
import parlai.utils.logging as logging
import re


class MultiWozDSTTeacher(FixedDialogTeacher):
    """
    MultiWOZ DST Teacher.
    """

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
    named_entity_interested = {"restaurant--name", "hotel--name", "attraction--name"}

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'multiwoz_laug_dst'

        # # # reading args
        self.just_test = opt.get('just_test', False)
        self.seed = opt.get('rand_seed', 0)
        self.data_aug = opt.get('augmentation_method', "orig")
        self.val_reduced = opt.get("val_reduced", False)
        self.test_reduced = opt.get("test_reduced", False)
        self.reduce_train_factor = opt.get("reduce_train_factor", 1)
        self.flag_compute = 0
        self.few_shot = opt.get("few_shot", False)

        # # # set random seeds
        random.seed(self.seed)

        opt['datafile'], data_dir = self._path(opt)

        if self.data_aug.lower() == "nei":
            "Setting up entity banks for named entity invariance"
            self.ner_inv_init()

        self._setup_data(opt['datafile'], data_dir)

        self.reset()

    def _my_strip(self, s):

        while not s[0].isalpha():
            s = s[1:]

        while not s[-1].isalpha():
            s = s[:-1]

        return s

    def ner_inv_init(self):
        # copied over from Tianjian's code. Load entity names for swapping
        names = {}

        ff = "/data/home/justincho/project/ParlAI/data/dstc8-schema-guided-dialogue/"
        # ff = "/home/w/Downloads/dstc8-schema-guided-dialogue/"

        def get_sgd_entities_from_file(filename: str, fold: str):
            """
            fold one of ["train", "valid", "test"]
            """
            with open(filename, "r") as f:
                for line in f:
                    line = line.strip()
                    entity_name_search_result = re.search("\"(.*)_name\":", line)
                    if entity_name_search_result and not re.search(
                        "\"(.*)_name\": \[", line
                    ):
                        domain = entity_name_search_result[1]
                        word = line.split("\": \"")[1][:-2]
                        # restaurant_names[fold].add(word.lower())
                        if f'{domain}-name' not in names:
                            names[f'{domain}-name'] = defaultdict(set)
                        names[f'{domain}-name'][fold].add(word.lower())
                    # if "\"hotel_name\":" in line and not "\"hotel_name\": [" in line:
                    #     word = line.split("\": \"")[1][:-2]
                    #     hotel_names[fold].add(word.lower())

        for i in range(1, 128):
            filename = ff + "train/dialogues_%03d.json" % i
            get_sgd_entities_from_file(filename, fold="train")

        for i in range(1, 20):
            filename = ff + "dev/dialogues_%03d.json" % i
            get_sgd_entities_from_file(filename, fold="dev")

        for i in range(1, 34):
            filename = ff + "test/dialogues_%03d.json" % i
            get_sgd_entities_from_file(filename, fold="test")

        def sort_names(entity_names_dict: defaultdict):
            for domain, entity_names in entity_names_dict.items():
                for key in entity_names:
                    entity_names[key] = sorted(list(entity_names[key]))

        sort_names(names)
        # sort_names(hotel_names)
        # sort_names(restaurant_names)

        # if self.opt["one_entity"] == True:
        #     hotel_names["train"] = ["holiday inn"]
        #     restaurant_names["train"] = ["taco bell"]

        # g_sgd_bank = {"hotel-name": hotel_names, "restaurant-name": restaurant_names}
        g_sgd_bank = names
        g_sgd_bank["hotel-name"]["test"] = g_sgd_bank["hotel-name"]["dev"]

        names = {}

        # fname = "/data/home/tianjianh/multiwoz/data/MultiWOZ_2.2/"
        fname = "/data/home/justincho/project/ParlAI/data/multiwoz_dst/MULTIWOZ2.2/"

        def get_multiwoz_entities_from_file(filename: str, fold: str):
            """
            fold one of ["train", "valid", "test"]
            """
            with open(filename, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    entity_name_search_result = re.search("\"(.*)-name\": \[", lines[i])
                    if entity_name_search_result:
                        domain = entity_name_search_result[1]
                        word = self._my_strip(lines[i + 1].strip()[1:-1])
                        if f"{domain}-name" not in names:
                            names[f"{domain}-name"] = defaultdict(set)
                        names[f"{domain}-name"][fold].add(word.lower())
                    # if "\"hotel-name\": [" in lines[i]:
                    #     word = self._my_strip(lines[i + 1].strip()[1: -1])
                    #     hotel_names[fold].add(word.lower())

        for i in range(1, 18):
            filename = fname + "train/dialogues_%03d.json" % i
            get_multiwoz_entities_from_file(filename, fold="train")

        for i in range(1, 3):
            filename = fname + "dev/dialogues_%03d.json" % i
            get_multiwoz_entities_from_file(filename, fold="dev")

        for i in range(1, 3):
            filename = fname + "test/dialogues_%03d.json" % i
            get_multiwoz_entities_from_file(filename, fold="test")

        # if self.opt["one_entity"] == True:
        #     hotel_names["train"] = ["holiday inn"]
        #     restaurant_names["train"] = ["taco bell"]

        sort_names(names)
        multiwoz_bank = names

        self.entity_bank = {"multiwoz": multiwoz_bank, "g_sgd": g_sgd_bank}

        print_stats = True
        if print_stats:
            for dataset, bank in self.entity_bank.items():
                print(f"Dataset: {dataset}")
                for domain_slot, all_names in bank.items():
                    print(f"\tDomain slot: {domain_slot}")
                    for fold, names in all_names.items():
                        print(f"\t\t{fold}: {len(names)}")

    @classmethod
    # def add_cmdline_args(cls, argparser):
    def add_cmdline_args(cls, argparser, partial_opt):
        agent = argparser.add_argument_group('MultiWozDST Teacher Args')
        agent.add_argument(
            '-aug',
            '--augmentation_method',
            type=str,
            default="orig",
            help="one of ['orig', 'TP', 'SD', 'NEI'] where TP: text paraphrase, SD: speech disfluencies, NEI: named entity invariance",
        )
        agent.add_argument(
            '-swap',
            '--swap_entity',
            type=str,
            default="g_sgd",
            help="one of ['g_sgd', 'scrambled', 'random'] ",
        )
        agent.add_argument(
            '--just_test',
            type='bool',
            default=False,
            help="True if one would like to test agents with small amount of data (default: False).",
        )
        agent.add_argument(
            '--reduce_train_factor',
            type=int,
            default=1,
            help="Factor to use in shrinking the training dataset size",
        )
        agent.add_argument(
            '-fs',
            '--few_shot',
            type=bool,
            default=False,
            help="Whether to simulate few shot setting by limiting trainig to 50 conversaions per domain",
        )
        agent.add_argument(
            '--rand_seed',
            type=int,
            default=0,
            help="specify to set random seed (default: 0).",
        )

        agent.add_argument(
            '--val_reduced',
            type='bool',
            default=False,
            help="use smaller evaluation set.",
        )

        agent.add_argument(
            '--test_reduced', type='bool', default=False, help="use smaller test set."
        )

        return argparser

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def _path(self, opt):
        # set up path to data (specific to each dataset)
        data_dir = os.path.join(opt['datapath'], 'LAUG', opt['augmentation_method'])
        # data_dir = os.path.join('/checkpoint/kunqian/multiwoz/data/MultiWOZ_2.1/')

        # for NER invariance, load from original data and augment it dynamically
        if opt['augmentation_method'].lower() == "nei":
            data_dir = os.path.join(opt['datapath'], 'LAUG', 'orig')

        data_path = os.path.join(data_dir, 'data_reformat_official_v23_slots.json')

        # # uncomment to use original LAUG slot values
        # data_path = os.path.join(data_dir, 'data_reformat.json')

        # build the data if it does not exist
        build(opt)

        return data_path, data_dir

    def nei_augment(self, messages):

        # swap out entities as need be
        augmented_messages = []
        # keep only the examples for which the entities are swapped
        for m in messages:
            m_copy = m.copy()
            orig_context = m_copy['context']
            new_context = orig_context
            orig_belief_state = m_copy["slots_inf"]

            replaced_names = []
            belief_states = orig_belief_state.split(",")
            new_belief_states = []
            swapped = False
            for bs in belief_states:

                if bs == "":
                    continue

                splits = bs.split()
                assert len(bs) >= 3, f"Split is not >= 3 {bs}"

                domain = splits[0]
                slot_key = splits[1]
                slot_val = " ".join(splits[2:])
                domain_slot_key = f"{domain}-{slot_key}"
                # check if it is in the entity bank
                if domain_slot_key in self.entity_bank[self.opt["swap_entity"]]:

                    if slot_val in orig_context:

                        new_slot_value = random.choice(
                            self.entity_bank[self.opt["swap_entity"]][domain_slot_key][
                                self.opt['datatype']
                            ]
                        )
                        # if self.opt["comp_scramble"] == True:
                        #     new_slot_value = self.mutator[self.opt["scramble_mode"]](new_slot_value)
                        new_context = orig_context.replace(slot_val, new_slot_value)
                        new_belief_states.append(bs.replace(slot_val, new_slot_value))
                        replaced_names.append((slot_val, new_slot_value))
                    else:
                        new_belief_states.append(bs)

                # if not, pass
                else:
                    new_belief_states.append(bs)

            for (orig, replacement) in replaced_names:
                for idx in range(len(new_belief_states)):
                    new_belief_states[idx] = new_belief_states[idx].replace(
                        orig, replacement
                    )

            m_copy["slots_inf"] = ", ".join(new_belief_states).strip()
            m_copy['context'] = new_context
            # only add examples that have entities swapped.

            if new_context != orig_context:
                augmented_messages.append(m)
                augmented_messages.append(m_copy)

        return augmented_messages

    def _setup_data(self, data_path, jsons_path):
        # # # loading directly from test file or val file
        if self.datatype.startswith('test'):
            test_path = data_path.replace(".json", "_test.json")
            if self.few_shot:
                test_path = test_path.replace(".json", "_fewshot.json")
            test_data = self._load_json(test_path)
            self.messages = list(test_data.values())
            if self.test_reduced:
                k = min(len(self.messages), 300)
                self.messages = random.sample(list(test_data.values()), k=k)
        elif self.datatype.startswith('valid'):
            valid_path = data_path.replace(".json", "_valid.json")
            if self.few_shot:
                valid_path = valid_path.replace(".json", "_fewshot.json")
            valid_data = self._load_json(valid_path)
            self.messages = list(valid_data.values())
            if self.val_reduced:
                k = min(len(self.messages), 300)
                self.messages = random.sample(list(valid_data.values()), k=k)

            # for simulating low resource setting
            assert (
                self.reduce_train_factor >= 1
            ), f"Reduce train factor must be an integer larger than 1. Provided value: {self.reduce_train_factor}"
            if self.reduce_train_factor > 1:
                n_samples = len(self.messages) // self.reduce_train_factor
                self.messages = random.sample(self.messages, k=n_samples)
        else:
            train_path = data_path.replace(".json", "_train.json")
            if self.few_shot:
                train_path = train_path.replace(".json", "_fewshot.json")
            train_data = self._load_json(train_path)
            self.messages = list(train_data.values())

        if self.just_test:
            self.messages = self.messages[:10]

        if self.data_aug.lower() not in {"nei", "orig"}:
            # create separate samples for the original examples and the paraphrased ones
            unfolded_messages = []
            for msg in self.messages:
                if "orig_context" not in msg:
                    print(msg)
                    continue
                item = {
                    "context": msg["orig_context"],
                    "dial_id": msg["dial_id"],
                    "slots_inf": msg["slots_inf"],
                    "turn_num": msg["turn_num"],
                }

                # make sure that the context is different.
                if msg["orig_context"] != msg["context"]:
                    unfolded_messages.append(item)
                    item_ = item.copy()
                    item_['context'] = msg["context"]
                    unfolded_messages.append(item_)

            self.messages = unfolded_messages

            # verify that the data is in correct form: consecutive examples other than for NEI should have the same labels
            for idx in range(0, len(self.messages), 2):
                orig_label = self.messages[idx]["slots_inf"]
                augmented_label = self.messages[idx + 1]["slots_inf"]
                assert orig_label == augmented_label, print(
                    f"idx: {idx}, {idx+1}\n\t Original example: {self.messages[idx]}\n\tAugmented example: {self.messages[idx+1]}"
                )

        elif self.data_aug.lower() == "nei":
            # dynamically augment examples
            self.messages = self.nei_augment(self.messages)
        # for original data, there is nothing to do

        # shuffle for training only: it's important that validation and test sets are not shuffled for proper invariance scoring
        if self.datatype.startswith('train'):
            random.shuffle(self.messages)

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom--slot_type--slot_val", ... ]
        """
        slots_list = []

        slot_val_conversion = {
            "centre": "center",
            "3-star": "3",
            "2-star": "2",
            "1-star": "1",
            "0-star": "0",
            "4-star": "4",
            "5-star": "5",
        }

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
                if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                    slot_type = slot[1] + " " + slot[2]
                    slot_val = " ".join(slot[3:])
                else:
                    slot_type = slot[1]
                    slot_val = " ".join(slot[2:])
                slot_val = slot_val_conversion.get(slot_val, slot_val)
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
        resp = model_response.get('text')
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

        def add_slot_p_r(type: str):
            for gt_slot in slots_truth:
                self.metrics.add(f"slot_r_{type}", AverageMetric(gt_slot in slots_pred))
                curr_domain = gt_slot.split("--")[0]
                self.metrics.add(
                    f"{curr_domain}/slot_r_{type}", AverageMetric(gt_slot in slots_pred)
                )
            for predicted_slot in slots_pred:
                self.metrics.add(
                    f"slot_p_{type}", AverageMetric(predicted_slot in slots_truth)
                )
                curr_domain = predicted_slot.split("--")[0]
                self.metrics.add(
                    f"{curr_domain}/slot_p_{type}",
                    AverageMetric(predicted_slot in slots_truth),
                )

        def add_hallucination(type: str):
            for gt_slot in slots_truth_named_entity:
                self.metrics.add(
                    f"all_ne/slot_r_{type}", AverageMetric(gt_slot in slots_pred)
                )
                curr_domain = gt_slot.split("--")[0]
                self.metrics.add(
                    f"{curr_domain}_ne/slot_r_{type}",
                    AverageMetric(gt_slot in slots_pred),
                )
            for predicted_slot in slots_pred_named_entity:
                self.metrics.add(
                    f"all_ne/slot_p_{type}",
                    AverageMetric(predicted_slot in slots_truth),
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
                    f"{curr_domain}_ne/slot_p_{type}",
                    AverageMetric(predicted_slot in slots_truth),
                )
                self.metrics.add(
                    f"{curr_domain}_ne/hallucination_{type}",
                    AverageMetric(not (ne.strip() in teacher_action.get("text"))),
                )

                # get combined hallucination
                self.metrics.add(
                    f"all_ne/hallucination_{type}",
                    AverageMetric(not (ne.strip() in teacher_action.get("text"))),
                )

        def add_jga(type: str):
            self.metrics.add(f'jga_{type}', AverageMetric(jga_curr))
            self.metrics.add(
                f"named_entities/jga_{type}",
                AverageMetric(
                    set(slots_truth_named_entity) == set(slots_pred_named_entity)
                ),
            )
            for domain in slots_truth_per_domain:
                if domain in slots_pred_per_domain:
                    self.metrics.add(
                        f"{domain}/jga_{type}",
                        AverageMetric(
                            slots_truth_per_domain[domain]
                            == slots_pred_per_domain[domain]
                        ),
                    )

        jga_curr = set(slots_truth) == set(slots_pred)
        # print out when predictions are wrong
        if jga_curr == False:
            # logging.info(teacher_action)
            tag = "perturbed" if self.flag_compute else "orig"
            # logging.info(f"{tag}\n\tslots_truth: {slots_truth}\n\tslots_pred: {slots_pred}")

        # metrics on original test set
        if self.flag_compute == 0:
            add_jga("original")
            add_slot_p_r("original")
            add_hallucination("original")

        # no need to calculate any other metrics for regular test set
        if self.data_aug == "orig":
            return

        # metrics on the perturbed version of the test set
        if self.flag_compute:
            self.metrics.add(
                f'consistency', AverageMetric(slots_pred == self.slots_pred_prev)
            )
            # the slots should be the same
            # assert slots_truth == self.slots_truth_prev, (slots_truth, self.slots_truth_prev)
            add_jga(type="perturbed")
            add_slot_p_r(type="perturbed")
            add_hallucination("perturbed")

            # conditional metrics (conditioned on the original prediction being correct)
            if self.jga_prev:
                add_jga(type="conditional")
                add_slot_p_r(type="conditional")
                add_hallucination("conditional")

        # combined metrics (original + perturbed)
        add_slot_p_r(type="all")
        add_jga(type="all")
        add_hallucination("all")

        self.jga_prev = jga_curr
        self.slots_truth_prev = slots_truth
        self.slots_pred_prev = slots_pred

        # to switch between computing paraphrase invariance and regular scores
        self.flag_compute = 1 - self.flag_compute

    def num_examples(self):
        # each turn be seen as a individual dialog
        return len(self.messages)

    def num_episodes(self):
        return len(self.messages)

    def get(self, episode_idx, entry_idx=0):
        # log_idx = entry_idx
        entry = self.messages[episode_idx]['context']
        episode_done = True
        action = {
            'id': self.id,
            'text': entry,
            'episode_done': episode_done,
            'labels': [self.messages[episode_idx]['slots_inf']],
            'dial_id': self.messages[episode_idx]['dial_id'],
            'turn_num': self.messages[episode_idx]['turn_num'],
        }

        return action


class DefaultTeacher(MultiWozDSTTeacher):
    """
    Default teacher.
    """

    pass
