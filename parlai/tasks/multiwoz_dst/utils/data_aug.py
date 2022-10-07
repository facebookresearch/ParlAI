#!/usr/bin/env python3
#
import os, sys, json
import math, random, re
import pdb
from collections import defaultdict, OrderedDict

DOMAINS = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train"]
SLOT_TYPES = [
    "type",
    "area",
    "stars",
    "department",
    "people",
    "food",
    "name",
    "internet",
    "parking",
    "day",
    "pricerange",
    "book stay",
    "book people",
    "book time",
    "book day",
    "destination",
    "leaveat",
    "arriveby",
    "departure",
]
SCRAMBLE_SLOT = [
    "destination",
    "leaveat",
    "arriveby",
    "departure",
    "food",
    "name",
    "time",
    "pricerange",
    "book time",
]


EASY_ERR_SLOT_TYPES = ["destination", "leaveat", "arriveby", "departure"]
NON_CAND_SLOT_TYPES = [
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
CAN_BE_REPLACED = [
    "area",
    "department",
    "food",
    "name",
    "book time",
    "book day",
    "pricerange",
    "destination",
    "leaveat",
    "arriveby",
    "departure",
]


class DATA_AUG(object):
    def __init__(self, data_path, data_dir):
        # augmentation is only applied on training data
        self.data_path = data_path.replace(".json", "_train.json")
        self.data_dir = data_dir
        self.aug_data_path = self.data_path.replace(".json", "_aug_all.json")

        # load original data
        self.old_data = self._load_json(self.data_path)
        # load ontology file
        self.ontology_path = os.path.join(
            self.data_dir, "../../multiwoz_dst/MULTIWOZ2.1/ontology.json"
        )
        self.ontology = self._load_ontology(self.ontology_path)

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def _load_ontology(self, ontology_path):
        """
        load ontology file from multiwoz
        input format: {
                dom - slot_type : [val1, val2, ...],
                ...
        }
        output format: {
            dom: {
                slot_type: [val1, val2, val3, ...]
            }
        }

        """
        with open(self.ontology_path) as ot:
            orig_ontology = json.loads(ot.read().lower())

        ontology = {}
        for dom_type, vals in orig_ontology.items():
            dom, slot_type = dom_type.split("-")
            # format slot type, e.g. "price range" --> "pricerange"
            if slot_type not in SLOT_TYPES:
                if slot_type.replace(" ", "") in SLOT_TYPES:
                    slot_type = slot_type.replace(" ", "")
            if dom not in ontology:
                ontology[dom] = {}
            # remove do not care
            if "do n't care" in vals:
                vals.remove("do n't care")
            ontology[dom][slot_type] = vals
        return ontology

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        [[dom, slot_type, slot_val], ... ]
        """
        slots_list = []

        # # # split according to ","
        str_split = slots_string.split(",")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        str_split = [slot.strip() for slot in str_split]

        for slot_ in str_split:
            slot = slot_.split()
            if len(slot) > 2 and slot[0] in DOMAINS:
                domain = slot[0]
                if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                    slot_type = slot[1] + " " + slot[2]
                    slot_val = " ".join(slot[3:])
                else:
                    slot_type = slot[1]
                    slot_val = " ".join(slot[2:])
                slots_list.append([domain, slot_type, slot_val])
            else:
                pass
                # pmul4204.json-5
                # in slots "taxi arriveby 16:15" --> "taxi arriveby 16,15"
        return slots_list

    def augmentation(self):
        """
        data augmentation by replacing values in both context and slot
        """
        self.new_data = {}
        # num of time to do augmentation
        aug_num = 3

        for n in range(aug_num):
            for dial_id, turn in self.old_data.items():
                # slots string to slots list [[dom, slot_type, slot_val], [], ...]
                slots_list = self._extract_slot_from_string(turn["slots_inf"])
                # to keep old_data unchanged
                turn = turn.copy()

                flag = 0
                for slot in slots_list:
                    if slot[1] in EASY_ERR_SLOT_TYPES:
                        # replace slot value
                        old_val = slot[2]
                        new_val = random.choice(self.ontology[slot[0]][slot[1]])
                        slot[2] = new_val
                        # replace all the token in context
                        # TODO: two slot type share the same slot value (train/taxi leaveat/arriveby -- restaurant book time)
                        turn["context"] = re.sub(old_val, new_val, turn["context"])
                        # mark turn has been changed
                        flag = 1

                # if flag == 1:
                turn["slots_inf"] = ", ".join([" ".join(slot) for slot in slots_list])
                turn["slots_err"] = ""
                self.new_data[dial_id + "-" + str(n)] = turn
            # pdb.set_trace()

        # merge aug data and original data
        self.new_data.update(self.old_data)
        # save new data
        with open(self.aug_data_path, "w") as tf:
            json.dump(self.new_data, tf, indent=2)


def data_aug(data_path, data_dir):
    data_aug = DATA_AUG(data_path, data_dir)
    data_aug.augmentation()


class DATA_SCARMBLE(object):
    def __init__(self, data_path, data_dir):
        # augmentation is only applied on training data
        self.data_path = data_path.replace(".json", "_train.json")
        self.data_dir = data_dir
        self.scr_data_path = self.data_path.replace(".json", "_scr_all.json")

        # load original data
        self.old_data = self._load_json(self.data_path)

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        [[dom, slot_type, slot_val], ... ]
        """
        slots_list = []

        # # # split according to ","
        str_split = slots_string.split(",")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        str_split = [slot.strip() for slot in str_split]

        for slot_ in str_split:
            slot = slot_.split()
            if len(slot) > 2 and slot[0] in DOMAINS:
                domain = slot[0]
                if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                    slot_type = slot[1] + " " + slot[2]
                    slot_val = " ".join(slot[3:])
                else:
                    slot_type = slot[1]
                    slot_val = " ".join(slot[2:])
                if slot_val != "none":
                    slots_list.append([domain, slot_type, slot_val])
            else:
                pass
                # pmul4204.json-5
                # in slots "taxi arriveby 16:15" --> "taxi arriveby 16,15"
        return slots_list

    def scramble(self):
        """
        data modification by scramble slot values in both context and slot
        """
        self.scr_data = {}
        # num of time to do augmentation
        aug_num = 1

        for n in range(aug_num):
            for dial_id, turn in self.old_data.items():
                # slots string to slots list [[dom, slot_type, slot_val], [], ...]
                slots_list = self._extract_slot_from_string(turn["slots_inf"])
                # to keep old_data unchanged
                turn = turn.copy()
                # list of all slot values
                slot_val_list = [slot[-1] for slot in slots_list]
                # sample half of the slots for scramble
                cands = random.sample(slots_list, k=int(len(slots_list)))
                for slot in cands:
                    if slot[1] in SCRAMBLE_SLOT:
                        # replace slot value
                        old_val = slot[2]
                        slot_val_list_copy = slot_val_list[:]
                        slot_val_list_copy.remove(old_val)
                        if old_val not in " ".join(
                            slot_val_list_copy
                        ):  # and random.choice([0,1]):
                            new_val = list(old_val)
                            random.shuffle(new_val)
                            slot[2] = "".join(new_val)
                            # replace all the token in context
                            turn["context"] = re.sub(old_val, slot[2], turn["context"])
                            # mark turn has been changed
                # pdb.set_trace()
                turn["slots_inf"] = ", ".join([" ".join(slot) for slot in slots_list])
                turn["slots_err"] = ""
                self.scr_data[dial_id + "-" + str(n)] = turn
            # pdb.set_trace()

        # save new data
        with open(self.scr_data_path, "w") as tf:
            json.dump(self.scr_data, tf, indent=2)


def data_scr(data_path, data_dir):
    data_scr = DATA_SCARMBLE(data_path, data_dir)
    data_scr.scramble()


def main():
    pass


if __name__ == "__main__":
    main()
