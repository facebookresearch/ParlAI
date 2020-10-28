#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
from tqdm import tqdm

# global variables
SYSTEM_TOK = "<system>"
USER_TOK = "<user>"

class Reformat_Multiwoz(object):
    """
    reformat multiwoz (maybe sgd later) into
    utt-to-slots
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_path = os.path.join(self.data_dir, "data.json")

        self.val_path = os.path.join(self.data_dir, "valListFile.json")
        self.test_path = os.path.join(self.data_dir, "testListFile.json")
        self.val_list = self._load_txt(self.val_path)
        self.test_list = self._load_txt(self.test_path)


    def _load_dials(self, data_path=None):
        if data_path is None:
            data_path = self.data_path
        with open(data_path) as df:
            self.dials = json.loads(df.read().lower())


    def _load_txt(self, file_path):
        with open(file_path) as df:
            data=df.read().lower().split("\n")
            data.remove('')
        return data


    def _extract_slots(self, turn):
        # ACCUMULATED dialog states, extracted based on "belief_state"
        slots = []
        for state in turn["belief_state"]:
            if state["act"] == "inform":
                domain = state["slots"][0][0].split("-")[0]
                slot_type = state["slots"][0][0].split("-")[1]
                slot_val  = state["slots"][0][1]
                slots += [domain, slot_type, slot_val+","]
        return " ".join(slots)


    def reformat_dial(self):
        """
        following trade's code for normalizing multiwoz*
        now the data has format:
        file=[{
            "dialogue_idx": dial_id,
            "domains": [dom],
            "dialogue": [
                    {
                        "turn_idx"          : 0,
                        "domain"            : "hotel",
                        "system_transcript" : "system response", # respond to
the utterance in previous turn
                        "transcript"        : "user utterance",
                        "system_act"        : [],
                        "belief_state"      : [{
                                                "slots":[["domain-slot_type","slot_vale"]],
                                                "act":  "inform"
                                                },
                                                ...], # accumulated
                        "turn_label"        :
[["domain-slot_type","slot_vale"],...],    # for current turn
                    },
                    ...
                ],
                ...
            },
            ]
        and output with format like:
        file={
            dial_id-turn_num:
                    {
                        "dial_id"   : dial_id
                        "turn_num"  : 0,
                        "context"   : "User: ... Sys: ... User:...",
                        "slots_inf" : "dom slot_type1 slot_val1, dom slot_type2
...",
                        "response"  : system response as a string,
                    },
            ...
            }
        """
        self.reformat_data_name = "data_trade_reformat.json"
        self.data_trade_proc_path = os.path.join(self.data_dir,
"data_trade.json")
        self._load_dials(data_path = self.data_trade_proc_path)
        self.dials_form = {}
        self.dials_train, self.dials_val, self.dials_test = {}, {}, {}

        for dial in tqdm(self.dials):
            context = []
            turn_form = {}
            for turn in dial["dialogue"]:
                # turn number
                turn_id = turn_form["turn_num"] = turn["turn_idx"]

                # dialog id
                dial_id = turn_form["dial_id"] = dial["dialogue_idx"]
                turn_form["domain"] = turn["domain"]

                if turn["system_transcript"] == "":
                    # for the first turn, there is no system response
                    context.append(USER_TOK + " " + turn["transcript"])
                    # dialog slots
                    turn_form["slots"] = self._extract_slots(turn)
                    continue
                else:
                    # for the rest turns
                    turn_form["response"] = turn["system_transcript"]

                turn_form["context"] = " ".join(context)

                # adding this turn to train/val/test dict
                unique_id = f'{dial_id}-{turn_id}'
                self.dials_form[unique_id] = turn_form.copy()
                if dial_id in self.test_list:
                    self.dials_test[unique_id] = turn_form.copy()
                elif dial_id in self.val_list:
                    self.dials_val[unique_id] = turn_form.copy()
                else:
                    self.dials_train[unique_id] = turn_form.copy()

                # adding current turn to dialog history
                context.append(SYSTEM_TOK + " " + turn["system_transcript"])
                context.append(USER_TOK + " " + turn["transcript"])
                # dialog slots
                turn_form["slots"] = self._extract_slots(turn)

        self.reformat_data_path = os.path.join(self.data_dir,
self.reformat_data_name)
        self.reformat_train_data_path = self.reformat_data_path.replace(".json", "_train.json")
        self.reformat_valid_data_path = self.reformat_data_path.replace(".json", "_valid.json")
        self.reformat_test_data_path = self.reformat_data_path.replace(".json","_test.json")

        with open(self.reformat_data_path, "w") as tf:
            json.dump(self.dials_form, tf, indent=2)
        with open(self.reformat_train_data_path, "w") as tf:
            json.dump(self.dials_train, tf, indent=2)
        with open(self.reformat_valid_data_path, "w") as tf:
            json.dump(self.dials_val, tf, indent=2)
        with open(self.reformat_test_data_path, "w") as tf:
            json.dump(self.dials_test, tf, indent=2)


def reformat_dial(data_dir):
    """
    reformat multiwoz for dialog response generation
    """
    reformat = Reformat_Multiwoz(data_dir)
    reformat.reformat_dial()

