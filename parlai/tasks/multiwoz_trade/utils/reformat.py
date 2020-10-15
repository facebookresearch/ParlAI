#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os, sys, json
import math, argparse, random, re
from tqdm import tqdm
import pdb

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
        self.val_list = self.load_txt(self.val_path)
        self.test_list = self.load_txt(self.test_path)

    def load_dials(self, data_path=None):
        if data_path is None:
            data_path = self.data_path
        with open(data_path) as df:
            self.dials = json.loads(df.read().lower())
    
    def load_txt(self, file_path):
        with open(file_path) as df:
            data=df.read().lower().split("\n")
            data.remove('')
        return data

    def save_dials(self):
        with open(self.reformat_data_path, "w") as tf:
            json.dump(self.dials_form, tf, indent=2)
        print(f"Saved reformatted data to {self.reformat_data_path} ...")

    def reformat_dst(self):
        """
        following trade's code for normalizing multiwoz*
        now the data has format:
        file=[{
            "dialogue_idx": dial_id,
            "domains": [dom],
            "dialogue": [
                    {
                        "turn_idx": 0,
                        "domain": "hotel",
                        "system_transcript": "system response",
                        "transcript": "user utterance",
                        "system_act": [],
                        "belief_state": [{
                            "slots":[["domain-slot_type","slot_vale"]],
                            "act":  "inform"
                        }, ...], # accumulated
                        "turn_label": [["domain-slot_type","slot_vale"],...],    # for current turn
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
                        "dial_id": dial_id
                        "turn_num": 0,
                        "slots_inf": slot sequence ("dom slot_type1 slot_val1, dom slot_type2 ..."),\
                        "context" : "User: ... Sys: ... User:..."
                    },
            ...
            }
        """
        self.reformat_data_name = "data_reformat_trade_dst.json"
        self.reformat_data_path = os.path.join(self.data_dir, self.reformat_data_name)
        self.data_trade_proc_path = os.path.join(self.data_dir, "dials_trade.json")
        self.load_dials(data_path = self.data_trade_proc_path)
        self.dials_form = {}
        self.dials_train, self.dials_val, self.dials_test = {}, {}, {}

        for dial in tqdm(self.dials):
            # self.dials_form[dial["dialogue_idx"]] = []
            context = []
            for turn in dial["dialogue"]:
                turn_form = {}
                # # # turn number
                turn_form["turn_num"] = turn["turn_idx"]

                # # # # dialog id
                turn_form["dial_id"] = dial["dialogue_idx"]
                
                # # # slots/dialog states
                slots_inf = []
                
                # # # ACCUMULATED dialog states, extracted based on "belief_state"
                for state in turn["belief_state"]:
                    if state["act"] == "inform":
                        domain = state["slots"][0][0].split("-")[0]
                        slot_type = state["slots"][0][0].split("-")[1]
                        slot_val  = state["slots"][0][1]
                        slots_inf += [domain, slot_type, slot_val+","]

                turn_form["slots_inf"] = " ".join(slots_inf)

                # # # dialog history
                if turn["system_transcript"] != "":
                    context.append("<system> " + turn["system_transcript"])

                # # # adding current turn to dialog history
                context.append("<user> " + turn["transcript"])

                turn_form["context"] = " ".join(context)

                self.dials_form[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form
                if dial["dialogue_idx"] in self.test_list:
                    self.dials_test[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form
                elif dial["dialogue_idx"] in self.val_list:
                    self.dials_val[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form
                else:
                    self.dials_train[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form

        self.reformat_train_data_path = self.reformat_data_path.replace(".json", "_train.json")
        self.reformat_valid_data_path = self.reformat_data_path.replace(".json", "_valid.json")
        self.reformat_test_data_path = self.reformat_data_path.replace(".json", "_test.json")

        with open(self.reformat_train_data_path, "w") as tf:
            json.dump(self.dials_train, tf, indent=2)
        with open(self.reformat_valid_data_path, "w") as tf:
            json.dump(self.dials_val, tf, indent=2)
        with open(self.reformat_test_data_path, "w") as tf:
            json.dump(self.dials_test, tf, indent=2)
        with open(self.reformat_data_path, "w") as tf:
            json.dump(self.dials_form, tf, indent=2)

    def reformat_dial(self):
        """
        following trade's code for normalizing multiwoz*
        now the data has format:
        file=[{
            "dialogue_idx": dial_id,
            "domains": [dom],
            "dialogue": [
                    {
                        "turn_idx": 0,
                        "domain": "hotel",
                        "system_transcript": "system response", # respond to the utterance in previous turn
                        "transcript": "user utterance",
                        "system_act": [],
                        "belief_state": [{
                            "slots":[["domain-slot_type","slot_vale"]],
                            "act":  "inform"
                        }, ...], # accumulated
                        "turn_label": [["domain-slot_type","slot_vale"],...],    # for current turn
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
                        "dial_id": dial_id
                        "turn_num": 0,
                        "response": system response,
                        "context" : "User: ... Sys: ... User:..."
                    },
            ...
            }
        """
        self.reformat_data_name = "data_reformat_trade_dial.json"
        self.reformat_data_path = os.path.join(self.data_dir, self.reformat_data_name)
        self.data_trade_proc_path = os.path.join(self.data_dir, "dials_trade.json")
        self.load_dials(data_path = self.data_trade_proc_path)
        self.dials_form = {}
        self.dials_train, self.dials_val, self.dials_test = {}, {}, {}

        for dial in tqdm(self.dials):
            # self.dials_form[dial["dialogue_idx"]] = []
            context = []
            for turn in dial["dialogue"]:
                turn_form = {}
                # # # turn number
                turn_form["turn_num"] = turn["turn_idx"]

                # # # # dialog id
                turn_form["dial_id"] = dial["dialogue_idx"]
                

                if turn["system_transcript"] == "":
                    # for the first turn, there is no system response
                    context.append("<user> " + turn["transcript"])
                    continue
                else:
                    # for the rest turns
                    turn_form["response"] = turn["system_transcript"]

                turn_form["context"] = " ".join(context)

                self.dials_form[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form
                if dial["dialogue_idx"] in self.test_list:
                    self.dials_test[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form
                elif dial["dialogue_idx"] in self.val_list:
                    self.dials_val[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form
                else:
                    self.dials_train[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form

                # # # adding current turn to dialog history
                if turn["system_transcript"] != "":
                    context.append("<system> " + turn["system_transcript"])
                context.append("<user> " + turn["transcript"])

        self.reformat_train_data_path = self.reformat_data_path.replace(".json", "_train.json")
        self.reformat_valid_data_path = self.reformat_data_path.replace(".json", "_valid.json")
        self.reformat_test_data_path = self.reformat_data_path.replace(".json", "_test.json")

        with open(self.reformat_train_data_path, "w") as tf:
            json.dump(self.dials_train, tf, indent=2)
        with open(self.reformat_valid_data_path, "w") as tf:
            json.dump(self.dials_val, tf, indent=2)
        with open(self.reformat_test_data_path, "w") as tf:
            json.dump(self.dials_test, tf, indent=2)
        with open(self.reformat_data_path, "w") as tf:
            json.dump(self.dials_form, tf, indent=2)

def reformat_dst(data_dir):
    reformat = Reformat_Multiwoz(data_dir)
    reformat.reformat_dst()

def reformat_dial(data_dir):
    reformat = Reformat_Multiwoz(data_dir)
    reformat.reformat_dial()

def Parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",   default="multiwoz")
    parser.add_argument(      "--reformat_data_name", default=None)
    parser.add_argument(      "--save_dial", default=True, type=bool)
    parser.add_argument(      "--data_dir", default=None)
    args = parser.parse_args()
    return args

def main():
    args = Parse_args()
    reformat = Reformat_Multiwoz(args)
    if args.reformat_data_name is not None:
        reformat.reformat_data_path = os.path.join(reformat.data_dir, args.reformat_data_name)
    reformat.reformat_dst()
    

if __name__ == "__main__":
    main()