#!/usr/bin/env python3
#
import os, sys, json
import math, argparse, random, re
from tqdm import tqdm
import pdb


class Reformat_SGD(object):
    def __init__(self, data_dir=None):
        # self.data_dir = "/checkpoint/kunqian/dstc8-schema-guided-dialogue/"
        self.data_dir = data_dir
        self.data_path = os.path.join(self.data_dir, "data.json")
        # self.reformat_data_path = os.path.join(self.data_dir, "data_reformat.json")

    def _load_data(self, data_type):
        self.data_type_path = self.data_path.replace(".json", "_" + data_type + ".json")
        self.dials = {}

        if os.path.exists(self.data_type_path) and os.path.isfile(self.data_type_path):
            with open(self.data_type_path) as df:
                self.dials = json.loads(df.read().lower())
        else:
            data_dir = os.path.join(self.data_dir, data_type)
            sys.stdout.write("Extracting " + data_type + " data .... \n")
            for filename in tqdm(os.listdir(data_dir)):
                if "dialog" in filename:  # exclude schema.json
                    file_path = os.path.join(data_dir, filename)
                else:
                    continue
                if os.path.isfile(file_path):
                    data_json = open(file_path, "r", encoding="utf-8")
                    data_in_file = json.loads(data_json.read().lower())
                    data_json.close()
                else:
                    continue

                for dial in data_in_file:
                    dial_id = data_type + "_" + dial["dialogue_id"]
                    if dial_id in self.dials:
                        pdb.set_trace()
                    self.dials[dial_id] = dial

            with open(self.data_type_path, "w") as df:
                json.dump(self.dials, df, indent=2)

    def reformat_slots_sgd(self):
        """
        file={
            dial_id: [
                    {
                        "turn_num": 0,
                        "utt": user utterance,
                        "slots_inf": slot sequence ("dom slot_type1 slot_val1, dom slot_type2 ..."),
                        # "slots_err": slot sequence ("dom slot_type1 slot_type2, ..."),
                        "context" : "User: ... Sys: ..."
                    },
                    ...
                ],
                ...
            }
        """
        for data_type in ["train", "dev", "test"]:
            self._load_data(data_type)
            self.dials_form = {}
            sys.stdout.write("Reformating " + data_type + " data .... \n")
            for dial_id, dial in tqdm(self.dials.items()):
                # self.dials_form[dial_id] = []
                turn_form = {}
                turn_num = 0
                bspan = {}  # {dom:{slot_type:val, ...}, ...}
                context = []
                for turn in dial["turns"]:
                    # turn number
                    turn_form["turn_num"] = turn_num

                    if turn["speaker"] == "system":
                        # turn_form['sys'] = self._tokenize_punc(turn['utterance'])
                        context.append(
                            "<system> " + self._tokenize_punc(turn["utterance"])
                        )

                    if turn["speaker"] == "user":
                        context.append(
                            "<user> " + self._tokenize_punc(turn["utterance"])
                        )

                        # dialog history/context
                        turn_form["context"] = " ".join(context)

                        # belief span
                        turn_form["slots_inf"], bspan = self._extract_slots(
                            bspan,
                            turn["frames"],
                            turn["utterance"],
                            turn_form["context"],
                        )

                        # # user utterance
                        turn_form["utt"] = self._tokenize_punc(turn["utterance"])

                    # let each episode just be a single context-slot pair
                    if "utt" in turn_form:
                        self.dials_form[dial_id + f"-{turn_num}"] = turn_form
                        turn_form = {}
                        turn_num += 1

            # save reformatted dialogs
            self.reformat_data_path = os.path.join(
                self.data_dir, "data_reformat_" + data_type + ".json"
            )
            with open(self.reformat_data_path, "w") as tf:
                json.dump(self.dials_form, tf, indent=2)

    def _extract_slots(self, bspan, frames, utt, context):
        """
        Input:
            bspan = {
                        dom:{
                            slot_type : slot_val,
                            ...
                            },
                        ...
                    }

            frames = [
                        {
                            "service" : domain,
                            "slots"   : [
                                            {
                                                "exclusive_end" : idx,
                                                "slot" : slot_type,
                                                "start": idx
                                            },
                                            ...
                                        ],
                            "state"   : {"slot_values": {slot_type:[slot_val, ...], ...}}
                        },
                        ...
                     ]

        Notice that:
            frames["slots"] contains only non-categorical slots, while
            frames["state"] contains both non-categorical and categorical slots,
            but it may contains multiple slot_vals for non-categorical slots.
            Therefore, we extract non-categorical slots based on frames["slots"]
            and extract categorical slots based on frames["state"]

        Output:
            formalize dialog states into string like:
                "restaurant area centre, restaurant pricerange cheap, ..."
        """

        dial_state = []
        for frame in frames:
            # extract Non-Categorical slots, based on frame["slots"]
            domain = frame["service"]
            if domain not in bspan:
                bspan[domain] = {}
            for slot in frame["slots"]:
                slot_type = slot["slot"]
                slot_val = utt[slot["start"] : slot["exclusive_end"]] + ","
                bspan[domain][slot_type] = slot_val

            # extract Categorical slots, based on frame["state"]
            for slot_type in frame["state"]["slot_values"]:
                if slot_type not in bspan[domain]:
                    if len(set(frame["state"]["slot_values"][slot_type])) == 1:
                        bspan[domain][slot_type] = (
                            frame["state"]["slot_values"][slot_type][0] + ","
                        )
                    elif len(set(frame["state"]["slot_values"][slot_type])) > 1:
                        count = 0
                        slot_val_list = sorted(
                            list(set(frame["state"]["slot_values"][slot_type])),
                            key=lambda i: len(i),
                        )

                        for slot_val in slot_val_list:
                            # shown up in previous utt/bspan, referring to slots in other domain
                            if slot_val in utt:
                                bspan[domain][slot_type] = slot_val + ","
                                count += 1
                            # if count > 1:
                            #     print("utt contains non-categorical slot vals: ", slot_type)
                            #     print(frame["state"]["slot_values"][slot_type])
                            #     print(dial_id)
                            #     pdb.set_trace()

                        if count == 0:
                            for slot_val in slot_val_list:
                                # shown up in previous utt/bspan, referring to slots in other domain
                                if slot_val + "," in self._extract_all_slot_vals(bspan):
                                    bspan[domain][slot_type] = slot_val + ","
                                    count += 1
                            # if count > 1:
                            #     print("multiple non-categorical slot vals: ", slot_type)
                            #     print(frame["state"]["slot_values"][slot_type])
                            #     print(dial_id)
                            #     pdb.set_trace()

                        if count == 0:
                            for slot_val in slot_val_list:
                                if (
                                    slot_val in context
                                    and self._find_speaker(slot_val, context) == "user"
                                ):
                                    bspan[domain][slot_type] = slot_val + ","
                                    count += 1

                            # if count > 1:
                            #     print("non-categorical slot vals in context user utt: ", slot_type)
                            #     print(frame["state"]["slot_values"][slot_type])
                            #     print(dial_id)
                            #     pdb.set_trace()

                        if count == 0:
                            for slot_val in slot_val_list:
                                if slot_val in context:
                                    bspan[domain][slot_type] = slot_val + ","
                                    count += 1

                    # elif len(set(frame["state"]["slot_values"][slot_type])) == 0:
                    #     print("non-categorical slots with no value: ", slot_type)
                    #     print(frame["state"]["slot_values"][slot_type])
                    #     print(dial_id)
                    #     pdb.set_trace()

        # rewrite all the slots:
        for domain in bspan:
            for slot_type in bspan[domain]:
                dial_state += [domain, slot_type, bspan[domain][slot_type]]

        return " ".join(dial_state), bspan

    def _extract_all_slot_vals(self, bspan):
        """
        Input:
            bspan = {
                        dom:{
                            slot_type : slot_val,
                            ...
                            },
                        ...
                    }
        To extract all the slot_val in this bspan.
        Return in a form of list
        """
        slot_val_list = []
        for dom in bspan:
            slot_val_list += list(bspan[dom].values())
        return slot_val_list

    def _find_speaker(self, slot_val, context):
        """
        assume the slot_val exists in the context, try
        to find out who is the first speaker mentioned
        the slot_val
        context = "User: ... Sys: ... User: ... "
        """
        for utt in context.split("user:"):
            if slot_val in utt:
                if slot_val in utt.split("sys:")[0]:
                    return "user"
                else:
                    return "sys"

    def _tokenize_punc(self, utt):
        """ """
        # corner_case = ['\.\.+', '!\.', '\$\.']
        # for case in corner_case:
        #     utt = re.sub(case, ' .', utt)

        # utt = re.sub('(?<! )\?+', ' ?', utt)
        # utt = re.sub('(?<! )!+', ' !', utt)
        # utt = re.sub('(?<! ),+(?= )', ' ,', utt)

        # utt = re.sub('(?<=[a-z0-9])>(?=$)', ' .', utt)
        # utt = re.sub('(\?>|>\?)', ' ?', utt)

        # # utt = re.sub('(?<=[a-z0-9\]])(,|\.)$', ' .', utt)
        # utt = re.sub('(?<!\.[a-z])(?<! )\.(?= )', ' .', utt)
        # utt = re.sub('(?<!\.[a-z])(?<! )(,|\.)$', ' .', utt)
        # utt = re.sub('(?<! )!$', ' !', utt)
        # utt = re.sub('(?<! )\?$', ' ?', utt)

        # utt = self.clean_text(utt)
        return utt

    def clean_text(self, text):
        text = text.strip()
        text = text.lower()
        text = text.replace("’", "'")
        text = text.replace("‘", "'")
        text = text.replace(";", ",")
        text = text.replace('"', " ")
        text = text.replace("/", " and ")
        text = text.replace("don't", "do n't")
        text = clean_time(text)
        baddata = {
            r"c\.b (\d), (\d) ([a-z])\.([a-z])": r"cb\1\2\3\4",
            "c.b. 1 7 d.y": "cb17dy",
            "c.b.1 7 d.y": "cb17dy",
            "c.b 25, 9 a.q": "cb259aq",
            "isc.b 25, 9 a.q": "is cb259aq",
            "c.b2, 1 u.f": "cb21uf",
            "c.b 1,2 q.a": "cb12qa",
            "0-122-336-5664": "01223365664",
            "postcodecb21rs": "postcode cb21rs",
            r"i\.d": "id",
            " i d ": "id",
            "Telephone:01223358966": "Telephone: 01223358966",
            "depature": "departure",
            "depearting": "departing",
            "-type": " type",
            r"b[\s]?&[\s]?b": "bed and breakfast",
            "b and b": "bed and breakfast",
            r"guesthouse[s]?": "guest house",
            r"swimmingpool[s]?": "swimming pool",
            "wo n't": "will not",
            " 'd ": " would ",
            " 'm ": " am ",
            " 're' ": " are ",
            " 'll' ": " will ",
            " 've ": " have ",
            r"^\'": "",
            r"\'$": "",
        }
        for tmpl, good in baddata.items():
            text = re.sub(tmpl, good, text)

        text = re.sub(
            r"([a-zT]+)\.([a-z])", r"\1 . \2", text
        )  # 'abc.xyz' -> 'abc . xyz'
        text = re.sub(r"(\w+)\.\.? ", r"\1 . ", text)  # if 'abc. ' -> 'abc . '
        return text


def reformat_parlai(data_dir, force_reformat=False):
    # args = Parse_args()
    # args.data_dir = data_dir
    if (
        os.path.exists(os.path.join(data_dir, "data_reformat_train.json"))
        and os.path.exists(os.path.join(data_dir, "data_reformat_dev.json"))
        and os.path.exists(os.path.join(data_dir, "data_reformat_test.json"))
        and not force_reformat
    ):
        pass
        # print("already reformat data before, skipping this time ...")
    else:
        reformat = Reformat_SGD(data_dir)
        reformat.reformat_slots_sgd()
