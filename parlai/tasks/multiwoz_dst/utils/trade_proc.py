# -*- coding: utf-8 -*-
import copy
import json
import os
import re
import shutil
import urllib.request
from collections import OrderedDict
from io import BytesIO
from zipfile import ZipFile
import difflib
import numpy as np
from loguru import logger

np.set_printoptions(precision=3)

np.random.seed(2)


"""
Most of the codes are from https://github.com/budzianowski/multiwoz
"""
PARLAI_DIR = os.environ.get("PARLAI_DIR", "ParlAI")

logger.info(PARLAI_DIR)

# GLOBAL VARIABLES
DICT_SIZE = 400
MAX_LENGTH = 100
IGNORE_KEYS_IN_GOAL = ["eod", "topic", "messageLen", "message"]
mapping_fn = os.path.join(PARLAI_DIR, "parlai/tasks/multiwoz_dst/utils/mapping.pair")
logger.info(mapping_fn)

fin = open(mapping_fn, "r")
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace("\n", "").split("\t")
    replacements.append((" " + tok_from + " ", " " + tok_to + " "))

# class TradePreprocess(object):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if (
            sidx + 1 < len(text)
            and re.match("[0-9]", text[sidx - 1])
            and re.match("[0-9]", text[sidx + 1])
        ):
            sidx += 1
            continue
        if text[sidx - 1] != " ":
            text = text[:sidx] + " " + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != " ":
            text = text[: sidx + 1] + " " + text[sidx + 1 :]
        sidx += 1
    return text


def normalize(text, clean_value=True):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r"^\s*|\s*$", "", text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    if clean_value:
        # normalize phone number
        ms = re.findall("\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})", text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == "(":
                    sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], "".join(m))

        # normalize postcode
        ms = re.findall(
            "([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})",
            text,
        )
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub("[,\. ]", "", m) + text[eidx:]

    # weird unicode bug
    text = re.sub("(\u2018|\u2019)", "'", text)

    if clean_value:
        # replace time and and price
        text = re.sub(timepat, " [value_time] ", text)
        text = re.sub(pricepat, " [value_price] ", text)
        # text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(";", ",")
    text = re.sub("$\/", "", text)
    text = text.replace("/", " and ")

    # replace other special characters
    text = text.replace("-", " ")
    text = re.sub('["\<>@\(\)]', "", text)  # remove

    # insert white space before and after tokens:
    for token in ["?", ".", ",", "!"]:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace("'s", text)

    # replace it's, does't, you'd ... etc
    text = re.sub("^'", "", text)
    text = re.sub("'$", "", text)
    text = re.sub("'\s", " ", text)
    text = re.sub("\s'", " ", text)
    for fromx, tox in replacements:
        text = " " + text + " "
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(" +", " ", text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match("^\d+$", tokens[i]) and re.match("\d+$", tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = " ".join(tokens)

    return text


def fixDelex(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    try:
        turn = data2[filename.strip(".json")][str(idx_acts)]
    except:
        return data

    if not isinstance(turn, str):  # and not isinstance(turn, unicode):
        for k, act in turn.items():
            if "Attraction" in k:
                if "restaurant_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "restaurant", "attraction"
                    )
                if "hotel_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "hotel", "attraction"
                    )
            if "Hotel" in k:
                if "attraction_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "attraction", "hotel"
                    )
                if "restaurant_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "restaurant", "hotel"
                    )
            if "Restaurant" in k:
                if "attraction_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "attraction", "restaurant"
                    )
                if "hotel_" in data["log"][idx]["text"]:
                    data["log"][idx]["text"] = data["log"][idx]["text"].replace(
                        "hotel", "restaurant"
                    )

    return data


def getDialogueAct(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    acts = []
    try:
        turn = data2[filename.strip(".json")][str(idx_acts)]
    except:
        return acts

    if not isinstance(turn, str):  # and not isinstance(turn, unicode):
        for k in turn.keys():
            # temp = [k.split('-')[0].lower(), k.split('-')[1].lower()]
            # for a in turn[k]:
            #     acts.append(temp + [a[0].lower()])

            if k.split("-")[1].lower() == "request":
                for a in turn[k]:
                    acts.append(a[0].lower())
            elif k.split("-")[1].lower() == "inform":
                for a in turn[k]:
                    acts.append([a[0].lower(), normalize(a[1].lower(), False)])

    return acts


def get_summary_bstate(bstate, get_domain=False):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [
        "taxi",
        "restaurant",
        "hospital",
        "hotel",
        "attraction",
        "train",
        "police",
    ]
    summary_bstate = []
    summary_bvalue = []
    active_domain = []
    for domain in domains:
        domain_active = False

        booking = []
        # print(domain,len(bstate[domain]['book'].keys()))
        for slot in sorted(bstate[domain]["book"].keys()):
            if slot == "booked":
                if len(bstate[domain]["book"]["booked"]) != 0:
                    booking.append(1)
                    # summary_bvalue.append("book {} {}:{}".format(domain, slot, "Yes"))
                else:
                    booking.append(0)
            else:
                if bstate[domain]["book"][slot] != "":
                    booking.append(1)
                    summary_bvalue.append(
                        [
                            "{}-book {}".format(domain, slot.strip().lower()),
                            normalize(
                                bstate[domain]["book"][slot].strip().lower(), False
                            ),
                        ]
                    )  # (["book", domain, slot, bstate[domain]['book'][slot]])
                else:
                    booking.append(0)
        if domain == "train":
            if "people" not in bstate[domain]["book"].keys():
                booking.append(0)
            if "ticket" not in bstate[domain]["book"].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]["semi"]:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]["semi"][slot] == "not mentioned":
                slot_enc[0] = 1
            elif bstate[domain]["semi"][slot] in [
                "dont care",
                "dontcare",
                "don't care",
                "do not care",
            ]:
                slot_enc[1] = 1
                summary_bvalue.append(
                    ["{}-{}".format(domain, slot.strip().lower()), "dontcare"]
                )  # (["semi", domain, slot, "dontcare"])
            elif bstate[domain]["semi"][slot]:
                summary_bvalue.append(
                    [
                        "{}-{}".format(domain, slot.strip().lower()),
                        normalize(bstate[domain]["semi"][slot].strip().lower(), False),
                    ]
                )  # (["semi", domain, slot, bstate[domain]['semi'][slot]])
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
            active_domain.append(domain)
        else:
            summary_bstate += [0]

    # print(len(summary_bstate))
    assert len(summary_bstate) == 94
    if get_domain:
        return active_domain
    else:
        return summary_bstate, summary_bvalue


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d["log"]) % 2 != 0:
        # print path
        print("odd # of turns")
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp["goal"] = d["goal"]  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    # last_bvs = []
    for i in range(len(d["log"])):
        if len(d["log"][i]["text"].split()) > maxlen:
            print("too long")
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            text = d["log"][i]["text"]
            if not is_ascii(text):
                print("usr turn not ascii")
                return None
            usr_turns.append(d["log"][i])
        else:  # sys turn
            text = d["log"][i]["text"]
            if not is_ascii(text):
                print("sys turn not ascii")
                return None
            belief_summary, belief_value_summary = get_summary_bstate(
                d["log"][i]["metadata"]
            )
            d["log"][i]["belief_summary"] = str(belief_summary)
            d["log"][i]["belief_value_summary"] = belief_value_summary
            sys_turns.append(d["log"][i])
    d_pp["usr_log"] = usr_turns
    d_pp["sys_log"] = sys_turns

    return d_pp


def get_dial(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t["text"] for t in d_orig["usr_log"]]
    sys = [t["text"] for t in d_orig["sys_log"]]
    sys_a = [t["dialogue_acts"] for t in d_orig["sys_log"]]
    bvs = [t["belief_value_summary"] for t in d_orig["sys_log"]]
    domain = [t["domain"] for t in d_orig["usr_log"]]
    for item in zip(usr, sys, sys_a, domain, bvs):
        dial.append(
            {
                "usr": item[0],
                "sys": item[1],
                "sys_a": item[2],
                "domain": item[3],
                "bvs": item[4],
            }
        )
    return dial


def loadData():
    data_url = "data/multi-woz/data.json"
    dataset_url = "https://www.repository.cam.ac.uk/bitstream/handle/1810/280608/MULTIWOZ2.zip?sequence=3&isAllowed=y"
    if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data/multi-woz")

    if not os.path.exists(data_url):
        # print("Downloading and unzipping the MultiWOZ dataset")
        # resp = urllib.request.urlopen(dataset_url)
        # zip_ref = ZipFile(BytesIO(resp.read()))
        # zip_ref.extractall("data/multi-woz")
        # zip_ref.close()
        shutil.copy("data/multi-woz/2.1/data.json", "data/multi-woz/")
        shutil.copy("data/multi-woz/2.1/valListFile.json", "data/multi-woz/")
        shutil.copy("data/multi-woz/2.1/testListFile.json", "data/multi-woz/")
        shutil.copy("data/multi-woz/2.1/dialogue_acts.json", "data/multi-woz/")


def getDomain(idx, log, domains, last_domain):
    if idx == 1:
        active_domains = get_summary_bstate(log[idx]["metadata"], True)
        crnt_doms = active_domains[0] if len(active_domains) != 0 else domains[0]
        return crnt_doms
    else:
        ds_diff = get_ds_diff(log[idx - 2]["metadata"], log[idx]["metadata"])
        if len(ds_diff.keys()) == 0:  # no clues from dialog states
            crnt_doms = last_domain
        else:
            crnt_doms = list(ds_diff.keys())
        # print(crnt_doms)
        return crnt_doms[0]  # How about multiple domains in one sentence senario ?


def get_ds_diff(prev_d, crnt_d):
    diff = {}
    # Sometimes, metadata is an empty dictionary, bug?
    if not prev_d or not crnt_d:
        return diff

    for ((k1, v1), (k2, v2)) in zip(prev_d.items(), crnt_d.items()):
        assert k1 == k2
        if v1 != v2:  # updated
            diff[k2] = v2
    return diff


def createData(data_dir):
    # download the data
    # loadData()

    # create dictionary of delexicalied values that then we will search against, order matters here!
    # dic = delexicalize.prepareSlotValuesIndependent()
    delex_data = {}

    fin1 = open(os.path.join(data_dir, "data.json"), "r")
    data = json.load(fin1)

    fin2 = open(os.path.join(data_dir, "dialogue_acts.json"), "r")
    data2 = json.load(fin2)

    for didx, dialogue_name in enumerate(data):

        dialogue = data[dialogue_name]

        domains = []
        for dom_k, dom_v in dialogue["goal"].items():
            if (
                dom_v and dom_k not in IGNORE_KEYS_IN_GOAL
            ):  # check whether contains some goal entities
                domains.append(dom_k)

        idx_acts = 1
        last_domain, last_slot_fill = "", []
        for idx, turn in enumerate(dialogue["log"]):
            # normalization, split and delexicalization of the sentence
            origin_text = normalize(turn["text"], False)
            # origin_text = delexicalize.markEntity(origin_text, dic)
            dialogue["log"][idx]["text"] = origin_text

            if idx % 2 == 1:  # if it's a system turn

                cur_domain = getDomain(idx, dialogue["log"], domains, last_domain)
                last_domain = [cur_domain]

                dialogue["log"][idx - 1]["domain"] = cur_domain
                dialogue["log"][idx]["dialogue_acts"] = getDialogueAct(
                    dialogue_name, dialogue, data2, idx, idx_acts
                )
                idx_acts += 1

            # FIXING delexicalization:
            dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)

        delex_data[dialogue_name] = dialogue

        # if didx > 10:
        #     break

    # with open('data/multi-woz/woz2like_data.json', 'w') as outfile:
    #     json.dump(delex_data, outfile)
    # self.delex_data = delex_data
    return delex_data


def SaveData(data, data_dir):
    dials = []
    count = 0
    # data = self.delex_data
    for dialogue_name in data:
        # print dialogue_name
        dial_item = data[dialogue_name]
        domains = []
        for dom_k, dom_v in dial_item["goal"].items():
            if (
                dom_v and dom_k not in IGNORE_KEYS_IN_GOAL
            ):  # check whether contains some goal entities
                domains.append(dom_k)

        dial = get_dial(data[dialogue_name])
        if dial:
            dialogue = {}
            dialogue["dialogue_idx"] = dialogue_name
            dialogue["domains"] = list(
                set(domains)
            )  # list(set([d['domain'] for d in dial]))
            last_bs = []
            dialogue["dialogue"] = []

            for turn_i, turn in enumerate(dial):
                # usr, usr_o, sys, sys_o, sys_a, domain
                turn_dialog = {}
                turn_dialog["system_transcript"] = (
                    dial[turn_i - 1]["sys"] if turn_i > 0 else ""
                )
                turn_dialog["turn_idx"] = turn_i
                turn_dialog["belief_state"] = [
                    {"slots": [s], "act": "inform"} for s in turn["bvs"]
                ]
                turn_dialog["turn_label"] = [
                    bs["slots"][0]
                    for bs in turn_dialog["belief_state"]
                    if bs not in last_bs
                ]
                turn_dialog["transcript"] = turn["usr"]
                turn_dialog["system_acts"] = (
                    dial[turn_i - 1]["sys_a"] if turn_i > 0 else []
                )
                turn_dialog["domain"] = turn["domain"]
                last_bs = turn_dialog["belief_state"]
                dialogue["dialogue"].append(turn_dialog)

            dials.append(dialogue)
            count += 1

        else:
            print(dialogue_name)

    # save all dialogues
    with open(os.path.join(data_dir, "dials_trade.json"), "w") as f:
        json.dump(dials, f, indent=2)


def trade_process(data_dir=None, force_proc=False):
    # trade_proc = TradePreprocess(data_dir)
    # trade_proc.createData()
    # trade_proc.SaveData()
    if os.path.exists(os.path.join(data_dir, "dials_trade.json")) and not force_proc:
        pass
        # print("already preprocessed before, skipping this time ...")
    else:
        delex_data = createData(data_dir)
        SaveData(delex_data, data_dir)


def main():
    print("Create WOZ-like dialogues. Get yourself a coffee, this might take a while.")
    delex_data = createData()
    print("Divide dialogues...")
    divideData(delex_data)
    # print('Building dictionaries')
    # buildDictionaries(word_freqs_usr, word_freqs_sys)


if __name__ == "__main__":
    main()
