# Adapted from LAUG work to format data for parlai training


import json
from argparse import ArgumentParser
import os
import json
import zipfile
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
from loguru import logger
import regex as re


def my_strip(context):
    context = re.sub("\s{2,}", " ", context)

    return context.strip().lower()


def format_dst_spans(spans: List):
    """
    Format turn-level span annotations from MultiWOZ2.3 data into slots
    input:  [[domain_intent, slot, value, span_start, span_end]]
    output: domain converted_slot_key value, domain slot value ... 
    """

    # try to match those with 2.2, but not necessary
    slot_key_conversions = {
        "depart": "departure",
        "dest": "destination",
        "leave": "leaveat",
        "arrive": "arriveby",
        "price": "pricerange",
    }

    domains = {
        "hotel",
        "attraction",
        "restaurant",
        "hospital",
        "police",
        "train",
        "taxi",
    }

    slot_str = ""
    for slot in spans:
        domain_da = slot[0]
        domain = domain_da.split("-")[0].strip().lower()
        if domain not in domains:
            continue
        slot_key, slot_val = slot[1], slot[2]
        slot_key = slot_key.replace(" '", "'").replace(",", "").lower().strip()
        slot_val = slot_val.replace(" '", "'").replace(",", "").lower().strip()
        # slot_key = slot_key_conversions.get(slot_key, slot_key.lower())
        if slot_val in {"?", "none"}:
            # logger.info(slots)
            continue
        if domain and slot_key and slot_val:
            slot_str += f"{domain} {slot_key} {slot_val}, "

    return slot_str.strip()


def seq2dict(slot_str: str):
    """
    convert sequences to dictionary. 
    Mainly for comparing DSTs from LAUG data (which uses MultiWOZ2.3) and with MultiWOZ22 data 
    input: domain slot-key slot-value, ... 
    output: [domain: {slot-key: slot-value}]
    """
    slot_str = slot_str.strip().replace("\n", "")
    if slot_str == "":
        return {}

    if slot_str[-1] == ",":
        slot_str = slot_str[:-1]

    dst_dict = defaultdict(dict)
    slots = slot_str.split(",")
    # print(slots)
    for slot in slots:

        split = slot.strip().split(" ")
        if len(split) < 3:
            print(slot_str)
            continue
        domain, slot_key, slot_val = split[0], split[1], " ".join(split[2:])
        dst_dict[domain][slot_key] = slot_val

    return dst_dict


def dict2seq(d):
    """
    convert {domain: {slot: value}} to "domain slot value, ..."
    """
    slot_str = ""
    for domain, slots in d.items():
        for slot_key, slot_val in slots.items():
            slot_str += f"{domain} {slot_key} {slot_val}, "

    return slot_str.strip()


def proper_dst_format(slot_str):
    slot_str = slot_str.strip().replace("\n", "")
    if slot_str == "":
        return True

    if slot_str[-1] == ",":
        slot_str = slot_str[:-1]

    dst_dict = defaultdict(dict)
    slots = slot_str.split(",")
    # print(slots)
    for slot in slots:
        split = slot.strip().split(" ")
        if len(split) < 3:
            print(slot_str)
            return False
    return True


def read_zipped_json(filepath, filename):
    print("zip file path = ", filepath)
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def seq2dict(slot_str: str):
    """
    convert sequences to dictionary. 
    Mainly for comparing DSTs from LAUG data (which uses MultiWOZ2.3) and with MultiWOZ22 data 
    input: domain slot-key slot-value, ... 
    output: [domain: {slot-key: slot-value}]
    """
    if slot_str == "":
        return {}

    if slot_str[-1] == ",":
        slot_str = slot_str[:-1]

    dst_dict = defaultdict(dict)
    slots = slot_str.split(",")
    # print(slots)
    for slot in slots:
        split = slot.strip().split(" ")
        if len(split) < 3:
            # print(slot)
            continue
        domain, slot_key, slot_val = split[0], split[1], " ".join(split[2:])
        dst_dict[domain][slot_key] = slot_val

    return dst_dict


def format_dst_slots(dialog_act: Dict):
    """
    Format turn-level dialog acts from LAUG data into slots
    input:  {domain: {slot key: slot value}}}
    output: domain slot_key slot_value, ... 
    """

    # optional conversions to match previous versions
    slot_key_conversions = {
        "Depart": "departure",
        "Dest": "destination",
        "Leave": "leaveat",
        "Arrive": "arriveby",
        "Price": "pricerange",
    }

    slot_val_conversions = {"Moderate": "moderately", "boat": "boats"}

    slot_str = []
    for domain, slots in dialog_act.items():
        for slot_key, slot_val in slots.items():
            slot_str.append(f"{domain} {slot_key} {slot_val},")

    return " ".join(slot_str).strip()


def multiwoz_diag_act_to_dict(diag_act, coreference=False):

    slot_val_conversions = {
        "centre": "center",
        "0-star": "0",
        "1-star": "1",
        "2-star": "2",
        "3-star": "3",
        "4-star": "4",
        "5-star": "5",
        "cheaper": "cheap",
        "theatre": "theater",
    }

    domains = {
        "restaurant",
        "hotel",
        "taxi",
        "hospital",
        "police",
        "train",
        "attraction",
    }

    # Dict[domain, Dict[slot key, slot val]]]
    dst_dict = defaultdict(dict)
    for domain_da, slots in diag_act.items():
        domain = domain_da.split("-")[0].lower()
        if domain not in domains:
            continue
        for slot in slots:
            try:
                if coreference:
                    slot_key, slot_val = slot[0], slot[2]
                else:
                    slot_key, slot_val = slot
            except:
                import pdb

                pdb.set_trace()
            slot_key = slot_key.replace(" '", "'").replace(",", "").lower().strip()
            slot_val = slot_val.replace(" '", "'").replace(",", "").lower().strip()
            if slot_val == "" or slot_val in {"?", "none"}:
                # logger.info(slots)
                continue

            if slot_val in slot_val_conversions:
                slot_val = slot_val_conversions[slot_val]

            dst_dict[domain][slot_key] = slot_val

    return dst_dict


def update_slots(dict_to_update, dict2):

    for domain, slots in dict2.items():
        if domain in dict_to_update:
            for key, val in slots.items():
                dict_to_update[domain][key] = val

        else:
            dict_to_update[domain] = slots

    return dict_to_update


def test_dialog_act():
    sample = {
        "dialog_act": {
            "Taxi-Inform": [
                ["Depart", "saint johns college"],
                ["Dest", "pizza hut fen ditton"],
            ]
        }
    }

    sample2 = {
        "dialog_act": {
            "Taxi-Inform": [
                ["Dest", "pizza hut fen ditton"],
                ["Depart", "saint johns college"],
            ]
        }
    }

    slots_str = format_dst_slots(sample['dialog_act'])
    print(slots_str)
    dict_slots = seq2dict(slots_str)
    print(dict_slots)

    slots_str2 = format_dst_slots(sample2['dialog_act'])
    print(slots_str2)
    dict_slots2 = seq2dict(slots_str2)
    print(dict_slots2)

    print(slots_str2 == slots_str)
    assert dict_slots == dict_slots2


if __name__ == '__main__':

    # test_dialog_act()

    parser = ArgumentParser()
    parser.add_argument(
        "-sd",
        "--subdir",
        required=True,
        help="folderpath that contains data to reformat",
    )

    args = parser.parse_args()

    sub_dir = args.subdir

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = cur_dir

    keys = ['train', 'val', 'test']
    data = {}
    for key in keys:
        data_key = read_zipped_json(
            os.path.join(data_dir, sub_dir, key + '.json.zip'), key + '.json'
        )
        print('load {}, size {}'.format(key, len(data_key)))
        data = dict(data, **data_key)

    with open(os.path.join(data_dir, 'valListFile'), 'r') as f:
        val_list = f.read().splitlines()
    with open(os.path.join(data_dir, 'testListFile'), 'r') as f:
        test_list = f.read().splitlines()

    results = {}
    results_val = {}
    results_test = {}

    # load original data to test whether we have the same results
    keys = ['train', 'valid', 'test']
    orig_multiwoz = {}
    for key in keys:
        multiwoz_path = f"/data/home/justincho/project/ParlAI/data/multiwoz_dst/MULTIWOZ2.2+/data_reformat_{key}.json"
        with open(multiwoz_path, "r") as f:
            orig_data = json.load(f)
        orig_multiwoz = dict(orig_multiwoz, **orig_data)

    match = 0
    no_match = 0
    skipped = 0
    for title, sess in tqdm(data.items()):
        logs = sess['log']
        context = ""

        # decide which list to become a part of
        if title in val_list:
            current = results_val
        elif title in test_list:
            current = results_test
        else:
            current = results

        slots = ""
        for i, diag in enumerate(logs):
            # format utterance
            text = diag['text'].replace('\t', ' ').replace('\n', ' ')
            for c in [" '", " ?", " ,", " .", " !", " n't"]:
                # remove spaces in front of punctuation and n't
                text = text.replace(c, c[1:])

            # odd turns are user turns. add DST example
            if i % 2 == 0:
                text = diag['originalText'].replace('\t', ' ').replace('\n', ' ')

                slots += format_dst_slots(diag["dialog_act"])
                context += "<user> " + text + " "
                turn_num = int(i / 2)
                turn = {
                    'turn_num': turn_num,
                    'dial_id': title.lower() + ".json",
                    'slots_inf': slots,
                    "context": context.strip(),
                }
                sample_name = turn['dial_id'] + f"-{turn_num}"
                current[sample_name] = turn

                slot_dict = seq2dict(slots)
                if sample_name not in orig_multiwoz:
                    skipped += 1
                    continue
                orig_slot = seq2dict(orig_multiwoz[sample_name]["slots_inf"])
                orig_context = orig_multiwoz[sample_name]["context"]
                if slot_dict == orig_slot:
                    match += 1
                else:
                    no_match += 1

                    # Compare against 2.2+
                    error_out = (
                        f"{'='*50}\nMutliWOZ2.3: {dict(slot_dict)}\n\nMultiWOZ2.2+{dict(orig_slot)}\n{'='*50}\n\n"
                        + f"\n2.3 Context: {context}\n\n2.2+ Context: {orig_context}"
                    )
                    # assert slot_dict == gold, error_out
                    import pdb

                    pdb.set_trace()
                    print(error_out)

            # even turns are system turns. nothing to do other than extend context
            else:
                context += "<system> " + text + " "

    print(f"Skipped: {skipped}\nMatch: {match}\nNo match: {no_match}")
