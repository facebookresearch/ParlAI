import json
from typing import Dict, List
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import os
from pprint import pprint
from utils import (
    format_dst_slots,
    seq2dict,
    dict2seq,
    proper_dst_format,
    multiwoz_diag_act_to_dict,
    update_slots,
    my_strip,
)


data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "multiwoz_dst/MULTIWOZ2.3"
)
with open(os.path.join(data_dir, 'valListFile'), 'r') as f:
    val_list = f.read().splitlines()
with open(os.path.join(data_dir, 'testListFile'), 'r') as f:
    test_list = f.read().splitlines()

# with open(os.path.join(data_dir, 'data.json'), 'r') as f:
#     data = json.load(f)

keys = ["train", "val", "test"]
data = {}
for key in keys:
    with open(os.path.join(data_dir, f'data_{key}.json'), 'r') as f:
        data.update(json.load(f))

results = {}
results_val = {}
results_test = {}

for title, sess in tqdm(data.items()):
    logs = sess['log']
    context = ""
    title = title.replace(".json", "")

    # if title.lower() == "mul0003":
    #     pprint(logs)

    # decide which list to become a part of
    if title in val_list:
        current = results_val
    elif title in test_list:
        current = results_test
    else:
        current = results

    # keep track of all slots in format Dict[domain, Dict[slot key, slot val]]]
    slots = {}
    for i, diag in enumerate(logs):
        # format utterance
        text = diag['text'].replace('\t', ' ').replace('\n', ' ')
        for c in [" '", " ?", " ,", " .", " !", " n't"]:
            # remove spaces in front of punctuation and n't
            text = text.replace(c, c[1:])

        # odd turns are user turns. add DST example
        if i % 2 == 0:
            diag_act = multiwoz_diag_act_to_dict(diag["dialog_act"])
            if "coreference" in diag:
                update_dict = multiwoz_diag_act_to_dict(
                    diag["coreference"], coreference=True
                )
                diag_act = update_slots(diag_act, update_dict)
            slots = update_slots(slots, diag_act)

            slot_str = format_dst_slots(slots)

            assert isinstance(slot_str, str), slot_str
            if not proper_dst_format(slot_str):
                logger.info(slot_str)
                pprint(slots)
                pprint(diag["dialog_act"])
            # print(diag["dialog_act"])
            # print(slots)
            # form back to dictionary and then to seq in case there is any redundancy:
            # dict_slots = seq2dict(slots)
            # slots = dict2seq(dict_slots).replace("  ", " ")
            # break

            context += "<user> " + text + " "
            turn_num = int(i / 2)
            turn = {
                'turn_num': turn_num,
                'dial_id': title.lower() + ".json",
                'slots_inf': slot_str.strip(),
                "context": my_strip(context),
            }
            sample_name = turn['dial_id'] + f"-{turn_num}"
            current[sample_name] = turn

        # even turns are system turns. nothing to do other than extend context
        else:
            context += "<system> " + text + " "

    # break

logger.info(f"# training examples: {len(results)}")
logger.info(f"# validation examples: {len(results_val)}")
logger.info(f"# test examples: {len(results_test)}")
path = os.path.join(data_dir, "data_reformat_train.json")
print(path)
with open(os.path.join(data_dir, "data_reformat_train.json"), 'w') as f:
    json.dump(fp=f, obj=results, indent=2, sort_keys=True)
with open(os.path.join(data_dir, "data_reformat_valid.json"), 'w') as f:
    json.dump(fp=f, obj=results_val, indent=2, sort_keys=True)
with open(os.path.join(data_dir, "data_reformat_test.json"), 'w') as f:
    json.dump(fp=f, obj=results_test, indent=2, sort_keys=True)
