# get 50 converations from each of the attraction, train, restaurant, hotel domain for training
# another 50 for validation
# and 200 for testing
import json
import os
from argparse import ArgumentParser
from utils import seq2dict
from collections import defaultdict
import random


def load_all_data(path):

    dial_by_domain = defaultdict(dict)
    keys = ["test", "train", "valid"]

    # domains = ["hotel", "restaurant", "train", "taxi"]
    domains = ["attraction", "hotel", "restaurant", "train", "taxi"]

    dialid2turns = defaultdict(dict)
    for key in keys:

        if "LAUG" in path:
            with open(
                os.path.join(path, f"data_reformat_official_v23_slots_{key}.json"), "r"
            ) as f:
                data = json.load(f)

        else:
            with open(os.path.join(path, f"data_reformat_{key}.json"), "r") as f:
                data = json.load(f)

        # group by dialids, check last turns dst length, keep only those with single target domain
        # split apart to train test and valid in same format

        # only add conversations that have a single domain throughout the entire conversation
        for id, turn_item in data.items():
            dial_id = id.split(".json")[0]
            dialid2turns[dial_id][id] = turn_item

    for dialid, turns in dialid2turns.items():
        max_idx = -1
        for id, turn in turns.items():
            max_idx = max(turn["turn_num"], max_idx)

        last_dst = seq2dict(turns[f"{dialid}.json-{max_idx}"]["slots_inf"])

        if len(last_dst) != 1:
            continue

        else:
            domain = list(last_dst.keys())[0]
            if domain not in domains:
                continue
            dial_by_domain[domain][dialid] = turns

    print(f"Single domain counts:")
    for k, v in dial_by_domain.items():
        print(f"\t{k}: {len(v)}")
        # for id, turns in v.items():
        #     print(turns)
        #     break

    return dial_by_domain


parser = ArgumentParser()

parser.add_argument(
    "-p",
    "--path",
    help="directory of data that has reformatted multiwoz data (should have data_reformat_X.json name)",
)

args = parser.parse_args()


# path = "/data/home/justincho/project/ParlAI/data/multiwoz_dst/MULTIWOZ2.3"
# path = "/data/home/justincho/project/ParlAI/data/LAUG/TP"

path = args.path
dial_by_domain = load_all_data(path)

# goal: {domain: {dial_id: {dial_turn_id: dial_turn_item}}}


few_shot_sizes = {"train": 50, "valid": 50, "test": 200}
few_shot_train = 50
few_shot_valid = 50

few_shot_data = defaultdict(dict)

for domain, convs in dial_by_domain.items():

    # import pdb; pdb.set_trace()
    if domain == "attraction":
        few_shot_test = 80
    else:
        few_shot_test = 200

    conv_list = random.sample(
        list(convs.keys()), k=few_shot_train + few_shot_valid + few_shot_test
    )

    for conv_id in conv_list[:few_shot_train]:
        few_shot_data['train'].update(convs[conv_id])

    for conv_id in conv_list[few_shot_train : few_shot_train + few_shot_valid]:
        few_shot_data['valid'].update(convs[conv_id])

    for conv_id in conv_list[few_shot_train + few_shot_valid : few_shot_test]:
        few_shot_data['test'].update(convs[conv_id])


for k, v in few_shot_data.items():
    if "LAUG" in path:
        save_path = os.path.join(
            path, f"data_reformat_official_v23_slots_{k}_fewshot.json"
        )
    else:
        save_path = os.path.join(path, f"data_reformat_{k}_fewshot.json")
    with open(save_path, "w") as f:
        json.dump(obj=v, fp=f, indent=2, sort_keys=True)
