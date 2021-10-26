# Usage: python compare_versions -fn1 file1 -fn2 file2 -r False
#
#

import json
from argparse import ArgumentParser
from pprint import pprint
import random
from loguru import logger
from utils import seq2dict
from collections import defaultdict
from tqdm import tqdm

parser = ArgumentParser()

parser.add_argument(
    "-fn1", "--fname1", required=True, help="first file to use for comparison"
)
parser.add_argument(
    "-fn2", "--fname2", required=True, help="second file to use for comparison"
)
parser.add_argument(
    "-r",
    "--random",
    required=False,
    default=False,
    help="set as true to use random shuffling of examples to examine",
)

args = parser.parse_args()

with open(args.fname1, "r") as f:
    data1 = json.load(f)

with open(args.fname2, "r") as f:
    data2 = json.load(f)


# assert data1.keys() == data2.keys(), "make sure the two files have the same format and examples"

# order is important
v1 = "file1"
v2 = "file2"
versions = ["2.3", "2.2", "2.2+" "2.1"]
for v in versions:
    if v in args.fname1:
        v1 = v
    if v in args.fname2:
        v2 = v

keys = sorted(list(set(list(data1.keys()) + list(data2.keys()))))
if args.random:
    random.shuffle(keys)


# import pdb; pdb.set_trace()


def get_diffs(dict1, dict2):
    diff_dict1 = defaultdict(dict)
    diff_dict2 = defaultdict(dict)

    # get slot values that are not in dict2
    for domain, slots in dict1.items():
        for key, val in slots.items():
            if (
                domain not in dict2
                or key not in dict2[domain]
                or dict1[domain][key] != dict2[domain][key]
            ):
                diff_dict1[domain][key] = val

    # get slot values that are not in dict1
    for domain, slots in dict2.items():
        for key, val in slots.items():
            if (
                domain not in dict2
                or key not in dict1[domain]
                or dict2[domain][key] != dict1[domain][key]
            ):
                diff_dict2[domain][key] = val

    return dict(diff_dict1), dict(diff_dict2)


next = True
prev_diff1, prev_diff2 = None, None
for key in tqdm(list(keys)[1144:]):
    if not next:
        break

    if key not in data1 or key not in data2:
        if key not in data1 and key in data2:
            logger.info(data2[key])
        if key in data1 and key not in data2:
            logger.info(data1[key])
        continue
    slot1 = dict(seq2dict(data1[key]['slots_inf']))
    slot2 = dict(seq2dict(data2[key]['slots_inf']))
    if slot1 != slot2:

        diff_dict1, diff_dict2 = get_diffs(slot1, slot2)
        if prev_diff1 == diff_dict1 and prev_diff2 == diff_dict2:
            continue
        prev_diff1, prev_diff2 = diff_dict1, diff_dict2

        # print(f"slots for {v1}: {slot1}")
        # print(f"slots for {v2}: {slot2}")

        print(f"Dialogue ID: {key}")
        if data1[key]['context'] != data2[key]['context']:
            print(f"Context1: {data1[key]['context']}")
            print(f"Context2: {data2[key]['context']}")
        else:
            print(f"shared context: {data1[key]['context']}")

        print(f"slots in {v1} not in {v2}: {diff_dict1}")
        print(f"slots in {v2} not in {v1}: {diff_dict2}")

        next = input("next? y/n: ") == "y"


logger.info("reached end of both files")
