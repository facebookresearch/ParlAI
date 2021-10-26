# This is necessary to make sure that we make a fair comparison with the same kind of slot values seen during training and testing
# Assumption: predictions are the same

from argparse import ArgumentParser
from pathlib import Path
import json
import os
from tqdm import tqdm

# parser = ArgumentParser()

# parser.add_argument("-sd", "--sub_dir", default="orig", help="subdirectory with files to have the slot inference values replaced with those from v23")
# args = parser.parse_args()

invs = ["SD", "TP", "orig"]

for inv in invs:
    keys = ["train", "valid", "test"]
    # keys = ["test"]
    for key in keys:
        fn = os.path.join("LAUG", inv, f"data_reformat_{key}.json")
        with open(fn, "r") as f:
            data = json.load(f)

        gold_path = os.path.join(
            "multiwoz_dst/MULTIWOZ2.3", f"data_reformat_{key}.json"
        )
        with open(gold_path, "r") as f:
            gold_data = json.load(f)

        more_laug = len(set(data.keys()) - set(gold_data.keys()))
        more_official_v23 = len(set(gold_data.keys()) - set(data.keys()))
        # assert data.keys() == gold_data.keys(), (more_laug, more_official_v23)
        print(
            f"For split '{key}': # keys more in laug: {more_laug}, # keys more in official_v23: {more_official_v23}"
        )

        new_data = {}
        fn = os.path.join("LAUG", inv, f"data_reformat_official_v23_slots_{key}.json")
        for k, v in tqdm(data.items()):
            if k not in gold_data:
                continue
            new_data[k] = data[k].copy()
            new_data[k]["slots_inf"] = gold_data[k]["slots_inf"]
            new_data[k]["orig_context"] = gold_data[k]["context"]
            if "orig" in inv:
                new_data[k]["context"] = gold_data[k]["context"]

        with open(fn, "w") as f:
            json.dump(obj=new_data, fp=f, indent=2)
        print(len(new_data))
