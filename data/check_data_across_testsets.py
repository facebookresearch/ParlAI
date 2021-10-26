# for comparing test set used for invariance metrics and the one used for training (in original multiwoz_dst/MULTIWOZ2.3)

import json
from loguru import logger


def test_full_shot_data():

    return


def test_few_shot_data():

    return


with open("multiwoz_dst/MULTIWOZ2.3/data_reformat_test.json", "r") as f:
    official = json.load(f)


augs = ["TP", "SD", "orig"]

for aug in augs:
    with open(f"LAUG/{aug}/data_reformat_official_v23_slots_test.json", "r") as f:
        laug_orig = json.load(f)

    off_keys = set(official.keys())
    laug_orig_keys = set(laug_orig.keys())

    extra_in_off = off_keys - laug_orig_keys
    extra_in_laug = laug_orig_keys - off_keys

    # if extra_in_off:
    #     print(len(extra_in_off))
    #     for k in extra_in_off:
    #         print(k)
    #         print(official[k])

    # if extra_in_laug:
    #     print(len(extra_in_laug))
    #     for k in extra_in_laug:
    #         print(k)
    #         print(laug_orig[k])

    # for those that intersect, check that the labels are the same
    intersection = off_keys.intersection(laug_orig_keys)
    count = 0
    for k in intersection:
        for key in official[k]:
            # if not original, comparison of the context key should be with "orig_context"
            if aug != "orig" and key == "context":
                laug_key = "orig_context"
            # for other keys, compare with same key
            else:
                laug_key = key
            # check for differences
            if official[k][key] != laug_orig[k][laug_key]:
                count += 1
                if key == "context":
                    print(official[k]["slots_inf"])
                    print(laug_orig[k]["slots_inf"])
                    for idx in range(len(official[k][key])):
                        if official[k][key][idx] != laug_orig[k][key][idx]:
                            print(official[k][key][max(idx - 3, 0) :])
                            print()
                            print(laug_orig[k][key][max(idx - 3, 0) :])
                            break
                else:
                    print(k, key)
                    print(official[k][key])
                    print(laug_orig[k][key])
        if count == 1:
            break

    if count == 0:
        logger.info(f"All clear for {aug}")
