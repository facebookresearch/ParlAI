import json
from utils import (
    read_zipped_json,
    format_dst_slots,
    seq2dict,
    dict2seq,
    multiwoz_diag_act_to_dict,
    update_slots,
    proper_dst_format,
    my_strip,
)
import os
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser
from pprint import pprint


keys = ["train", "val", "test"]

parser = ArgumentParser()
parser.add_argument(
    "-sd", "--subdir", required=True, help="folderpath that contains data to reformat"
)

invs = ["SD", "TP", "orig"]

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LAUG")

for inv in invs:
    sub_dir = inv

    for key in keys:
        data = read_zipped_json(
            os.path.join(data_dir, sub_dir, key + '.json.zip'), key + '.json'
        )
        print('load {}, size {}'.format(key, len(data)))

        results = {}
        skipped_no_aug = 0
        for title, sess in tqdm(data.items()):
            logs = sess['log']
            context = ""
            context_ = (
                ""
            )  # for keeping track of augmented conversation if there is any augmentation for 'text' field such that is different from the 'originalText' field

            slots = {}
            for i, diag in enumerate(logs):
                # format utterance
                text = diag['text'].replace('\t', ' ').replace('\n', ' ')
                for c in [" '", " ?", " ,", " .", " !", " n't"]:
                    # remove spaces in front of punctuation and n't
                    text = text.replace(c, c[1:])

                # odd turns are user turns. add DST example
                if i % 2 == 0:
                    # get slots in desired format
                    # slots += " " + format_dst_slots(diag["dialog_act"])

                    # # handle redundancies
                    # dict_slots = seq2dict(slots)
                    # slots = dict2seq(dict_slots).replace("  ", " ")

                    # context += "<user> " + text + " "
                    # turn_num = int(i/2)
                    # turn = {
                    #     'turn_num': turn_num,
                    #     'dial_id': title.lower()+".json",
                    #     'slots_inf':slots.strip(),
                    #     "context": context.strip().lower(),
                    # }
                    # sample_name = turn['dial_id'] + f"-{turn_num}"

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

                    # if we are reformatting a datset with augmentation, keep the original text as well
                    if sub_dir != "orig":
                        try:
                            orig_text = (
                                diag.get("originalText", "")
                                .replace('\t', ' ')
                                .replace('\n', ' ')
                            )
                        except:
                            logger.info("Instance without 'originalText' field found: ")
                            import pdb

                            pdb.set_trace()
                            pprint(diag)
                            continue

                        context_ += "<user> " + orig_text + " "

                        # make sure that the original and the augmented version are not the same
                        # if they are the same, exclude
                        if orig_text == text:
                            # import pdb; pdb.set_trace()
                            # pprint(diag)
                            skipped_no_aug += 1
                            continue
                        turn["orig_context"] = my_strip(context_)

                    # if it's the original, there is no difference
                    else:
                        turn["orig_context"] = turn["context"]

                    results[sample_name] = turn

                else:
                    context += "<system> " + text + " "
                    context_ += "<system> " + text + " "

        if key == "val":
            key = "valid"
        logger.info(f"# {key} examples: {len(results)}")
        logger.info(f"# skipped because of no augmentation: {skipped_no_aug}")
        with open(
            os.path.join(data_dir, sub_dir, f"data_reformat_{key}.json"), "w"
        ) as f:
            json.dump(fp=f, obj=results, indent=2, sort_keys=True)
