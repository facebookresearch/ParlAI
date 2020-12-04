import configargparse
from copy import copy
import json
import os
from os import path
import re
from typing import List, Optional, Union
import torch
import subprocess
from parlai_internal.projects.empathy_generation.data_utils.upsample_data import (
    upsample_examples,
)
from parlai_internal.tasks.empathy_labeled_utterances.agents import (
    _path as get_data_path,
)

DATA_PREFIXES = [
    "original",
    "original_upsampled",
    "relabeled_prev_curr",
    "relabeled_prev_curr_upsampled",
    "relabeled_prev_curr_future",
    "relabeled_prev_curr_future_upsampled",
    "relabeled_second_pass_prev_curr",
    "relabeled_second_pass_prev_curr_upsampled",
    "relabeled_second_pass_prev_curr_future",
    "relabeled_second_pass_prev_curr_future_upsampled",
]


def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "config", is_config_file=True, type=str, help="path to config file"
    )
    parser.add_argument(
        "--data_prefix_in", type=str, choices=DATA_PREFIXES, required=True, help="TODO"
    )
    parser.add_argument(
        "--data_prefix_out", type=str, choices=DATA_PREFIXES, required=True, help="TODO"
    )
    parser.add_argument(
        "--classifier_path", type=str, required=True, help="path to trained classifier"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=["prev_curr", "prev_curr_future"],
        help="either load PrevCurUtteranceClassifier (prev_curr) PrevCurrFutureUtteranceClassiifer (prev_curr_future)",
    )
    parser.add_argument(
        "--output_file_prefix",
        type=str,
        required=True,
        help="file prefix for relabeled splits",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        required=True,
        help="directory to save evaluation reports",
    )
    # for formatting for classifier
    parser.add_argument(
        "--exclude_from_eval",
        nargs="+",
        choices=[
            "bst",
            "style_gen",
            "ed_train",
            "ed_valid",
            "ed_test",
            "wow",
            "convai2",
        ],
        default=[],
        help="these datasets will be removed during evaluation run",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="per device eval batch size"
    )
    parser.add_argument(
        "--relabeling_threshold",
        type=float,
        default=0.6,
        help="probability threshold for relabeling mislabeled examples from classifier",
    )
    parser.add_argument(
        "--results_file_prefix",
        type=str,
        required=True,
        help="path to directory to write relabeled examples",
    )
    args = parser.parse_args()

    if not os.path.exists(args.report_dir):
        os.mkdir(args.report_dir)

    if args.task_name == "prev_curr":
        task = "parlai_internal.tasks.empathy_labeled_utterances.agents:PrevCurrUttEmpathyAgent"
    elif args.task_name == "prev_curr_future":
        task = "parlai_internal.tasks.empathy_labeled_utterances.agents:PrevCurrFutureUttEmpathyAgent"
    else:
        raise Exception(
            "Either specify use_prev_curr_utt or use_future_utt to define task.."
        )

    for datatype in ["train:evalmode", "valid", "test"]:
        opt = {
            "datapath": "/private/home/ccross/ParlAI/data",
            "data_prefix": "original",
            "datatype": datatype,
            "exclude_src_datasets": "none",
            "exclude_bot_responses": False,
        }
        split_name = datatype.split(":")[0]
        raw_examples = {}
        relabeled_examples = []
        with open(get_data_path(opt)) as f_data_in, open(
            f"{args.output_file_prefix}_{args.task_name}_{split_name}.jsonl", "w"
        ) as f_data_out, open(
            f"{args.results_file_prefix}_{split_name}.txt", "w"
        ) as f_results:
            # read examples for GT
            for idx, line in enumerate(f_data_in):
                conv = json.loads(line)
                raw_key = conv["src_dataset"] + "_" + conv["conv_turn_id"]
                raw_examples[raw_key] = conv

            # run evaluation before relabeling
            cmd = f"""parlai eval_model \
                    -t {task} \
                    -mf {args.classifier_path} \
                    -m parlai_internal.projects.empathy_generation.style_classifier.from_pretrained_model:FromPreTrainedClassifierAgent \
                    --batchsize {args.batch_size} \
                    --data_prefix {args.data_prefix_in} \
                    --datatype {datatype} \
                    --exclude-src-datasets none \
                    --exclude-bot-responses False \
                    --save-world-logs True \
                    --report-filename {args.report_dir}/report_{datatype}.txt \
                    --retain_none_type True \
                    --print-scores True \
                    --ignore_labels True \
                    --fp16-impl mem_efficient \
                    --fp16 True \
                """
            subprocess.run(cmd.split(), check=True)
            report_file = f"{args.report_dir}/report_{datatype}_{task}_replies.jsonl"
            print(f"Saved report to {report_file}")

            updates_by_key = {"empathetic": [], "not_empathetic": [], "none": []}
            with open(report_file) as f:
                for idx, report_line in enumerate(f):
                    episode, prediction = json.loads(report_line)["dialog"][0]

                    # load raw conversation
                    if "text" not in episode or "text" not in prediction:
                        continue
                    raw_key = episode["src_dataset"] + "_" + episode["conv_turn_id"]
                    raw_conv = raw_examples[raw_key]
                    ground_truth = raw_conv["style"]

                    # get predicted style
                    text_fields = re.fullmatch(
                        r"Predicted class: (.+)\nwith probability: ([\d.]+)",
                        prediction["text"],
                    )
                    pred = text_fields.group(1)
                    prob = float(text_fields.group(2))

                    assert episode["conv_turn_id"] == raw_conv["conv_turn_id"], (
                        f"{idx} {len(raw_examples)}"
                        + episode["conv_turn_id"]
                        + " "
                        + raw_conv["conv_turn_id"]
                    )

                    if (
                        pred != ground_truth and prob >= args.relabeling_threshold
                    ) or ground_truth in ["none", None]:
                        raw_conv["style"] = pred  # update ground-truth
                        raw_examples[raw_key] = raw_conv  # update example in list
                        relabeled_examples.append(
                            (prob, ground_truth, pred, episode["text"])
                        )

                    if ground_truth is None:
                        ground_truth = "none"  # change to str key
                    updates_by_key[ground_truth].append(pred != ground_truth)

                    json.dump(raw_conv, f_data_out)
                    f_data_out.write("\n")
            print(
                "Saved relabeling results to",
                f"{args.output_file_prefix}_{split_name}.jsonl",
            )

            for key, relabels in updates_by_key.items():
                f_results.write(
                    f"For {key}, relabeled {sum(relabels)} of {len(relabels)} examples\n"
                )
                print(
                    f"For {key}, relabeled {sum(relabels)} of {len(relabels)} examples\n"
                )

            # write relabeled examples to file
            for prob, ground_truth, pred, input_text in sorted(
                relabeled_examples, key=lambda x: x[0], reverse=True
            ):
                f_results.write(
                    f"{prob} - gt/pred: {ground_truth}/{pred} for {input_text}\n"
                )
            print(f"Wrote results to {args.results_file_prefix}_{split_name}.txt")

            # upsample
            if "train" in split_name:
                style_candidates = ["empathetic", "not_empathetic"]
                upsampled_input_examples, _ = upsample_examples(
                    list(raw_examples.values()), style_candidates
                )
                with open(
                    f"{args.output_file_prefix}_{args.task_name}_upsampled_{split_name}.jsonl",
                    "w",
                ) as f_data_upsampled:
                    for ex in upsampled_input_examples:
                        json.dump(ex, f_data_upsampled)
                        f_data_upsampled.write("\n")

    # run evaluation again after labeling
    for datatype in ["valid"]:  # ['train:evalmode', 'valid', 'test']:
        print("Eval results after relabeling:")
        cmd = f"""parlai eval_model \
            --batchsize {args.batch_size} \
            --data_prefix {args.data_prefix_out} \
            --datatype {datatype} \
            --exclude-src-datasets style_gen convai2 wow \
            --exclude-bot-responses True \
            --model parlai_internal.projects.empathy_generation.style_classifier.from_pretrained_model:FromPreTrainedClassifierAgent \
            --model-file {args.classifier_path} \
            --model-parallel True \
            --retain_none_type False \
            --print-scores False \
            -t {task} \
            """
        # print(' '.join(cmd.split('             ')))
        subprocess.run(cmd.split(), check=True)
        # print(f'Finished eval for {datatype}')


if __name__ == "__main__":
    main()
