import configargparse
from collections import Counter
from itertools import permutations
import json
import os
from os import path
from random import randint
import subprocess
from parlai_internal.projects.empathy_generation.style_controlled_generation.model_utils import (
    BatchGeneratorWithFutureClassifier,
    load_gen_agent,
    HISTORY_DELIMITER,
)
from parlai_internal.tasks.empathy_labeled_utterances.agents import (
    _path as get_datapath,
)
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.message import Message
from parlai.core.params import ParlaiParser

# see https://github.com/fairinternal/ParlAI-Internal/blob/8f914fed38240fd1e9180e00769afee6c66a3380/projects/style_gen/auto_evals/s2020_04_30__auto_metrics/generate_given_styles.py

MAX_TURN_NUM = 2  # we'll keep either turns 0, 1, or 2


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "config", is_config_file=True, type=str, help="path to config file"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--data_prefix",
        type=str,
        choices=[
            "original",
            "original_upsampled",
            "relabeled_prev_curr",
            "relabeled_prev_curr_upsampled",
            "relabeled_prev_curr_future",
            "relabeled_prev_curr_future_upsampled",
            "relabeled_second_pass_prev_curr_future",
            "relabeled_second_pass_prev_curr_future_upsampled",
        ],
        required=True,
        help="TODO",
    )
    parser.add_argument(
        "--generator_path", type=str, required=True, help="path to trained generator"
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        required=True,
        help="path to trained future classifier",
    )
    parser.add_argument(
        "--class_path", type=str, required=True, help="path to list of classes"
    )
    parser.add_argument(
        "--history_size",
        type=int,
        default=2,
        help="history size (in # of utterances) to retain for each turn",
    )
    parser.add_argument(
        "--generation_dir",
        type=str,
        required=True,
        help="directory to save generations",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        required=True,
        help="directory to save evaluation reports",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.generation_dir, exist_ok=True)

    for datatype in ["test"]:  # ['test']: #['train:evalmode', 'valid', 'test']:
        print("\nLoading data.")
        # load task agent and format example for generator
        agent = load_gen_agent(datatype, args.data_prefix)
        examples = []
        for episode_id in range(agent.num_episodes()):
            episode = agent.data[episode_id]
            num_turns = len(episode)
            if num_turns < 2:
                continue
            entry_id = randint(1, num_turns - 1)
            entry = episode[entry_id]

            context = entry["history"] + [entry["speaker_utt"]]
            if len(context) > args.history_size:
                context = context[-args.history_size :]
            entry["text"] = HISTORY_DELIMITER.join(context)
            entry["episode_id"] = episode_id
            entry["entry_id"] = entry_id
            entry["episode_done"] = True
            examples.append(entry)

        print("Load and run generator with no controlled style")
        # first run with no controlled style
        # generator = BatchGeneratorWithFutureClassifier(
        #    args.generator_path,
        #    args.classifier_path,
        #    args.class_path,
        #    args.batch_size,
        #    args.history_size,
        #    style_frac=0,
        # )
        # num_iter = 64
        # with open(
        #    f"{args.generation_dir}/generations_{datatype}_no_style.jsonl", "w"
        # ) as f_out:
        #    for start_idx in range(0, len(examples), num_iter):
        #        print(f"Starting idx: {start_idx}/{len(examples)}")
        #        generations = generator.batch_generate(
        #            examples[start_idx : start_idx + num_iter]
        #        )
        #        for conv in generations:
        #            json.dump(conv, f_out)
        #            f_out.write("\n")
        # print("wrote to file")

        # next use ground truth style label
        # generator = BatchGeneratorWithFutureClassifier(
        #    args.generator_path,
        #    args.classifier_path,
        #    args.class_path,
        #    args.batch_size,
        #    args.history_size,
        #    style_frac=1,
        # )
        # num_iter = 64
        # with open(
        #    f"{args.generation_dir}/generations_{datatype}_ground_truth_style.jsonl", "w"
        # ) as f_out:
        #    for start_idx in range(0, len(examples), num_iter):
        #        print(f"Starting idx: {start_idx}/{len(examples)}")
        #        generations = generator.batch_generate(
        #            examples[start_idx : start_idx + num_iter]
        #        )
        #        for conv in generations:
        #            json.dump(conv, f_out)
        #            f_out.write("\n")
        # print("wrote to file")

        # finally invert style
        print("Run with inverted style")
        generator = BatchGeneratorWithFutureClassifier(
            args.generator_path,
            args.classifier_path,
            args.class_path,
            args.batch_size,
            args.history_size,
            style_frac=1,
        )
        invert_style = {'empathetic': 'not_empathetic', 'not_empathetic': 'empathetic'}
        for i in range(len(examples)):
            orig_style = examples[i]['style']
            new_style = invert_style[orig_style]
            examples[i]['style'] = new_style

        num_iter = 64
        with open(
            f"{args.generation_dir}/generations_{datatype}_inverted_style.jsonl", "w"
        ) as f_out:
            for start_idx in range(0, len(examples), num_iter):
                print(f"Starting idx: {start_idx}/{len(examples)}")
                generations = generator.batch_generate(
                    examples[start_idx : start_idx + num_iter]
                )
                for conv in generations:
                    json.dump(conv, f_out)
                    f_out.write("\n")
        print("wrote to file")


if __name__ == "__main__":
    main()
