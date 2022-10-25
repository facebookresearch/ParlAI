#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Restore perturbed synthetic data to maintain certain percent of perturbations.

Usage:
with perturbation IDs file:
python projects/roscoe/roscoe_data/restore_data.py --dataset-path ~/test --perturbation-ids \
projects/roscoe/roscoe_data/unperturbed_ids.json --percentage 50 --out-dir ~/test

random perturbations:
python projects/roscoe/roscoe_data/restore_data.py --dataset-path ~/test --percentage 50 --out-dir ~/test
"""
import copy
import json
import os
import random
from typing import Any, Dict, Iterable, List, Set

from parlai.core.params import ParlaiParser


def random_indices(list_length: int, percentage: int) -> Set[int]:
    random.seed(42)

    return set(
        random.sample(range(list_length), list_length * (100 - percentage) // 100)
    )


def restore_positives(
    neg_samples: List[Dict[str, Any]], indices: List[int]
) -> List[Dict[str, Any]]:
    mixed_samples = copy.deepcopy(neg_samples)

    for i in indices:
        mixed_samples[i]["perturbed"] = False
        mixed_samples[i]["dialog"][0][0]["steps"] = mixed_samples[i]["dialog"][0][0][
            "original_steps"
        ]
        mixed_samples[i]["dialog"][0][0]["eval_labels"] = [
            "\t".join(mixed_samples[i]["dialog"][0][0]["steps"])
        ]
        mixed_samples[i]["dialog"][0][1]["text"] = "\t".join(
            mixed_samples[i]["dialog"][0][0]["steps"]
        )
    return mixed_samples


def read_json_lines(file_name: str) -> Iterable[Dict[str, Any]]:
    with open(file_name, 'r') as f:
        for line in f:
            yield json.loads(line)


def read_negative_file(directory_path: str, file_name: str) -> Iterable[Dict[str, Any]]:
    for sample in read_json_lines(os.path.join(directory_path, file_name)):
        sample["perturbed"] = True
        yield sample


def write_json_lines(
    file_name: str,
    samples: List[Dict[str, Any]],
) -> None:
    print(f"Writing {len(samples)} samples to {file_name}")
    with open(file_name, 'w') as f:
        for item in samples:
            json.dump(item, f)
            f.write('\n')


def restore_and_write(
    samples: List[Dict[str, Any]], indices: List[int], directory: str, output_file: str
):
    if not os.path.exists(directory):
        os.makedirs(directory)
    mixed_samples = restore_positives(neg_samples=samples, indices=indices)
    write_json_lines(os.path.join(directory, output_file), mixed_samples)


def random_perturbations(set_path: str, percent: int):
    for root, _dirnames, filenames in os.walk(set_path):
        for filename in filenames:
            if "jsonl" in filename:
                set_name = root.split('/')[-1]
                out_file_name = f"{percent}%_{filename}.jsonl"
                neg_samples = list(
                    read_negative_file(
                        root,
                        filename,
                    )
                )
                indices_to_dupe = random_indices(len(neg_samples), percentage=percent)
                restore_and_write(
                    samples=neg_samples,
                    indices=indices_to_dupe,
                    directory=os.path.join(opt['out_dir'], set_name),
                    output_file=out_file_name,
                )


def deterministic_perturbations(set_path: str, ids_path: str, percent: int):
    with open(ids_path, 'r') as f:
        indices = json.load(f)
    for root, _dirnames, filenames in os.walk(set_path):
        for filename in filenames:
            if "jsonl" in filename:
                set_name = root.split('/')[-1]
                if set_name not in indices:
                    raise ValueError(
                        f"Perturbation IDs not found for set {set_name}. Available sets are {indices.keys()}"
                    )
                if filename not in indices[set_name]:
                    raise ValueError(
                        f"Perturbation IDs not found for file {filename} in {set_name}"
                    )
                out_file_name = f"{percent}%_{filename}"
                neg_samples = list(
                    read_negative_file(
                        root,
                        filename,
                    )
                )
                restore_and_write(
                    samples=neg_samples,
                    indices=indices[set_name][filename],
                    directory=os.path.join(opt['out_dir'], set_name),
                    output_file=out_file_name,
                )


def main(opt):

    pert_percent = int(opt['percentage'])
    if opt['perturbation_ids']:
        filename = opt['perturbation_ids']
        print(f"Using perturbation indices provided in {filename}")
        deterministic_perturbations(
            set_path=opt['dataset_path'], ids_path=filename, percent=pert_percent
        )
    else:
        print(f"Including random perturbations")
        random_perturbations(set_path=opt['dataset_path'], percent=pert_percent)


if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument(
        '--dataset-path',
        '-d',
        type=str,
        required=True,
        help='Path to files with perturbations',
    )
    parser.add_argument(
        '--perturbation-ids',
        '-p',
        type=str,
        required=False,
        help='Path to the file containing unperturbed IDs',
    )
    parser.add_argument(
        '--percentage',
        '-p',
        type=str,
        required=True,
        help='Percentage of data to leave perturbed. 100 means keep all perturbations, 0 means restore all to ground truth.'
        + 'If perturbation-ids file provided, only used to generate the perturbed file name',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        required=True,
        help='Path where new perturbation mixes will be saved.',
    )

    opt = parser.parse_args()

    main(opt)
