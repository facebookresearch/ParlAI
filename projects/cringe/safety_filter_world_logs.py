#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import argparse
import json
import os
from tqdm import tqdm
import random
import collections


def filter_world_logs_for_classifier_accuracy(
    world_logs_files: List[str], filtered_world_logs_file: str
):
    filtered_data_pos = []
    filtered_data_neg = []

    all_world_logs_files = []
    for file in world_logs_files:
        if not file.endswith('jsonl'):
            # Is it a directory that we can expand?
            all_world_logs_files.extend(
                [
                    os.path.join(file, f)
                    for f in os.listdir(file)
                    if 'world_logs' in f and f.endswith('jsonl')
                ]
            )
        else:
            all_world_logs_files.append(file)
    world_logs_files = all_world_logs_files

    count = 0
    pos_labels_count = collections.defaultdict(int)
    for world_logs_file in tqdm(world_logs_files):
        with open(world_logs_file, 'r') as f:
            for line in f.readlines():
                count += 1
                line_dict = json.loads(line)
                if line_dict['dialog'][0][1]['metrics']['classifier_accuracy'] == 0.0:
                    filtered_data_neg.append(line)
                elif line_dict['dialog'][0][1]['metrics']['classifier_accuracy'] == 1.0:
                    label = line_dict['dialog'][0][1]['text']
                    # Allow at most twice the same positive generation.
                    if label in pos_labels_count and pos_labels_count[label] >= 2:
                        continue
                    pos_labels_count[label] += 1
                    filtered_data_pos.append(line)

    num_filtered_data = min(len(filtered_data_neg), len(filtered_data_pos))
    filtered_data = (
        filtered_data_neg[:num_filtered_data] + filtered_data_pos[:num_filtered_data]
    )
    random.shuffle(filtered_data)

    with open(filtered_world_logs_file, 'w') as f:
        for line in filtered_data:
            f.write(line)

    print(
        f'Wrote {len(filtered_data)}/{count} examples to filtered log file: {filtered_world_logs_file}'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-logs-file', type=str, help='')
    parser.add_argument('--filtered-world-logs-file', type=str, help='')
    args = parser.parse_args()
    world_logs = args.world_logs_file.split(',')
    filter_world_logs_for_classifier_accuracy(world_logs, args.filtered_world_logs_file)
