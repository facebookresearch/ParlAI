#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script to increase bart size, should be run one-time to increase the position embedding size
to incorporate more context
Example usage
```
python /private/home/jingxu23/ParlAI/parlai_internal/projects/personal_knowledge/follow_up/model_augment/add_positions.py \
--model_save_path /checkpoint/jingxu23/projects/wiz2/inverse_persona/data/models/pretrainReddit3B_add_positions/ \
--original_model_path /checkpoint/parlai/zoo/meena/20200319_meenav0data_tall_2.7B_adamoptimizer/20200319_13.3ppl_200kupdates \
--n_positions 1024
```
args:
--n_positions: default 2048
--model_save_path: the path to save the enlarged mode
--original_model_path: the path to the original model
"""
from glob import glob
import torch
import numpy as np
import os
import json
from datetime import datetime
import argparse

from parlai.utils import logging
import parlai.utils.pickle
from parlai.utils.torch import atomic_save
from parlai_internal.projects.param_sweep_utils.param_sweep import bash
from fairseq.modules.sinusoidal_positional_embedding import (
    SinusoidalPositionalEmbedding,
)
from parlai.utils.io import PathManager
import parlai.utils.logging as logging


"""
python projects/safety_recipes/human_safety_evaluation/format_safety_ready.py --world-logs-path tmp/world_logs.jsonl --eval-logs-dir tmp/human_safety_evaluation
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world-logs-path",
        type=str,
        help="path to the world_logs.jsonl generated from parlai evaluate_model",
    )
    parser.add_argument(
        "--eval-logs-dir",
        type=str,
        help="dir to the store annotate_indices.jsonl and task_data.jsonl ready for direct use in human safety evaluation format",
    )

    args = parser.parse_args()
    world_logs_path = args.world_logs_path
    chatlogs = []
    with PathManager.open(world_logs_path) as data_file:
        for l in data_file.readlines():
            episode = json.loads(l.strip())
            # TODO: when conversation format is finished please remove this line;
            # TODO: 'human_eval_turn_range' unnecessary for WTC
            if 'human_eval_turn_range' not in episode['dialog'][0][0]:
                continue
            new_episode = []
            for turn in episode['dialog']:
                new_episode.append(
                    [
                        {'text': turn[0]['text'], 'episode_done': False, 'id': 'human'},
                        {
                            'text': turn[0]['eval_labels'][0],
                            'episode_done': False,
                            'id': 'bot',
                        },
                    ]
                )
            new_episode[-1][1]['text'] = episode['dialog'][-1][1]['text']
            new_episode[-1][1]['episode_done'] = True

            human_eval_turn_range = [
                int(x)
                for x in episode['dialog'][0][0]['human_eval_turn_range'].split('|')
            ]
            new_episode = new_episode[
                human_eval_turn_range[0] : human_eval_turn_range[1] + 1
            ]
            chatlogs.append(new_episode)

    task_data_path = os.path.join(args.eval_log_dir, 'task_data.jsonl')
    indices_path = os.path.join(args.eval_log_dir, 'annotation_indices.jsonl')
    with PathManager.open(task_data_path, 'w') as fw:
        for episode in chatlogs:
            fw.write(json.dumps(episode) + '\n')
    with PathManager.open(indices_path, 'w') as fw:
        for episode in chatlogs:
            fw.write(f'[{len(episode) * 2 -1}]' + '\n')

    logging.info(
        f'Saving task_data to {task_data_path} in human safety eval ready format'
    )
    logging.info(
        f'Saving annotation indices to {indices_path} in human safety eval ready format'
    )
