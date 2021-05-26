#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import argparse

from parlai.utils.io import PathManager
import parlai.utils.logging as logging


"""
python projects/safety_recipes/human_safety_evaluation/format_safety_ready.py --world-logs-path /tmp/world_logs.jsonl --eval-logs-dir /tmp/human_safety_evaluation
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
            new_episode = []
            dialogue_text = episode['dialog'][0][0]['text'].split('\n') + [
                episode['dialog'][0][1]['text']
            ]
            for i in range(0, len(dialogue_text), 2):
                new_episode.append(
                    [
                        {
                            'text': dialogue_text[i],
                            'episode_done': False,
                            'id': 'human',
                        },
                        {
                            'text': dialogue_text[i + 1],
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
            assert len(new_episode) == human_eval_turn_range[1] + 1
            new_episode = new_episode[
                human_eval_turn_range[0] : human_eval_turn_range[1] + 1
            ]
            chatlogs.append(new_episode)

    task_data_path = os.path.join(args.eval_logs_dir, 'task_data.jsonl')
    indices_path = os.path.join(args.eval_logs_dir, 'annotation_indices.jsonl')
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
