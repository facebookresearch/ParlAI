#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser
import json

from projects.self_feeding.utils import extract_fb_episodes, episode_to_examples
from parlai.utils.io import PathManager


def setup_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-if', '--infile', type=str, default='data/ConvAI2/valid_self_original.txt'
    )
    parser.add_argument(
        '-of', '--outfile', type=str, default='data/convai2meta/valid.txt'
    )
    parser.add_argument(
        '-histsz',
        '--history-size',
        type=int,
        default=-1,
        help="The number of turns to include in the prompt."
        "In general, include all turns and filter in the teacher.",
    )
    config = vars(parser.parse_args())
    return config


def main(config):
    """
    Converts a Fbdialog file of episodes into a json file of Parley examples.
    """
    examples = []
    for episode in extract_fb_episodes(config['infile']):
        examples.extend(episode_to_examples(episode, config['history_size']))

    with PathManager.open(config['outfile'], 'w') as outfile:
        for ex in examples:
            outfile.write(json.dumps(ex.to_dict()) + '\n')


if __name__ == '__main__':
    config = setup_args()
    main(config)
