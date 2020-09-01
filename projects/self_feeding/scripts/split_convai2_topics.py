#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser
import json
import os
import random

from parlai.projects.self_feeding.utils import extract_fb_episodes, episode_to_examples
from parlai.utils.io import PathManager

FAMILY = [
    "wife",
    "husband",
    "spouse",
    "mom",
    "momma",
    "mommy",
    "mum",
    "mother",
    "dad",
    "dadda",
    "daddy",
    "father",
    "parent",
    "grandparent",
    "grandma",
    "grandmom",
    "grandmum",
    "grandmother",
    "nana",
    "grandpa",
    "grandad",
    "granddad",
    "grandfather",
    "pappy",
    "brother",
    "sister",
    "sibling",
    "cousin",
    "aunt",
    "auntie",
    "uncle",
    "child",
    "children",
    "kid",
    "son",
    "daughter",
    "inlaw",
    "in-law",
]
variants = []
for word in FAMILY:
    variants.append(f"{word}s")
    variants.append(f"{word}'s")
FAMILY.extend(variants)

SPORTS = ["ball", "sports"]

# TODO: Make these a flag
# TOPIC = FAMILY # [FAMILY, SPORTS]
# TOPIC_NAME = 'family' # ['family', 'sports']

TOPIC = SPORTS  # [FAMILY, SPORTS]
TOPIC_NAME = 'sports'  # ['family', 'sports']


def setup_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-if', '--infile', type=str, default='data/ConvAI2/valid_self_original.txt'
    )
    parser.add_argument(
        '-of', '--outfile', type=str, default='data/convai2meta/dialog/valid.txt'
    )
    parser.add_argument(
        '--min-unit',
        type=str,
        default='example',
        choices=['episode', 'example'],
        help="The minimal unit that must stay grouped together",
    )
    parser.add_argument(
        '-shuf',
        '--shuffle',
        type=int,
        default=True,
        help="If True, shuffle the examples before writing them",
    )
    parser.add_argument(
        '-histsz',
        '--history-size',
        type=int,
        default=-1,
        help="The number of turns to concatenate and include in the "
        "prompt. Default: include all turns and filter in the teacher.",
    )
    opt = vars(parser.parse_args())
    return opt


def includes_topic(episode, topic):
    episode_words = (
        ' '.join([parley.context for parley in episode]).split()
        + ' '.join([parley.response for parley in episode]).split()
    )
    if TOPIC_NAME == 'family':
        return any(w in episode_words for w in topic)
    elif TOPIC_NAME == 'sports':
        episode_string = ' '.join(episode_words)
        return any(w in episode_string for w in topic)


def main(opt):
    """
    Converts a Fbdialog file of episodes into two self-feeding files (split by topic)

    All conversations including a word in the provided topic's bag of words will be
    separated from conversations without those words.
    """
    on_topic_exs = []
    off_topic_exs = []
    num_episodes = 0
    for episode in extract_fb_episodes(opt['infile']):
        num_episodes += 1
        if opt['min_unit'] == 'episode':
            if includes_topic(episode, TOPIC):
                on_topic_exs.extend(episode_to_examples(episode, opt['history_size']))
            else:
                off_topic_exs.extend(episode_to_examples(episode, opt['history_size']))
        elif opt['min_unit'] == 'example':
            for example in episode_to_examples(episode, opt['history_size']):
                if includes_topic([example], TOPIC):
                    on_topic_exs.append(example)
                else:
                    off_topic_exs.append(example)

    if opt['shuffle']:
        random.shuffle(on_topic_exs)
        random.shuffle(off_topic_exs)

    total = len(on_topic_exs) + len(off_topic_exs)
    on_pct = len(on_topic_exs) / total
    print(
        f"Separated {total} examples (from {num_episodes} episodes) into "
        f"{len(off_topic_exs)} off-topic and {len(on_topic_exs)} "
        f"({on_pct * 100:.1f}%) on-topic"
    )

    outfile_base, outfile_ext = os.path.splitext(opt['outfile'])
    unit_prefix = opt['min_unit'][:3]
    topic_prefix = TOPIC_NAME[:3]
    on_topic_filename = f"{outfile_base}_{unit_prefix}_{topic_prefix}{outfile_ext}"
    with PathManager.open(on_topic_filename, 'w') as outfile:
        for ex in on_topic_exs:
            outfile.write(json.dumps(ex.to_dict()) + '\n')
    off_topic_filename = f"{outfile_base}_{unit_prefix}_no{topic_prefix}{outfile_ext}"
    with PathManager.open(off_topic_filename, 'w') as outfile:
        for ex in off_topic_exs:
            outfile.write(json.dumps(ex.to_dict()) + '\n')


if __name__ == '__main__':
    print(
        "WARNING: With inline candidates, family words are still being encoded; "
        "DictAgent makes vocab from text,labels right now"
    )
    opt = setup_args()
    main(opt)
