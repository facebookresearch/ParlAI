#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser

from parlai.projects.self_feeding.utils import extract_parlai_episodes
from parlai.mturk.tasks.self_feeding.rating.worlds import (
    NEW_TOPIC_REQUEST,
    SUGGESTION_REQUEST,
)

# Initial prompts vary due to the random nouns, but all will start this way
INITIAL_PROMPT = "start a conversation"


def setup_args():
    argparser = ArgumentParser()
    argparser.add_argument('-if', '--infile', type=str)
    argparser.add_argument('-of', '--outfile', type=str)
    config = vars(argparser.parse_args())

    return config


def main(config):
    """
    Creates input files for y_exp mturk task from conversation/rating mturk task.

    input: file of logs (in ParlaiDialog format) from Mturk task 1 with turn-by-turn
        quality ratings 1-5
    output: file of logs (in ParlaiDialog format) sliced up to begin at the start of
        an episode or following a new topic request, and ending with a y_exp
    """
    new_episodes = []
    old_episodes = [e for e in extract_parlai_episodes(config['infile'])]
    for episode in old_episodes:
        for parley in episode:
            if any(
                parley.context.startswith(x)
                for x in (NEW_TOPIC_REQUEST.lower(), INITIAL_PROMPT.lower())
            ):
                new_episode = []
            new_episode.append(parley)
            if parley.context.startswith(SUGGESTION_REQUEST.lower()):
                new_episodes.append(new_episode)

    # Create parlai dialog file for easy viewing
    with open(config['outfile'], 'w') as f:
        for episode in new_episodes:
            num_parleys = len(episode)
            for i, parley in enumerate(episode):
                if i == num_parleys - 1:
                    parley.episode_done = True
                f.write(f"{i}\t{parley.to_parlai()}\n")
    print(
        f"Extracted {len(new_episodes)} episodes out of {len(old_episodes)} "
        f"original episodes and wrote them to {config['outfile']}."
    )


if __name__ == '__main__':
    config = setup_args()
    main(config)
