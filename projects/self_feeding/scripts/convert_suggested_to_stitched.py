#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser
import json

from parlai.projects.self_feeding.utils import (
    Parley,
    extract_parlai_episodes,
    add_person_tokens,
)


def setup_args():
    argparser = ArgumentParser()
    argparser.add_argument('-if', '--infile', type=str)
    argparser.add_argument('-of', '--outfile', type=str)
    config = vars(argparser.parse_args())

    return config


def main(config):
    """
    Creates .stitched files from .suggested files.

    input: a .suggested file of logs (in ParlaiDialog format) from Mturk task 2, each of
        which starts with an initial prompt or topic request, and ends with a y
        that corresponds to the y_exp given in the previous turn
    output: a .stitched file (in self-feeding format) with the original mistake by the
        bot replace with the mturked y (based on y_exp)
    """
    examples = []
    episodes = [e for e in extract_parlai_episodes(config['infile'])]
    for episode in episodes:
        history = []
        num_parleys = len(episode)
        for i, parley in enumerate(episode):
            if i == 0:  # Don't include the topic request
                history.append(parley.response)
                continue
            elif i == num_parleys - 3:
                # third to last was mistake and negative feedback
                continue
            elif i == num_parleys - 2:
                # penultimate turn was suggestion request and y_exp
                continue
            elif i == num_parleys - 1:
                # ultimate turn was verbatim request and y
                example = Parley(
                    context=add_person_tokens(history, last_speaker=1),
                    response=parley.response,  # y
                )
                examples.append(example)
            else:
                # normal turn; just add to history
                history.append(parley.context)
                history.append(parley.response)

    # Write new episodes to self-feeding format
    with open(config['outfile'], 'w') as outfile:
        for ex in examples:
            outfile.write(json.dumps(ex.to_dict()) + '\n')

    print(
        f"Extracted {len(examples)} self-feeding episodes out of "
        f"{len(episodes)} parlai episodes and wrote them to {config['outfile']}."
    )


if __name__ == '__main__':
    config = setup_args()
    assert config['infile'].endswith('.suggested')
    assert config['outfile'].endswith('.stitched')
    main(config)
