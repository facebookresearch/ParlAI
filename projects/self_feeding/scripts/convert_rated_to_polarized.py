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

from parlai.mturk.tasks.self_feeding.rating.worlds import (
    NEW_TOPIC_REQUEST,
    SUGGESTION_REQUEST,
)
from parlai.utils.io import PathManager

# Initial prompts vary due to the random nouns, but all will start this way
INITIAL_PROMPT = "start a conversation"

REPORT_DIR = '/private/home/bhancock/metadialog/mturk/reports'
DATA_DIR = '/private/home/bhancock/metadialog/data'

DEFAULT_IN = REPORT_DIR + '/20181105/pilot_1.txt'
DEFAULT_OUT = DATA_DIR + '/feedback_classifier/temp.txt'


def setup_args():
    parser = ArgumentParser()
    parser.add_argument('-if', '--infile', type=str, default=DEFAULT_IN)
    parser.add_argument('-of', '--outfile', type=str, default=DEFAULT_OUT)
    parser.add_argument(
        '-histsz',
        '--history-size',
        type=int,
        default=-1,
        help="The number of turns to include in the prompt.",
    )
    parser.add_argument(
        '-pos',
        '--positives',
        type=str,
        default='3,4,5',
        help="A comma-separated list of ratings with positive label",
    )
    parser.add_argument(
        '-neg',
        '--negatives',
        type=str,
        default='1',
        help="A comma-separated list of ratings with negative label",
    )
    config = vars(parser.parse_args())

    return config


def main(config):
    """
    Extracts training data for the negative response classifier (NRC) from Mturk logs.

    input: file of logs (in ParlaiDialog format) from Mturk task 1 with turn-by-turn
        quality ratings 1-5
    output: file of episodes (self-feeding format) w/ +1/-1 ratings indicating
        positive/negative example
    """
    examples = []
    positives = config['positives'].split(',')
    negatives = config['negatives'].split(',')
    assert len(set(positives).intersection(set(negatives))) == 0

    num_episodes = 0
    num_parleys = 0
    for episode in extract_parlai_episodes(config['infile']):
        num_episodes += 1
        history = []
        for parley in episode:
            num_parleys += 1

            # Update history (not including stock control flow responses)
            if parley.context.startswith(INITIAL_PROMPT.lower()):
                # Conversation prompt, first utterance
                history = [parley.response]
            elif parley.context.startswith(SUGGESTION_REQUEST.lower()):
                # Asked for y_exp, got y_exp
                pass
            elif parley.context.startswith(NEW_TOPIC_REQUEST.lower()):
                # Asked for new topic, got a first utterance
                history = [parley.response]
            else:
                history.append(parley.context)
                history.append(parley.response)

            # Only create a new example if this parley's rating is relevant
            if parley.reward in (positives + negatives):
                # Concatenate history and add speaker tokens as necessary
                # history_size refers to the total number of utterances
                # (history_size == 0 means predict sentiment from '__null__')
                # response that's being classified (so if history_size == 0 then
                # classify based only on the response w/o any extra context).
                # Note that the response being classified should always be preceded by
                # __p1__ (the human), not __p2__ (the bot).
                if config['history_size'] < 0:
                    utterances = history
                elif config['history_size'] == 0:
                    utterances = ['__null__']
                else:
                    utterances = history[-config['history_size'] :]

                context = add_person_tokens(utterances, last_speaker=1)

                label = 1 if parley.reward in positives else -1

                example = Parley(context, label)
                examples.append(example)

    with PathManager.open(config['outfile'], 'w') as outfile:
        for ex in examples:
            outfile.write(json.dumps(ex.to_dict()) + '\n')

    print(
        f"Extracted {len(examples)} ratings out of {num_episodes} episodes "
        f"({num_parleys} parleys) and wrote them to {config['outfile']} with "
        f"histsz == {config['history_size']}."
    )


if __name__ == '__main__':
    config = setup_args()
    main(config)
