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

# Initial prompts vary due to the random nouns, but all will start this way
INITIAL_PROMPT = "start a conversation"
EXP_REQUEST = "oops! i think i messed up."
NEWTOPIC = "thanks! i'll try to remember that."
RAT_REQUEST = "just checking:"
CONTINUE = "and in response to what you were saying before"  # Not the first words


def setup_args():
    argparser = ArgumentParser()
    argparser.add_argument('-if', '--infile', type=str)
    argparser.add_argument('-of', '--outfile', type=str)
    argparser.add_argument(
        '-histsz',
        '--history-size',
        type=int,
        default=-1,
        help="The number of turns to include in the prompt.",
    )
    argparser.add_argument(
        '-pos',
        '--positives',
        type=str,
        default='positive',
        help="A comma-separated list of ratings with positive label",
    )
    argparser.add_argument(
        '-neg',
        '--negatives',
        type=str,
        default='negative',
        help="A comma-separated list of ratings with negative label",
    )
    opt = vars(argparser.parse_args())

    return opt


def main(opt):
    """
    Extracts training data for the negative response classifier (NRC) from Mturk logs.

    input: file of logs (in ParlaiDialog format) from Mturk task 1 with turn-by-turn
        quality ratings 1-5
    output: file of episodes (self-feeding format) w/ +1/-1 ratings indicating
        positive/negative example
    """
    examples = []
    positives = opt['positives'].split(',')
    negatives = opt['negatives'].split(',')
    assert len(set(positives).intersection(set(negatives))) == 0

    num_episodes = 0
    num_parleys = 0
    for episode in extract_parlai_episodes(opt['infile']):
        num_episodes += 1
        history = []
        for parley in episode:
            num_parleys += 1

            # Update history (not including stock control flow responses)
            if parley.context.startswith(INITIAL_PROMPT):
                # Conversation prompt, first utterance
                # Begin history
                history = [parley.response]
            elif parley.context.startswith(EXP_REQUEST):
                # Asked for y_exp, got y_exp
                # Messed up, so blast history
                example = Parley(
                    context=add_person_tokens(history[:-2], last_speaker=1),
                    response=parley.response,  # y_exp
                )
                examples.append(example)
                history = []
            elif parley.context.startswith(NEWTOPIC):
                # Asked for new topic, got a first utterance
                # Begin new history
                history = [parley.response]
            elif parley.context.startswith(RAT_REQUEST):
                # Asked for rating, got one-word rating
                # Nothing to update in history
                pass
            elif CONTINUE in parley.context:
                # if response was negative, history will get blasted in EXP_REQUEST
                # if we're here, response was neutral/positive, so continue the history
                history.append(parley.context[parley.context.rindex(':') + 1 :])
                history.append(parley.response)
            else:
                # normal turn: maintain the history
                history.append(parley.context)
                history.append(parley.response)

    with open(opt['outfile'], 'w') as outfile:
        for ex in examples:
            outfile.write(json.dumps(ex.to_dict()) + '\n')

    print(
        f"Extracted {len(examples)} ratings out of {num_episodes} episodes "
        f"({num_parleys} parleys) and wrote them to {opt['outfile']} with "
        f"histsz == {opt['history_size']}."
    )


if __name__ == '__main__':
    opt = setup_args()
    main(opt)
