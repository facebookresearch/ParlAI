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
    parser = ArgumentParser()
    parser.add_argument('-if', '--infile', type=str)
    parser.add_argument('-of', '--outfile', type=str)
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
        default='positive',
        help="A comma-separated list of ratings with positive label",
    )
    parser.add_argument(
        '-neg',
        '--negatives',
        type=str,
        default='negative',
        help="A comma-separated list of ratings with negative label",
    )
    opt = vars(parser.parse_args())

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
                history = []
            elif parley.context.startswith(NEWTOPIC):
                # Asked for new topic, got a first utterance
                # Begin new history
                history = [parley.response]
            elif parley.context.startswith(RAT_REQUEST):
                # Concatenate history and add speaker tokens as necessary
                # history_size refers to the total number of utterances
                # (history_size == 0 means predict sentiment from '__null__')
                # response that's being classified (so if history_size == 0 then
                # classify based only on the response w/o any extra context).
                # Note that the response being classified should always be preceded by
                # __p1__ (the human), not __p2__ (the bot).
                if opt['history_size'] < 0:
                    utterances = history
                elif opt['history_size'] == 0:
                    utterances = ['__null__']
                else:
                    utterances = history[-opt['history_size'] :]
                context = add_person_tokens(utterances, last_speaker=1)

                if parley.response in positives:
                    label = 1
                elif parley.response in negatives:
                    label = -1
                else:
                    label = 0

                if label:
                    example = Parley(context, label)
                    examples.append(example)

            elif CONTINUE in parley.context:
                # if response was negative, history will get blasted in EXP_REQUEST
                # if we're here, response was neutral/positive, so continue the history
                history.append(parley.context[parley.context.rindex(':') + 1 :])
                history.append(parley.response)
            else:
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
