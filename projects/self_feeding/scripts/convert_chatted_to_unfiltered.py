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
from parlai.utils.io import PathManager

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
        '-mode',
        '--mode',
        type=str,
        choices=['bot', 'human'],
        help="Whether to use as target responses what the bot said, "
        "human said, or both",
    )
    parser.add_argument(
        '-fm',
        '--filter-mistake',
        type=int,
        default=0,
        help="If true, toss bot examples where the bot made a mistake",
    )
    parser.add_argument(
        '-fa',
        '--filter-accusation',
        type=int,
        default=0,
        help="If true, toss human examples where the human is "
        "expressing dissatisfaction",
    )
    parser.add_argument(
        '-histsz',
        '--history-size',
        type=int,
        default=-1,
        help="The number of turns to include in the prompt.",
    )
    opt = vars(parser.parse_args())

    if opt['filter_accusation']:
        assert not opt['outfile'].endswith('unfiltered')
    else:
        assert opt['outfile'].endswith('unfiltered')

    if opt['mode'] == 'all':
        raise Exception("Double check the logic for extracting bot comments first...")

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

    num_episodes = 0
    num_parleys = 0
    for episode in extract_parlai_episodes(opt['infile']):
        num_episodes += 1
        history = []
        for parley in episode:
            num_parleys += 1
            # Update history (not including stock control flow responses)
            if parley.context.startswith(INITIAL_PROMPT) or parley.context.startswith(
                NEWTOPIC
            ):
                # a prompt, first utterance
                # Begin history
                history = [parley.response]
                # NOTE: we now allow these one-utterance episodes to be examples
                # continue
            elif parley.context.startswith(EXP_REQUEST) or parley.context.startswith(
                RAT_REQUEST
            ):
                # If 'filter_accusation' is on and the last example added was a human,
                # toss the previous example, which is when the human expressed
                # dissatisfaction
                if (
                    opt['mode'] == 'human'
                    and opt['filter_accusation']
                    and parley.context.startswith(EXP_REQUEST)
                    and len(examples) > 0
                ):
                    examples.pop()
                # If 'filter_mistake' is on and the last example in the queue was a bot,
                # toss it too, since that's when the bot messed up
                if (
                    opt['mode'] == 'bot'
                    and opt['filter_mistake']
                    and parley.context.startswith(EXP_REQUEST)
                    and len(examples) > 0
                ):
                    examples.pop()

                # Asked for y_exp or rating, got it
                # Messed up, so blast history
                history = []
                continue
            elif CONTINUE in parley.context:
                # if response was negative, history will get blasted in EXP_REQUEST
                # if we're here, response was neutral/positive, so continue the history
                history.append(parley.context[parley.context.rindex(':') + 1 :])
                history.append(parley.response)
            else:
                # normal turn: maintain the history
                history.append(parley.context)
                history.append(parley.response)

            if opt['mode'] in ['bot'] and len(history) >= 2:
                if len(history) == 2:
                    example = Parley(context='__null__', response=history[0])
                else:
                    example = Parley(
                        context=add_person_tokens(history[:-2], last_speaker=1),
                        response=history[-2],  # What the bot said
                    )
                examples.append(example)

            if opt['mode'] in ['human']:
                if len(history) == 1:
                    example = Parley(context='__null__', response=history[0])
                else:
                    example = Parley(
                        # this is not technically true:
                        # the last speaker was the bot (__p2__),
                        # not the human (__p1__), but in all our data, __p1__ is always
                        # the speaking partner of the learner
                        context=add_person_tokens(history[:-1], last_speaker=1),
                        response=history[-1],  # What the bot said
                    )
                examples.append(example)

    with PathManager.open(opt['outfile'], 'w') as outfile:
        for ex in examples:
            outfile.write(json.dumps(ex.to_dict()) + '\n')

    print(
        f"Extracted {len(examples)} examples out of {num_episodes} episodes "
        f"({num_parleys} parleys) and wrote them to {opt['outfile']} with "
        f"histsz == {opt['history_size']}."
    )


if __name__ == '__main__':
    opt = setup_args()
    main(opt)
