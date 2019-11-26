#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains code to compute NIDF measures, used for specificity control.
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from collections import Counter
import os
import math
import pickle


CONTROLLABLE_DIR = 'controllable_dialogue'
PARLAI_FORMAT_DIR = 'controllable_dialogue/ConvAI2_parlaiformat'


def get_word_counts(opt, count_inputs):
    """
    Goes through the dataset specified in opt and gets word counts.

    Inputs:
      count_inputs: If True, include both input and reply when counting words
        and utterances. Otherwise, only include reply text.

    Returns:
      word_counter_per_sent: a Counter mapping each word to the number of
        utterances in which it appears.
      num_sents: int. number of utterances counted
    """
    # Create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    # Count word frequency for all words in dataset
    word_counter_per_sent = Counter()
    num_sents = 0
    count = 0
    log_timer = TimeLogger()
    while True:
        count += 1

        world.parley()
        reply = world.acts[0].get('labels', world.acts[0].get('eval_labels'))[0]

        words = reply.split()
        words_no_dups = list(set(words))  # remove duplicates
        word_counter_per_sent.update(words_no_dups)
        num_sents += 1

        # Optionally count words in input text
        if count_inputs:
            input = world.acts[0]['text']
            input = input.split('\n')[-1]  # e.g. in ConvAI2, this removes persona
            words = input.split()
            words_no_dups = list(set(words))  # remove duplicates
            word_counter_per_sent.update(words_no_dups)
            num_sents += 1

        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(world.total_parleys, world.num_examples())
            print(text)

        if world.epoch_done():
            print('EPOCH DONE')
            break

    return word_counter_per_sent, num_sents


def learn_nidf(opt):
    """
    Go through ConvAI2 and Twitter data, and count word frequences.

    Save word2count.pkl, which contains word2count, and total num_sents. These are both
    needed to calculate NIDF later.
    """

    opt['log_every_n_secs'] = 2

    print('Counting words in Twitter train set...')
    opt['datatype'] = 'train:ordered'
    opt['task'] = 'twitter'
    wc1, ns1 = get_word_counts(opt, count_inputs=True)

    print('Counting words in Twitter val set...')
    opt['datatype'] = 'valid'
    opt['task'] = 'twitter'
    wc2, ns2 = get_word_counts(opt, count_inputs=True)

    opt['task'] = 'fromfile:parlaiformat'

    print('Counting words in ConvAI2 train set...')
    opt['datatype'] = 'train:ordered'
    opt['fromfile_datapath'] = os.path.join(
        opt['datapath'], PARLAI_FORMAT_DIR, 'train.txt'
    )
    # Don't include inputs because ConvAI2 train set reverses every conversation
    wc3, ns3 = get_word_counts(opt, count_inputs=False)

    print('Counting words in ConvAI2 val set...')
    opt['datatype'] = 'valid'
    opt['fromfile_datapath'] = os.path.join(
        opt['datapath'], PARLAI_FORMAT_DIR, 'valid.txt'
    )
    wc4, ns4 = get_word_counts(opt, count_inputs=True)

    # Merge word counts
    word_counter = Counter()
    for wc in [wc1, wc2, wc3, wc4]:
        for word, count in wc.items():
            word_counter[word] += count
    num_sents = ns1 + ns2 + ns3 + ns4

    # Write word2count and num_sents to file
    word2count_fp = os.path.join(opt['datapath'], CONTROLLABLE_DIR, 'word2count.pkl')
    print("Saving word count stats to %s..." % word2count_fp)
    data = {"word2count": word_counter, "num_sents": num_sents}
    with open(word2count_fp, "wb") as f:
        pickle.dump(data, f)


def load_word2nidf(opt):
    """
    Loads word count stats from word2count.pkl file in data/controllable_dialogue,
    computes NIDF for all words, and returns the word2nidf dictionary.

    Returns:
      word2nidf: dict mapping words to their NIDF score (float between 0 and 1)
    """
    word2count_fp = os.path.join(opt['datapath'], CONTROLLABLE_DIR, 'word2count.pkl')
    print("Loading word count stats from %s..." % word2count_fp)
    with open(word2count_fp, "rb") as f:
        data = pickle.load(f)
    num_sents = data['num_sents']
    print('num_sents: ', num_sents)
    word2count = data['word2count']
    min_c = min(word2count.values())  # max count
    max_c = max(word2count.values())  # min count
    word2nidf = {
        w: (math.log(max_c) - math.log(c)) / (math.log(max_c) - math.log(min_c))
        for w, c in word2count.items()
    }
    print("Done loading word2nidf dictionary.")
    return word2nidf


if __name__ == '__main__':
    parser = ParlaiParser()
    opt = parser.parse_args()
    learn_nidf(opt)
