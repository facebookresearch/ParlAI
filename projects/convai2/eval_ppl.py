# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Base script for running official ConvAI2 validation eval for perplexity.
This uses a the version of the dataset which does not contain candidates.
Leaderboard scores will be run in the same form but on a hidden test set.

The official vocabulary for the competition is based on using the
"split_tokenize" method on in the ParlAI core dictionary (parlai/core/dict.py)
and is built on the training and validation sets of the "convai2" task.
This dictionary contains a total of 19304 tokens. The test set contains some
tokens which are not in this dictionary--this tokens will not be provided, but
we will also *SKIP* calculating perplexity on these tokens. The model should
still produce a good guess for the remaining tokens in the sentence, so
handling unknown words or expanding the vocabulary with pre-trained or
multitasked embeddings are legitimate strategies that may or may not impact the
score of the models.

Note that this tokenizer will also be used during the perplexity evaluation:
the model will be asked to predict one word at a time according to this
tokenizer's parsing of the text.

This requires agents to implement the following function:

def next_word_probability(observation, partial_out):
    Return probability distribution over next words given an input and
    partial true output. This is used to calculate the per-word perplexity.

    Arguments:
    observation -- input observation dict
    partial_out -- list of previous "true" words

    Returns a dict, where each key is a word and each value is a probability
    score for that word. Unset keys assume a probability of zero.

    e.g.
    {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
"""

from parlai.core.agents import create_agent
from parlai.core.build_data import download_models
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer, round_sigfigs
from parlai.core.worlds import create_task
from .build_dict import build_dict
from .worlds import PerplexityWorld


def setup_args():
    parser = ParlaiParser(True, True)
    parser.set_defaults(task='convai2:self:no_cands')
    return parser

def eval_ppl(parser):
    """Evaluates the the perplexity and f1 of a model (and hits@1 if model has
    ranking enabled.
    """
    dict_agent = build_dict()

    opt = parser.parse_args()

    # create agents
    agent = create_agent(opt)
    world = create_task(opt, [agent, dict_agent], default_world=PerplexityWorld)
    world.dict = dict_agent

    # set up logging
    log_time = Timer()
    tot_time = 0

    # Show some example dialogs:
    while not world.epoch_done():
        world.parley()

        if log_time.time() > 1:  # log every 1 sec
            tot_time += log_time.time()
            report = world.report()
            print('{}s elapsed, {}%% complete, {}'.format(
                int(tot_time),
                round_sigfigs(report['total'] / world.num_examples() * 100, 2),
                report))
            log_time.reset()
    if world.epoch_done():
        print('EPOCH DONE')
    tot_time += log_time.time()
    final_report = world.report()
    print('{}s elapsed: {}'.format(int(tot_time), final_report))

if __name__ == '__main__':
    eval_ppl(setup_args())
