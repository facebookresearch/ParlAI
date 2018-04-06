# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import create_agents_from_shared
from parlai.core.dict import DictionaryAgent
from parlai.core.thread_utils import SharedTable
from parlai.core.utils import round_sigfigs, no_lock
from parlai.core.worlds import World

import math


class PerplexityWorld(World):
    """Instead of calling act/observe on each agent, this world just calls
    act on the teacher and then calls `next_word_probability` on the agent.

    The label for each example is parsed by the provided tokenizer, and then
    for each word in the parsed label the model is given the input and all of
    the tokens up to the current word and asked to predict the current word.

    The model must return a probability of any words it thinks are likely in
    the form of a dict mapping words to scores. If the scores do not sum to 1,
    they are normalized to do so. If the correct word is not present or has a
    probablity of zero, it will be assigned a probability of 1e-8.
    """
    def __init__(self, opt, agents, shared=None, tokenizer=None):
        super().__init__(opt)
        if shared:
            # Create agents based on shared data.
            self.task, self.agent = create_agents_from_shared(shared['agents'])
            self.metrics = shared['metrics']
        else:
            if len(agents) != 2:
                raise RuntimeError('There must be exactly two agents.')
            if opt.get('batchsize', 1) > 1:
                raise RuntimeError('This world only works with bs=1. Try '
                                   'using multiple threads instead, nt>1.')
            self.task, self.agent = agents
            if not hasattr(self.agent, 'next_word_probability'):
                raise RuntimeError('Agent must implement function '
                                   '`next_word_probability`.')
            self.metrics = {'total': 0, 'loss': 0.0, 'num_tokens': 0}
            if opt.get('numthreads', 1) > 1:
                self.metrics = SharedTable(self.metrics)
        self.agents = [self.task, self.agent]
        self.acts = [None, None]

        if tokenizer is not None:
            self.tokenize = tokenizer
        else:
            self.tokenize = DictionaryAgent.split_tokenize
        if opt.get('dict_lower'):
            self._tokenize = self.tokenize
            self.tokenize = lambda t: self._tokenize(t.lower())

    def _lock(self):
        if hasattr(self.metrics, 'get_lock'):
            # use the shared_table's lock
            return self.metrics.get_lock()
        else:
            # otherwise do nothing
            return no_lock()

    def parley(self):
        action = self.task.act()
        self.acts[0] = action
        labels = action.get('eval_labels', action.get('labels'))
        if labels is None:
            return
        parsed = self.tokenize(labels[0])
        loss = 0
        losses = []
        for i in range(len(parsed)):
            probs = self.agent.next_word_probability(action, parsed[:i])
            # get probability of correct answer, divide by total prob mass
            prob_true = probs.get(parsed[i], 0)
            if prob_true > 0:
                prob_true /= sum(probs.values())
            else:
                prob_true = 1e-8  # for stability
            loss -= math.log(prob_true)
            losses.append(prob_true)
        with self._lock():
            self.metrics['total'] += 1
            self.metrics['loss'] += loss
            self.metrics['num_tokens'] += len(parsed)

    def epoch_done(self):
        return self.task.epoch_done()

    def num_examples(self):
        return self.task.num_examples()

    def num_episodes(self):
        return self.task.num_episodes()

    def share(self):
        shared = super().share()
        shared['metrics'] = self.metrics
        return shared

    def reset_metrics(self):
        with self._lock():
            self.metrics['total'] = 0
            self.metrics['loss'] = 0
            self.metrics['num_tokens'] = 0

    def report(self, compute_time=None):
        m = {}
        with self._lock():
            m['total'] = self.metrics['total']
            if m['total'] > 0:
                m['loss'] = round_sigfigs(self.metrics['loss'] / self.metrics['num_tokens'], 3)
                m['ppl'] = round_sigfigs(math.exp(self.metrics['loss'] / self.metrics['num_tokens']), 4)
        return m
