# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import create_agent, create_agents_from_shared
from parlai.core.build_data import download_models
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.thread_utils import SharedTable
from parlai.core.utils import Timer, round_sigfigs, no_lock
from parlai.core.worlds import World, create_task
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
import torch.nn.functional as F
import math


class Seq2seqEntry(Seq2seqAgent):
    def next_word_probability(self, observation, partial_out):
        """Return probability distribution over next words given an input and
        partial true output. This is used to calculate the per-word perplexity.

        Arguments:
        observation -- input observation dict
        partial_out -- previous "true" words

        Returns a dict, where each key is a word and each value is a probability
        score for that word. Unset keys assume a probability of zero.

        e.g.
        {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
        """
        # observe input
        observation['eval_labels'] = [' '.join(partial_out)]
        obs = self.observe(observation)
        batch = self.vectorize([obs])
        self.model.eval()
        _1, scores, _ = self.model(batch[0], batch[1] if len(partial_out) > 0 else None)
        probs = F.log_softmax(scores.squeeze()[-1].cpu(), dim=0)
        dist = {}
        for i in range(len(probs)):
            if hasattr(probs, 'item'):
                dist[self.dict[i]] = probs[i].item()
            else:
                dist[self.dict[i]] = probs[i][0]
        return dist


class PerplexityWorld(World):
    def __init__(self, opt, agents, shared=None, tokenizer=None):
        super().__init__(opt)
        if shared:
            # Create agents based on shared data.
            self.task, self.agent = create_agents_from_shared(shared['agents'])
            self.metrics = shared['metrics']
        else:
            if len(agents) != 2:
                raise RuntimeError('There must be exactly two agents.')
            self.task, self.agent = agents
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
            self.tokenize = lambda t: self.tokenize(t.lower())


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
        parsed = self.tokenize(action.get('eval_labels', action.get('labels'))[0])
        loss = 0
        for i in range(len(parsed) - 1):
            probs = self.agent.next_word_probability(action, parsed[:i])
            prob_true = probs.get(parsed[i], 0)
            loss -= prob_true
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


if __name__ == '__main__':
    parser = ParlaiParser(True, True)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=1)
    parser.set_defaults(
        task='convai2:self',
        model='projects.convai2.baselines.seq2seq.seq2seq_ppl:Seq2seqEntry',
        model_file='models:convai2/seq2seq/convai2_self_seq2seq_model',
        dict_file='models:convai2/seq2seq/dict_convai2_self',
        dict_lower=True,
        datatype='valid',
        batchsize=1,
        numthreads=2,
        no_cuda=True,
    )
    opt = parser.parse_args()
    fnames = ['convai2_self_seq2seq_model.tgz', 'dict_convai2_self']
    download_models(opt, fnames, 'convai2')

    # create agents
    agent = create_agent(opt)
    world = create_task(opt, agent, default_world=PerplexityWorld)

    # set up logging
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = Timer()
    tot_time = 0

    # Show some example dialogs:
    while not world.epoch_done():
        world.parley()

        if log_time.time() > log_every_n_secs:
            tot_time += log_time.time()
            report = world.report()
            print('{}s elapsed, {}%% complete, {}'.format(
                int(tot_time),
                round_sigfigs(report['total'] / world.num_examples(), 2),
                report))
            log_time.reset()
    print('EPOCH DONE')
    print(world.report)
