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
from torch.autograd import Variable
from parlai.agents.language_model.language_model import LanguageModelAgent
import torch.nn.functional as F
import math


class LanguageModelEntry(LanguageModelAgent):
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
        self.model.eval()
        if not hasattr(self, 'last_text'):
            self.last_text = None
            self.reset_next = False
        if observation['text'] != self.last_text:
            if self.reset_next:
                self.hidden = self.model.init_hidden(1)
                self.reset_next = False
            self.seen = False
            self.prev_enc = None
            self.last_text = observation.get('text')
            self.observe(observation)
        if observation['episode_done'] == True:
            self.reset_next = True
        else:
            self.reset_next = False

        obs = self.observation
        if len(partial_out) == 0:
            obs['eval_labels'] = ('PERSON2',)
        else:
            obs['eval_labels'] = (partial_out[-1],)
        data_list, targets_list, labels, valid_inds, y_lens = self.vectorize([obs], self.opt['seq_len'], False)
        data = data_list[0]
        targets = targets_list[0]

        if not self.seen:
            output, hidden = self.model(data.transpose(0,1), self.hidden)
            self.hidden = self.repackage_hidden(hidden)
            # feed in end tokens
            output, hidden = self.model(Variable(self.ends[:1].view(1,1)), self.hidden)
            # feed in person2 tokens
            output, hidden = self.model(targets.select(1,0).view(1, 1), self.hidden, no_pack=True)
            self.hidden = self.repackage_hidden(hidden)
            output_flat = output.view(-1, len(self.dict))
            self.seen = True
        else:
            output, hidden = self.model(targets.select(1,0).view(1, 1), self.hidden, no_pack=True)
            self.hidden = self.repackage_hidden(hidden)
            output_flat = output.view(-1, len(self.dict))

        probs = F.log_softmax(output_flat, dim=1).squeeze().cpu()
        probs = probs.tolist()
        dist = {}
        for i in range(len(probs)):
            dist[self.dict[i]] = probs[i]

        return dist


class PerplexityWorld(World):
    """Instead of just calling act/observe on each agent, this world just calls
    act on the teacher and then calls `next_word_probability` on the agent.
    The label for each example is parsed by the provided tokenizer, and then
    for each word in the parsed label the model is given the input and all of
    the tokens up to the current word and asked to predict the current word.
    The model must return a probability of any words it thinks are likely in
    the form of a dict mapping words to scores. If the scores do not sum to 1,
    they are normalized to do so. If the correct word is not present or has a
    probablity of zero, it will be assigned a probability of 1e-8.
    The API of the next_word_probability function which agents must implement
    is mentioned in the documentation for this file.
    """
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt)
        if shared:
            # Create agents based on shared data.
            self.task, self.agent, self.dict = create_agents_from_shared(shared['agents'])
            self.metrics = shared['metrics']
        else:
            if len(agents) != 3:
                raise RuntimeError('There must be exactly three agents.')
            if opt.get('batchsize', 1) > 1:
                raise RuntimeError('This world only works with bs=1. Try '
                                   'using multiple threads instead, nt>1.')
            self.task, self.agent, self.dict = agents
            if not hasattr(self.agent, 'next_word_probability'):
                raise RuntimeError('Agent must implement function '
                                   '`next_word_probability`.')
            self.metrics = {'total': 0, 'loss': 0.0, 'num_tokens': 0}
            if opt.get('numthreads', 1) > 1:
                self.metrics = SharedTable(self.metrics)
        self.agents = [self.task, self.agent, self.dict]
        self.acts = [None, None]

    def _lock(self):
        if hasattr(self.metrics, 'get_lock'):
            # use the shared_table's lock
            return self.metrics.get_lock()
        else:
            # otherwise do nothing
            return no_lock()

    def parley(self):
        action = self.task.act()
        self.acts[0] = action.copy()

        # hide labels from model
        labels = action.pop('eval_labels', action.pop('labels', None))
        if labels is None:
            # empty example, move on
            return

        parsed = self.dict.tokenize(labels[0])
        loss = 0
        for i in range(len(parsed)):
            if parsed[i] in self.dict:
                # only score words which are in the dictionary
                probs = self.agent.next_word_probability(action, parsed[:i])
                # get probability of correct answer, divide by total prob mass
                prob_true = probs.get(parsed[i], 0)
                if prob_true > 0:
                    prob_true /= sum(probs.values())
                    loss -= math.log(prob_true)
                else:
                    loss = float('inf')
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


def eval_ppl(parser):
    """Evaluates the the perplexity and f1 of a model (and hits@1 if model has
    ranking enabled.
    """
    dict_agent = build_dict()

    parser.set_defaults(task='convai2:self:no_cands', hide_labels=False)
    opt = parser.parse_args()

    # create agents
    agent = create_agent(opt)
    world = create_task(opt, [agent, dict_agent], default_world=PerplexityWorld)
    world.dict = dict_agent

    # set up logging
    log_time = Timer()
    tot_time = 0

    while not world.epoch_done():
        world.parley()  # process an example

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
