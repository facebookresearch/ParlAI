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
    """Instead of calling act/observe on each agent, this world just calls
    act on the teacher and then calls `next_word_probability` on the agent.
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
        parsed.append('__END__')
        loss = 0
        for i in range(len(parsed)):
            probs = self.agent.next_word_probability(action, parsed[:i])
            # get probability of correct answer, divide by total prob mass
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
                m['num_tokens'] = self.metrics['num_tokens']
        return m


if __name__ == '__main__':
    parser = ParlaiParser(True, True)
    parser.add_argument('-vme', '--validation-max-exs', type=int, default=-1)
    parser.set_defaults(
        task='convai2:self',
        model='projects.convai2.baselines.language_model.language_model_ppl:LanguageModelEntry',
        model_file='/checkpoint/edinan/20180328/languagemodel_convai2_328/hs=512_esz=512_nl=3_lr=20_dr=0.3_bs=40/model.checkpoint',
        dict_file='/checkpoint/edinan/20180328/languagemodel_convai2_328/hs=512_esz=512_nl=3_lr=20_dr=0.3_bs=40/model.dict',
        datatype='test',
        batchsize=1,
    )
    opt = parser.parse_args()
    # opt['model_type'] = 'seq2seq'
    # fnames = ['convai2_self_seq2seq_model.tgz', 'dict_convai2_self']
    # download_models(opt, fnames, 'convai2')

    # create agents
    agent = create_agent(opt)
    world = create_task(opt, agent, default_world=PerplexityWorld)

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
            if opt['validation_max_exs'] > 0 and report['total'] >= opt['validation_max_exs']:
                break
    if world.epoch_done():
        print('EPOCH DONE')
    tot_time += log_time.time()
    final_report = world.report()
    print('{}s elapsed: {}'.format(int(tot_time), final_report))
