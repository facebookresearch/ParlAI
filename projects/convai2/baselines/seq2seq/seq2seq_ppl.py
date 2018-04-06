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
        if not hasattr(self, 'prev_enc'):
            self.prev_enc = None
            self.last_text = None
        if observation['text'] != self.last_text:
            self.prev_enc = None
            self.last_text = observation.get('text')
            self.observe(observation)

        obs = self.observation
        obs['eval_labels'] = [' '.join(partial_out)]
        batch = self.vectorize([obs])
        self.model.eval()
        self.model.longest_label = 1  # no need to predict farther ahead
        out = self.model(
            batch[0], # xs
            ys=(batch[1] if len(partial_out) > 0 else None),
            prev_enc=self.prev_enc)
        scores, self.prev_enc = out[1], out[3]
        # scores is bsz x seqlen x num_words, so select probs of current index
        assert len(partial_out) == scores.size(1) - 1
        # probs = F.softmax(scores.select(1, len(partial_out)), dim=1).squeeze().cpu()
        probs = F.log_softmax(scores.select(1, len(partial_out)), dim=1).squeeze().cpu()
        dist = {}
        for i in range(len(probs)):
            try:
                val = probs[i].item()
            except AttributeError:
                val = probs[i][0]
            # if val > 0:
            #     dist[self.dict[i]] = val
            dist[self.dict[i]] = val
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
            # if prob_true > 0:
            #     prob_true /= sum(probs.values())
            #     loss -= math.log(prob_true)
            loss -= prob_true
        with self._lock():
            self.metrics['total'] += 1
            self.metrics['loss'] += loss
            self.metrics['num_tokens'] += len(parsed)
        # if round_sigfigs(loss, 3) == 25.5:
        #     import pdb; pdb.set_trace()
        # with open('tmp_ppl_1', 'a') as write:
        #     write.write(str(round_sigfigs(loss, 3)) + '\n')

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
        model='projects.convai2.baselines.seq2seq.seq2seq_ppl:Seq2seqEntry',
        model_file='models:convai2/seq2seq/convai2_self_seq2seq_model',
        dict_file='models:convai2/seq2seq/dict_convai2_self',
        dict_lower=True,
        datatype='valid',
        batchsize=1,
        numthreads=60,
        no_cuda=True,
    )
    opt = parser.parse_args()
    opt['model_type'] = 'seq2seq'
    fnames = ['convai2_self_seq2seq_model.tgz', 'dict_convai2_self']
    download_models(opt, fnames, 'convai2')

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
