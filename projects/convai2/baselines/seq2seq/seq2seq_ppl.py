# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
import torch.nn.functional as F

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


from parlai.core.agents import create_agent, create_task_agent_from_taskname
from parlai.core.build_data import download_models
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer, round_sigfigs

import math

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
        batchsize=1,  # important for non-batch act below
    )
    opt = parser.parse_args()
    fnames = ['convai2_self_seq2seq_model.tgz', 'dict_convai2_self']
    download_models(opt, fnames, 'convai2')

    # create agents
    agent = create_agent(opt)
    task = create_task_agent_from_taskname(opt)[0]

    # set up loggin
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = Timer()
    tot_time = 0

    # Show some example dialogs:
    cnt = 0
    num_tokens = 0
    loss = 0
    while not task.epoch_done():
        cnt += 1
        action = task.act()
        parsed = agent.dict.tokenize(action['eval_labels'][0])
        for i in range(len(parsed) - 1):
            probs = agent.next_word_probability(action, parsed[:i])
            prob_true = probs.get(parsed[i], 0)
            loss -= prob_true
        num_tokens += len(parsed)

        if log_time.time() > log_every_n_secs:
            tot_time += log_time.time()
            print('{}s elapsed: {}%% complete, {} loss, {} ppl'.format(
                int(tot_time),
                round_sigfigs(cnt / task.num_examples(), 2),
                round_sigfigs(loss / num_tokens, 3),
                round_sigfigs(math.exp(loss / num_tokens), 4)))
            log_time.reset()
