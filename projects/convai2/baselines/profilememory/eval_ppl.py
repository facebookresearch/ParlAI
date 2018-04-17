# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import create_agent, create_agents_from_shared
from parlai.core.build_data import download_models
from projects.convai2.build_dict import build_dict
from projects.convai2.eval_ppl import setup_args, eval_ppl
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser
from parlai.core.thread_utils import SharedTable
from parlai.core.utils import Timer, round_sigfigs, no_lock
from parlai.core.worlds import World, create_task
from projects.personachat.persona_seq2seq import PersonachatSeqseqAgentSplit
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torch
import math
import random


class ProfileMemoryEntry(PersonachatSeqseqAgentSplit):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if shared:
            self.probs = shared['probs']
        else:
            # default minimum probability mass for all tokens
            self.probs = {k: 1e-7 for k in build_dict().keys()}

    def share(self):
        shared = super().share()
        shared['probs'] = self.probs.copy()
        return shared

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
        if not hasattr(self, 'last_text'):
            self.last_text = None

        if observation['text'] != self.last_text:
            self.last_text = observation.get('text')
            self.observe(observation)

        obs = [self.observation]

        if len(partial_out) > 0:
            obs[0]['eval_labels'] = (' '.join(partial_out),)
        else:
            obs[0]['eval_labels'] = ['']

        xs, xs_persona, ys, labels, valid_inds, cands, valid_cands, zs, eval_labels = self.batchify(obs)

        self.encoder.train(mode=False)
        self.encoder_persona.train(mode=False)
        self.decoder.train(mode=False)

        encoder_output_persona, hidden_persona, guide_indices = self._encode_persona(xs_persona, ys, False)
        encoder_output, hidden = self._encode(xs, False)

        # next we use END as an input to kick off our decoder
        x = Variable(self.START_TENSOR)
        xe = self.lt(x).unsqueeze(1)
        xes = xe.expand(xe.size(0), 1, xe.size(2))

        attn_mask = Variable(xs_persona.data.sum(2).ne(0), requires_grad=False)

        output_lines = [[] for _ in range(1)]

        # we only ever want to generate the next token
        self.longest_label = 1

        attn_w_visual_tmp = []

        if zs is None:
            rnge = 1
        else:
            rnge = zs.size(1)

        for i in range(rnge):
            if type(self.decoder) == torch.nn.LSTM:
                h = hidden[0]
            if self.attention:
                if self.opt['personachat_attnsentlevel']:
                    h = self.attn_h2attn(h)
                output, attn_weights, attn_w_premask = self._apply_attention(xes, encoder_output_persona, h, attn_mask)
            else:
                output = xes

            output, hidden = self.decoder(output, hidden)

            new_output = F.dropout(output.squeeze(0), p=self.dropout, training=False)
            if self.sharelt:
                e = self.h2e(new_output)
                scores = self.e2o(e)[:,1:]
            else:
                scores = self.h2o(new_output)
            probs = F.softmax(scores)
            _max_score, preds = scores.max(1)

            if zs is not None:
                y = zs.select(1, i)
                # use the true token as the next input instead of predicted
                # this produces a biased prediction but better training
                if self.embshareonly_pm_dec:
                    xes = self.lt_enc(y).unsqueeze(0)
                else:
                    xes = self.lt(y).unsqueeze(0)
                xes = F.dropout(xes, p=self.dropout, training=False)
                for b in range(1):
                    # convert the output scores to tokens
                    token = self.v2t([(preds+1).data[b]])
                    output_lines[b].append(token)

        if random.random() < 0.01 and zs is not None:
            # sometimes output a prediction for debugging
            print('label:', self.dict.vec2txt(zs.data[0]),
                  '\nprediction:', output_lines[0][-1])

        # extract probability distribution
        probs = probs.tolist()
        dist = self.probs
        for i in range(0, len(probs[0])):
            if hasattr(probs, 'item'):
                dist[self.dict[i+1]] = probs[i].item()
            else:
                dist[self.dict[i+1]] = probs[0][i]
        return dist


if __name__ == '__main__':
    parser = setup_args()
    parser.add_argument('-n', '--num-examples', default=100000000)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    PersonachatSeqseqAgentSplit.add_cmdline_args(parser)

    parser.set_defaults(
        dict_file='models:convai2/profilememory/profilememory_convai2.dict',
        rank_candidates=True,
        model='projects.convai2.baselines.profilememory.eval_ppl:ProfileMemoryEntry',
        model_file='models:convai2/profilememory/profilememory_convai2_ppl_model',
    )

    opt = parser.parse_args()
    opt['model_type'] = 'profilememory'

    # build profile memory models
    fnames = ['profilememory_convai2_model',
              'profilememory_convai2_ppl_model',
              'profilememory_convai2.dict']
    download_models(opt, fnames, 'convai2', version='v2.0', use_model_type=True)

    eval_ppl(opt)
