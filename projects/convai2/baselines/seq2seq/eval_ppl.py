# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for ppl metric.
This seq2seq model was trained on convai2:self.
"""
from parlai.core.build_data import download_models
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser, modelzoo_path
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
from projects.convai2.build_dict import build_dict
from projects.convai2.eval_ppl import setup_args, eval_ppl
import torch.nn.functional as F


class Seq2seqEntry(Seq2seqAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if shared:
            self.probs = shared['probs']
        else:
            # default minimum probability mass for all tokens
            self.probs = {k: 1e-7 for k in build_dict().keys()}
        self.prev_enc = None
        self.prev_vec = None

    def share(self):
        shared = super().share()
        shared['probs'] = self.probs.copy()
        return shared

    def next_word_probability(self, partial_out):
        """Return probability distribution over next words given an input and
        partial true output. This is used to calculate the per-word perplexity.

        Arguments:
        observation -- input observation dict
        partial_out -- list of previous "true" words

        Returns a dict, where each key is a word and each value is a probability
        score for that word. Unset keys assume a probability of zero.

        e.g.
        {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
        """
        obs = self.observation
        obs['eval_labels'] = [' '.join(partial_out)]
        batch = self.vectorize([obs])
        if self.prev_enc is not None and batch[0].shape[1] != self.prev_enc[0].shape[1]:
            self.prev_enc = None  # reset prev_enc

        self.model.eval()
        self.model.longest_label = 1  # no need to predict farther ahead
        out = self.model(
            batch[0], # xs
            ys=(batch[1] if len(partial_out) > 0 else None),
            prev_enc=self.prev_enc)
        scores, self.prev_enc = out[1], out[-1]
        # scores is bsz x seqlen x num_words, so select probs of current index
        assert len(partial_out) == scores.size(1) - 1
        probs = F.softmax(scores.select(1, len(partial_out)), dim=1).squeeze()
        dist = self.probs
        for i in range(len(probs)):
            try:
                val = probs[i].item()
            except AttributeError:
                val = probs[i][0]
            dist[self.dict[i]] = val
        return dist


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        model='projects.convai2.baselines.seq2seq.eval_ppl:Seq2seqEntry',
        model_file='models:convai2/seq2seq/convai2_self_seq2seq_model',
        dict_file='models:convai2/seq2seq/convai2_self_seq2seq_model.dict',
        dict_lower=True,
        batchsize=1,
        numthreads=60,
        no_cuda=True,
    )
    opt = parser.parse_args()
    if opt.get('model_file', '').startswith('models:convai2'):
        opt['model_type'] = 'seq2seq'
        fnames = ['convai2_self_seq2seq_model.tgz',
                  'convai2_self_seq2seq_model.dict',
                  'convai2_self_seq2seq_model.opt']
        download_models(opt, fnames, 'convai2', version='v3.0')
    eval_ppl(opt)
