#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.build_data import download_models
from projects.convai2.build_dict import build_dict
from projects.convai2.eval_ppl import setup_args, eval_ppl
from torch.autograd import Variable
from parlai.agents.language_model.language_model import LanguageModelAgent
import torch.nn.functional as F


class LanguageModelEntry(LanguageModelAgent):
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

    def next_word_probability(self, partial_out):
        """Return probability distribution over next words given an partial
        true output. This is used to calculate the per-word perplexity.
        Arguments:
        partial_out -- previous "true" words
        Returns a dict, where each key is a word and each value is a probability
        score for that word. Unset keys assume a probability of zero.
        e.g.
        {'text': 'Run test program.'}, ['hello'] => {'world': 1.0}
        """
        obs = self.observation
        if not hasattr(self, 'last_text'):
            self.last_text = None
            self.reset_next = False
        if obs['text'] != self.last_text:
            if self.reset_next:
                # reset hidden state for new episodes
                self.hidden = self.model.init_hidden(1)
                self.reset_next = False
            self.seen = False
            self.last_text = obs.get('text')

        self.model.eval()

        if obs['episode_done'] == True:
            self.reset_next = True
        else:
            self.reset_next = False


        if len(partial_out) == 0:
            # first observe 'PERSON2' token
            obs['eval_labels'] = ('PERSON2',)
        else:
            # feed in words one at a time
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

        # get probabilites for all words
        probs = F.softmax(output_flat, dim=1).squeeze().cpu()
        probs = probs.tolist()
        dist = self.probs
        for i in range(len(probs)):
            dist[self.dict[i]] = probs[i]

        return dist


if __name__ == '__main__':
    parser = setup_args()
    parser.add_argument('-vme', '--validation-max-exs', type=int, default=-1)
    parser.set_params(
        model='projects.convai2.baselines.language_model.eval_ppl:LanguageModelEntry',
        model_file='models:convai2/language_model/model',
        dict_file='models:convai2/language_model/model.dict',
        batchsize=1,
    )
    opt = parser.parse_args()
    opt['model_type'] = 'language_model'
    fnames = ['model', 'model.dict', 'model.opt']
    download_models(opt, fnames, 'convai2', version='v2.0',
                    use_model_type=True)
    eval_ppl(opt)
