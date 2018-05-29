# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

try:
    import torch
except Exception as e:
    raise ModuleNotFoundError('Need to install Pytorch: go to pytorch.org')

import random


class TorchAgent(Agent):
    """Base Agent for all models which use Torch.

    This agent serves as a common framework for all ParlAI models which want
    to use PyTorch.
    """

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        if self.use_cuda:
            print('[ Using CUDA ]')
            torch.cuda.device(opt['gpu'])

        if not shared:
            # Need to set up the model from scratch
            self.dict = DictionaryAgent(opt)
        else:
            # ... copy initialized data from shared table
            self.opt = shared['opt']
            self.dict = shared['dict']

        self.NULL_IDX = self.dict[self.dict.null_token]
        self.END_IDX = self.dict[self.dict.end_token]
        self.START_IDX = self.dict[self.dict.start_token]

    def share(self):
        shared = super().share()
        shared['opt'] = self.opt
        shared['dict'] = self.dict
        return shared

    def vectorize(self, obs, use_cuda=True):
        """
        Converts 'text' and 'label'/'eval_label' field to vectors
        kwargs:
        -mode: which label should be passed through, is either train, eval or test
        -use_cuda: whether to use cuda
        """
        if 'text' not in obs:
            return obs
        # convert 'text' field to vector using self.txt2vec and then a tensor
        obs['text'] = torch.LongTensor(self.dict.txt2vec(obs['text']))
        if use_cuda:
            obs['text'] = obs['text'].cuda()

        if 'labels' in obs:
            new_labels = []
            for label in obs['labels']:
                vec_label = self.dict.txt2vec(label)
                vec_label.append(self.END_IDX)
                new_label = torch.LongTensor(vec_label)
                if use_cuda:
                    new_label = new_label.cuda()
                new_labels.append(new_label)
            obs['labels'] = new_labels
        elif 'eval_labels' in obs:
            new_labels = []
            for label in obs['eval_labels']:
                vec_label = self.dict.txt2vec(label)
                vec_label.append(self.END_IDX)
                new_label = torch.LongTensor(vec_label)
                if use_cuda:
                    new_label = new_label.cuda()
                new_labels.append(new_label)
            obs['eval_labels'] = new_labels

        return obs

    def permute(self, obs_batch, sort=True, is_valid=lambda obs: 'text' in obs, use_cuda=True):
        """Creates a batch of valid observations from an unchecked batch.
        Assumes each observation has been vectorized by vectorize function.
        """
        try:
            # valid examples and their indices
            valid_inds, exs = zip(*[(i, ex) for i, ex in
                                    enumerate(obs_batch) if is_valid(ex)])
        except ValueError:
            # zero examples to process in this batch, so zip failed to unpack
            return None, None, None, None, None, None
        x_text = [ex['text'] for ex in exs]
        x_lens = [ex.shape[0] for ex in x_text]

        if sorted:
            ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])

            exs = [exs[k] for k in ind_sorted]
            valid_inds = [valid_inds[k] for k in ind_sorted]
            x_text = [x_text[k] for k in ind_sorted]
            end_idxs = [x_lens[k] for k in ind_sorted]

        padded_xs = torch.LongTensor(len(exs),
                                     max(x_lens)).fill_(self.NULL_IDX)
        if use_cuda:
            padded_xs = padded_xs.cuda()

        for i, ex in enumerate(x_text):
            padded_xs[i, :ex.shape[0]] = ex

        xs = padded_xs

        eval_labels_avail = any(['eval_labels' in ex for ex in exs])
        labels_avail = any(['labels' in ex for ex in exs])
        some_labels_avail = eval_labels_avail or labels_avail

        # set up the target tensors
        ys = None
        labels = None
        y_lens = None
        if some_labels_avail:
            # randomly select one of the labels to update on (if multiple)
            if labels_avail:
                labels = [random.choice(ex.get('labels', [''])) for ex in exs]
            else:
                labels = [random.choice(ex.get('eval_labels', [''])) for ex in exs]
            y_lens = [y.shape[0] for y in labels]
            padded_ys = torch.LongTensor(len(exs),
                                         max(y_lens)).fill_(self.NULL_IDX)
            if use_cuda:
                padded_ys = padded_ys.cuda()
            for i, y in enumerate(labels):
                padded_ys[i, :y.shape[0]] = y
            ys = padded_ys
        return xs, ys, labels, valid_inds, end_idxs, y_lens

    def unpermute(self, predictions, valid_inds, batch_size):
        """Re-order permuted predictions to the initial ordering.
        """
        unpermuted = [None]*batch_size
        for pred, idx in zip(predictions, valid_inds):
            unpermuted[idx] = pred
        return unpermuted
