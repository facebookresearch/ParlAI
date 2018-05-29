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

from collections import deque
import random


class TorchAgent(Agent):
    """Base Agent for all models which use Torch.

    This agent serves as a common framework for all ParlAI models which want
    to use PyTorch.
    """

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('TorchAgent Arguments')
        agent.add_argument('-hist', '--history-length', default=10, type=int,
                           help='Number of past tokens to remember. ')
        agent.add_argument('-histr', '--history-replies',
                           default='label_else_model', type=str,
                           choices=['none', 'model', 'label',
                                    'label_else_model'],
                           help='Keep replies in the history, or not.')

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

        self.history = {}
        self.history_length = opt['history_length']
        self.history_replies = opt['history_replies']

    def share(self):
        shared = super().share()
        shared['opt'] = self.opt
        shared['dict'] = self.dict
        return shared

    def vectorize(self, obs, addEndIdx=True):
        """
        Converts 'text' and 'label'/'eval_label' field to vectors
        """
        if 'text' not in obs:
            return obs
        # convert 'text' field to vector using self.txt2vec and then a tensor
        obs['text'] = torch.LongTensor(self.dict.txt2vec(obs['text']))
        if self.use_cuda:
            obs['text'] = obs['text'].cuda()

        label_type = None
        if 'labels' in obs:
            label_type = 'labels'
        elif 'eval_labels' in obs:
            label_type = 'eval_labels'

        if label_type is not None:
            new_labels = []
            for label in obs[label_type]:
                vec_label = self.dict.txt2vec(label)
                if addEndIdx:
                    vec_label.append(self.END_IDX)
                new_label = torch.LongTensor(vec_label)
                if self.use_cuda:
                    new_label = new_label.cuda()
                new_labels.append(new_label)
            obs[label_type] = new_labels

        return obs

    def permute(self, obs_batch, sort=True, is_valid=lambda obs: 'text' in obs):
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
        if self.use_cuda:
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
            if self.use_cuda:
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

    def maintain_dialog_history(self, observation, reply='',
                                useStartEndIndices=True,
                                splitSentences=False):
        """Keeps track of dialog history, up to a truncation length.
        Either includes replies from the labels, model, or not all using param 'replies'."""

        def parse(txt, splitSentences=False):
            if dict is not None:
                if splitSentences:
                    vec = [self.dict.txt2vec(t) for t in txt.split('\n')]
                else:
                    vec = self.dict.txt2vec(txt)
                if useStartEndIndices:
                    parsed_x = deque([self.START_IDX])
                    parsed_x.extend(vec)
                    parsed_x.append(self.END_IDX)
                    return parsed_x
                else:
                    return vec
            else:
                return [txt]

        allow_reply = True

        if 'dialog' not in self.history:
            self.history['dialog'] = deque(maxlen=self.history_length)
            self.history['episode_done'] = False
            self.history['labels'] = []

        if self.history['episode_done']:
            self.history['dialog'].clear()
            self.history['labels'] = []
            allow_reply = False
            self.history['episode_done'] = False

        if self.history_replies != 'none' and allow_reply:
            if self.history_replies == 'model' or \
               (self.history_replies == 'label_else_model' and len(
                                                self.history['labels']) == 0):
                if reply != '':
                    self.history['dialog'].extend(parse(reply))
            elif len(self.history['labels']) > 0:
                r = self.history['labels'][0]
                self.history['dialog'].extend(parse(r, splitSentences))
        if 'text' in observation:
            self.history['dialog'].extend(parse(observation['text'], splitSentences))

        self.history['episode_done'] = observation['episode_done']
        if 'labels' in observation:
            self.history['labels'] = observation['labels']
        elif 'eval_labels' in observation:
            self.history['labels'] = observation['eval_labels']
        return self.history['dialog']
