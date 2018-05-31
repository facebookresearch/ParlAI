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

from collections import deque, namedtuple
import random

Batch = namedtuple("Batch", [
    "text_vec",  # bsz x seqlen tensor containing the parsed text data
    "label_vec", # bsz x seqlen tensor containing the parsed label (one per batch row)
    "labels", # list of length bsz containing the selected label for each batch row (some datasets have multiple labels per input example)
    "valid_indices",  # list of length bsz containing the original indices of each example in the batch. we use these to map predictions back to their proper row, since e.g. we may sort examples by their length or some examples may be invalid.
])

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
        agent.add_argument('-histk', '--history-tokens', default=-1, type=int,
                           help='Number of past tokens to remember.')
        agent.add_argument('-histd', '--history-dialog', default=-1, type=int,
                           help='Number of past dialog examples to remember.')
        agent.add_argument('-histr', '--history-replies',
                           default='label_else_model', type=str,
                           choices=['none', 'model', 'label',
                                    'label_else_model'],
                           help='Keep replies in the history, or not.')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        if not shared:
            # Need to set up the model from scratch
            self.dict = DictionaryAgent(opt)
        else:
            # ... copy initialized data from shared table
            self.opt = shared['opt']
            self.dict = shared['dict']

        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        if self.use_cuda:
            print('[ Using CUDA ]')
            torch.cuda.device(opt['gpu'])

        self.NULL_IDX = self.dict[self.dict.null_token]
        self.END_IDX = self.dict[self.dict.end_token]
        self.START_IDX = self.dict[self.dict.start_token]

        self.history = {}
        self.history_tokens = opt['history_tokens']
        self.history_dialog = opt['history_dialog']
        self.history_replies = opt['history_replies']

    def share(self):
        shared = super().share()
        shared['opt'] = self.opt
        shared['dict'] = self.dict
        return shared

    def vectorize(self, obs, addEndIdx=True):
        """
        Converts 'text' and 'label'/'eval_label' field to vectors.

        :param obs: single observation from observe function
        :param addEndIdx: default True, adds the end token to each label
        """
        if 'text' not in obs:
            return obs
        # convert 'text' field to vector using self.txt2vec and then a tensor
        obs['text_vec'] = torch.LongTensor(self.dict.txt2vec(obs['text']))
        if self.use_cuda:
            obs['text_vec'] = obs['text_vec'].cuda()

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
            obs[label_type + "_vec"] = new_labels
        return obs

    def map_valid(self, obs_batch, sort=True, is_valid=lambda obs: 'text_vec' in obs):
        """Creates a batch of valid observations from an unchecked batch.
        Assumes each observation has been vectorized by vectorize function.

        Returns a namedtuple Batch. See original definition for in-depth
        explanation of each field.

        :param obs_batch: list of vectorized observations
        :param sort:      default True, orders the observations by length of vector
        :param is_valid:  default function that checks if 'text_vec' is in the
                          observation, determines if an observation is valid
        """
        if len(obs_batch) == 0:
            return Batch()

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch()

        valid_inds, exs = zip(*valid_obs)

        x_text = [ex['text_vec'] for ex in exs]
        x_lens = [ex.shape[0] for ex in x_text]

        if sort:
            ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])

            exs = [exs[k] for k in ind_sorted]
            valid_inds = [valid_inds[k] for k in ind_sorted]
            x_text = [x_text[k] for k in ind_sorted]

        padded_xs = torch.LongTensor(len(exs),
                                     max(x_lens)).fill_(self.NULL_IDX)
        if self.use_cuda:
            padded_xs = padded_xs.cuda()

        for i, ex in enumerate(x_text):
            padded_xs[i, :ex.shape[0]] = ex

        xs = padded_xs

        eval_labels_avail = any(['eval_labels_vec' in ex for ex in exs])
        labels_avail = any(['labels_vec' in ex for ex in exs])
        some_labels_avail = eval_labels_avail or labels_avail

        # set up the target tensors
        ys = None
        labels = None
        if some_labels_avail:
            # randomly select one of the labels to update on (if multiple)
            if labels_avail:
                field = 'labels'
            else:
                field = 'eval_labels'

            num_choices = [len(ex.get(field + "_vec", [])) for ex in exs]
            choices = [random.choice(range(num)) if num != 0 else -1
                       for num in num_choices]
            label_vecs = [ex[field + "_vec"][choices[i]]
                          if choices[i] != -1 else torch.LongTensor([])
                          for i, ex in enumerate(exs)]
            labels = [ex[field][choices[i]]
                      if choices[i] != -1 else ''
                      for i, ex in enumerate(exs)]
            y_lens = [y.shape[0] for y in label_vecs]
            padded_ys = torch.LongTensor(len(exs),
                                         max(y_lens)).fill_(self.NULL_IDX)
            if self.use_cuda:
                padded_ys = padded_ys.cuda()
            for i, y in enumerate(label_vecs):
                if y.shape[0] != 0:
                    padded_ys[i, :y.shape[0]] = y
            ys = padded_ys

        return Batch(xs, ys, labels, valid_inds)

    def unmap_valid(self, predictions, valid_inds, batch_size):
        """Re-order permuted predictions to the initial ordering, includes the
        empty observations.

        :param predictions: output of module's predict function
        :param valid_inds: original indices of the predictions
        :param batch_size: overall original size of batch
        """
        unpermuted = [None]*batch_size
        for pred, idx in zip(predictions, valid_inds):
            unpermuted[idx] = pred
        return unpermuted

    def maintain_dialog_history(self, observation, reply='',
                                useStartEndIndices=False,
                                splitSentences=False):
        """Keeps track of dialog history, up to a truncation length.
        Either includes replies from the labels, model, or not all using param 'replies'."""

        def parse(txt, splitSentences):
            if dict is not None:
                if splitSentences:
                    vec = [self.dict.txt2vec(t) for t in txt.split('\n')]
                else:
                    vec = self.dict.txt2vec(txt)
                return vec
            else:
                return [txt]

        allow_reply = True

        if 'dialog' not in self.history:
            self.history['dialog'] = deque(maxlen=self.history_tokens)
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

        obs = observation
        if 'text' in obs:
            if useStartEndIndices:
                obs['text'] = self.dict.end_token + ' ' + obs['text']
            self.history['dialog'].extend(parse(obs['text'], splitSentences))

        self.history['episode_done'] = obs['episode_done']
        labels = obs.get('labels', obs.get('eval_labels', None))
        if labels is not None:
            if useStartEndIndices:
                self.history['labels'] = [self.dict.start_token + ' ' + l for l in labels]
            else:
                self.history['labels'] = labels

        return self.history['dialog']
